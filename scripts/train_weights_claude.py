import os, glob
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from network_pkg.admp_core.model.weight_net         import WeightMLP
from network_pkg.admp_core.core.basis.make_rbf      import make_rbf
from network_pkg.admp_core.core.dmp.dmp2d           import fit_weights_2d


# --------- Fixed Utils (Normalization / Resample) -----------
def resample_norm_safe(y, T, eps=1e-6):
    """정규화된 리샘플링 - 일관된 구현"""
    idx = np.linspace(0, len(y) - 1, T).astype(int)
    z = y[idx]
    
    # 🔧 수정: 항상 z[-1] - z[0] 사용 (start-goal 거리)
    d = np.linalg.norm(z[-1] - z[0])
    if d < eps:
        # fallback: 궤적 전체 길이 사용
        diffs = np.diff(z, axis=0)
        L = np.sum(np.linalg.norm(diffs, axis=1))
        d = max(L, eps)
    
    # 정규화: start를 원점으로, start-goal 거리를 1로
    normalized = (z - z[0]) / d
    return normalized.astype(np.float32)


def normalize_weights(wx, wy):
    """가중치 정규화 - 학습 안정성 향상"""
    # Z-score 정규화
    wx_mean, wx_std = np.mean(wx), np.std(wx)
    wy_mean, wy_std = np.mean(wy), np.std(wy)
    
    wx_norm = (wx - wx_mean) / (wx_std + 1e-8)
    wy_norm = (wy - wy_mean) / (wy_std + 1e-8)
    
    return wx_norm, wy_norm, (wx_mean, wx_std), (wy_mean, wy_std)


# ----------- Fixed Dataset --------------
class TrajWeightDataset(Dataset):
    def __init__(self, root="dataset/testbank_K128", T_in=200, T_out=600, K_fix=128, normalize_weights=True):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No npz files under : {root}")
        
        self.T_in = T_in
        self.T_out = T_out
        self.K_fix = K_fix
        self.normalize_weights = normalize_weights

        # RBF creation (for fixed K)
        self.c, self.h = make_rbf(K_fix)
        self.dt = 1.0 / (self.T_out - 1)
        
        # 🆕 가중치 통계 수집 (정규화용)
        if self.normalize_weights:
            self._collect_weight_stats()

    def _collect_weight_stats(self):
        """전체 데이터셋의 가중치 통계 수집"""
        print("Collecting weight statistics for normalization...")
        all_wx, all_wy = [], []
        
        for i in range(min(len(self.files), 100)):  # 샘플링으로 빠르게 추정
            data = np.load(self.files[i])
            y = data["traj"].astype(np.float64)
            y_fit = resample_norm_safe(y, self.T_out).astype(np.float64)
            wx, wy, _, _ = fit_weights_2d(y_fit, self.dt, self.c, self.h)
            all_wx.extend(wx.flatten())
            all_wy.extend(wy.flatten())
        
        self.wx_mean = np.mean(all_wx)
        self.wx_std = np.std(all_wx)
        self.wy_mean = np.mean(all_wy)
        self.wy_std = np.std(all_wy)
        
        print(f"Weight stats - wx: μ={self.wx_mean:.4f}, σ={self.wx_std:.4f}")
        print(f"                wy: μ={self.wy_mean:.4f}, σ={self.wy_std:.4f}")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            y = data["traj"].astype(np.float64)

            # 🔧 수정: 일관된 정규화 사용
            y_in = resample_norm_safe(y, self.T_in)
            y_fit = resample_norm_safe(y, self.T_out).astype(np.float64)
            
            # 가중치 피팅
            wx, wy, y0, g = fit_weights_2d(y_fit, self.dt, self.c, self.h)
            
            # 🆕 가중치 정규화
            if self.normalize_weights:
                wx = (wx - self.wx_mean) / (self.wx_std + 1e-8)
                wy = (wy - self.wy_mean) / (self.wy_std + 1e-8)
            
            # 텐서 변환
            y_in_t = torch.from_numpy(y_in)                    # (T_in, 2)
            wx_t = torch.from_numpy(wx.astype(np.float32))     # (K_fix,)
            wy_t = torch.from_numpy(wy.astype(np.float32))     # (K_fix,)
            
            return y_in_t, wx_t, wy_t
            
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            # fallback: 다른 샘플 반환
            return self.__getitem__((idx + 1) % len(self.files))


# ----------- Enhanced Train Loop --------------
def train(root="dataset/train", epochs=20, bs=64, lr=1e-3, T_in=200, T_out=600, K_fix=128, num_workers=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 🆕 가중치 정규화 활성화
    ds = TrajWeightDataset(root=root, T_in=T_in, T_out=T_out, K_fix=K_fix, normalize_weights=True)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers)

    model = WeightMLP(T_in=T_in, K_fix=K_fix).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 🆕 L2 정규화
    crit = nn.MSELoss()

    # 🆕 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5, verbose=True)

    model.train()
    best_loss = float('inf')
    
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        num_samples = 0
        
        for batch_idx, (y_in, wx_gt, wy_gt) in enumerate(dl):
            y_in = y_in.to(device)      # (B, T_in, 2)
            wx_gt = wx_gt.to(device)    # (B, K_fix)
            wy_gt = wy_gt.to(device)    # (B, K_fix)

            # Forward pass
            wx_pred, wy_pred = model(y_in)
            
            # 🔧 수정: 가중치별 loss 정규화
            loss_wx = crit(wx_pred, wx_gt)
            loss_wy = crit(wy_pred, wy_gt)
            loss = (loss_wx + loss_wy) / 2.0  # 평균으로 정규화

            # Backward pass
            opt.zero_grad()
            loss.backward()
            
            # 🆕 gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()

            tot_loss += loss.item() * y_in.size(0)
            num_samples += y_in.size(0)
            
            # 🆕 진행상황 출력
            if batch_idx % 10 == 0:
                print(f"[ep {ep}] batch {batch_idx}/{len(dl)}, loss = {loss.item():.6f}")

        avg_loss = tot_loss / num_samples
        print(f"[ep {ep}] avg_loss = {avg_loss:.6f}")
        
        # 🆕 스케줄러 업데이트 & 모델 저장
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            Path("artifacts").mkdir(exist_ok=True)
            out_path = f"artifacts/weight_mlp_K{K_fix}_best.pth"
            
            # 🆕 정규화 통계도 함께 저장
            save_dict = {
                'model_state_dict': model.state_dict(),
                'wx_mean': ds.wx_mean,
                'wx_std': ds.wx_std,
                'wy_mean': ds.wy_mean,
                'wy_std': ds.wy_std,
                'K_fix': K_fix,
                'T_in': T_in,
                'T_out': T_out
            }
            torch.save(save_dict, out_path)
            print(f"✓ Best model saved: {out_path}")

    print(f"Training completed! Best loss: {best_loss:.6f}")


# -------- Main --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="dataset/train")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T_in", type=int, default=200)
    ap.add_argument("--T_out", type=int, default=600)
    ap.add_argument("--Kfix", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    train(root=args.root, epochs=args.epochs, bs=args.bs, lr=args.lr,
          T_in=args.T_in, T_out=args.T_out, K_fix=args.Kfix, num_workers=args.workers)