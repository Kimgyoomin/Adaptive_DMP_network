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



# --------- Utils (Normalization / Resample) -----------
def resample_norm_safe(y, T, eps=1e-6):
    idx = np.linspace(0, len(y) - 1, T).astype(int)
    z = y[idx]
    d = np.linalg.norm(z[-1] - z[0])
    # if d < 1e-3:
    if d < eps:
        diffs = np.diff(z, axis=0)
        L = np.sum(np.linalg.norm(diffs, axis=1))
        d = max(L, eps)
    # Normalization : start  to (0,0) start-goal dist to 1
    normalized = (z - z[0]) / d
    # return ((z - z[0]) / d).astype(np.float32)
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------
# def normalize_weights(wx, wy):
#     """가중치 정규화 - 학습 안정성 향상"""
#     # Z-score 정규화
#     wx_mean, wx_std = np.mean(wx), np.std(wx)
#     wy_mean, wy_std = np.mean(wy), np.std(wy)
    
#     wx_norm = (wx - wx_mean) / (wx_std + 1e-8)
#     wy_norm = (wy - wy_mean) / (wy_std + 1e-8)
    
#     return wx_norm, wy_norm, (wx_mean, wx_std), (wy_mean, wy_std)

# ----------- Dataset --------------
class TrajWeightDataset(Dataset):
    """
    for each .npz, read 'traj' ;
    - input     : (T_in, 2) which normalized resampled by T_in
    - output    : wx, wy for fixed K_fix -> derived from OLS
    """
    def __init__(self, root="dataset/train", T_in=200, T_out=600, K_fix=128):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No npz files under : {root}")
        self.T_in   = T_in
        self.T_out  = T_out
        self.K_fix  = K_fix

        # RBF creation (for fixed K)
        self.c, self.h = make_rbf(K_fix)
        self.dt = 1.0 / (self.T_out - 1)

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        y = data["traj"].astype(np.float64)     # (T_raw, 2)

        # Input (Normalization, T_in)
        y_in = resample_norm_safe(y, self.T_in)     # (T_in, 2), float 32

        # Label Creation(Normalization, T_out)
        y_fit = resample_norm_safe(y, self.T_out).astype(np.float64)    #(T_out, 2)
        wx, wy, y0, g = fit_weights_2d(y_fit, self.dt, self.c, self.h)  # numpy float64

        # torch tensor translate
        y_int_t = torch.from_numpy(y_in)                    # (T_in, 2) float 32
        wx_t    = torch.from_numpy(wx.astype(np.float32))   # (K_fix, )
        wy_t    = torch.from_numpy(wy.astype(np.float32))   # (K_fix, )
        return y_int_t, wx_t, wy_t


# ----------- Train Loop --------------
def train(root = "dataset/train", epochs=20, bs=64, lr=1e-3, T_in=200, T_out=600, K_fix=128, num_workers=0):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    ds      = TrajWeightDataset(root=root, T_in=T_in, T_out=T_out, K_fix=K_fix)
    dl      = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers)

    model   = WeightMLP(T_in=T_in, K_fix=K_fix).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    crit    = nn.MSELoss()


    model.train()
    for ep in range(1, epochs+1):
        tot = 0.0
        for y_in, wx_gt, wy_gt, in dl:
            y_in    = y_in.to(device)           # (B, T_in, 2)
            wx_gt   = wx_gt.to(device)          # (B, K_fix)
            wy_gt   = wy_gt.to(device)          


            wx_pred, wy_pred = model(y_in)      # (B, K_fix), (B, K_fix)
            loss = crit(wx_pred, wx_gt) + crit(wy_pred, wy_gt)


            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item() * y_in.size(0)

        print(f"[ep {ep}] loss = {tot / len(ds):.6f}")

    Path("artifacts").mkdir(exist_ok=True)
    out_path = f"artifacts/weight_mlp_K{K_fix}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"saved {out_path}")



# -------- Main --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    type=str,   default="dataset/train")
    ap.add_argument("--epochs",  type=int,   default=20)
    ap.add_argument("--bs",      type=int,   default=64)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--T_in",    type=int,   default=200)
    ap.add_argument("--T_out",   type=int,   default=600)
    ap.add_argument("--Kfix",    type=int,   default=128)
    ap.add_argument("--workers", type=int,   default=0)
    args = ap.parse_args()

    train(root=args.root, epochs=args.epochs, bs=args.bs, lr=args.lr,
          T_in=args.T_in, T_out=args.T_out, K_fix=args.Kfix, num_workers=args.workers)