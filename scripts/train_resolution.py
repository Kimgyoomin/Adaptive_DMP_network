import os, glob, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from network_pkg.admp_core.model.resolution_net import ResolutionMLP

def resample_norm(traj, T):
    idx = np.linspace(0, len(traj) - 1, T).astype(int)
    y = traj[idx]
    y = (y - y[0]) / (np.linalg.norm(y[-1] - y[0]) + 1e-9)
    return y.astype(np.float32)


class TrajDataset(Dataset):
    def __init__(self, root="dataset/train", T_in=200):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.T_in = T_in

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        y = resample_norm(data["traj"], self.T_in)          # (T_in, 2)
        K = int(data["K"]); r = 1.0 / max(K, 1)             # \delta is almost 1/K
        # traj = data["traj"]
        # resample & regularization
        # idxs = np.linspace(0, len(traj)-1, self.T_in).astype(int)
        # y = traj[idxs]
        # y = (y - y[0]) / (np.linalg.norm(y[-1] - y[0]) + 1e-9)
        # K = int(data["K"]); r = 1.0 / max(K, 1)
        # return torch.tensor(y, dtype=torch.float32), torch.tensor([r], dtype=torch.float32)
        return torch.from_numpy(y), torch.tensor([r], dtype=torch.float32)
        

def train(root="dataset/train", epochs=20, bs=64, lr=1e-3, T_in=200, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrajDataset(root, T_in) 
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    model = ResolutionMLP(T_in).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # model.train()
    for ep in range(1, epochs+1):
        model.train()
        tot=0.0
        for y, r in dl:
            y, r = y.to(device), r.to(device)
            opt.zero_grad()
            r_pred = model(y)
            loss = loss_fn(r_pred, r)
            loss.backward() 
            opt.step()
            tot += loss.item()*y.size(0)
        print(f"[ep {ep}] loss = {tot / len(ds):.6f}")

    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/resolution_mlp.pth")
    print("saved artifacts/resolution_mlp.pth")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",       type=str,       default="dataset/train")
    ap.add_argument("--epochs",     type=int,       default=20)
    ap.add_argument("--bs",         type=int,       default=64)
    ap.add_argument("--lr",         type=float,     default=1e-3)
    ap.add_argument("--T_in",       type=int,       default=200)
    args = ap.parse_args()

    train(args.root, args.epochs, args.bs, args.lr, args.T_in)