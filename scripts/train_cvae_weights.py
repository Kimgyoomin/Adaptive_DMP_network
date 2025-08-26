# network_pkg/scripts/train_cvae_weights.py
import os, glob, argparse, numpy        as np
from pathlib                            import Path
import torch, torch.nn                  as nn
from torch.utils.data                   import Dataset, DataLoader

from network_pkg.admp_core.model.cvae_weights       import CVAEWeights
from network_pkg.admp_core.core.basis.make_rbf      import make_rbf
from network_pkg.admp_core.core.dmp.dmp2d           import fit_weights_2d


def resample_norm_safe(y, T, eps=1e-6):
    idx     = np.linspace(0, len(y) - 1, T).astype(int)
    z       = y[idx].astype(np.float64)
    d       = np.linalg.norm(z[-1] - z[0])
    if d < eps:
        diffs       = np.diff(z, axis=0)
        L           = np.sum(np.linalg.norm(diffs, axis=1))
        d           = max(L, eps)
    return ((z - z[0]) / d).astype(np.float32)


def make_context(y_world):
    s, g    = y_world[0], y_world[-1]
    diff    = g - s
    D       = float(np.linalg.norm(diff))
    tau     = 1.0
    return np.array([diff[0], diff[1], D, tau], dtype=np.float32)


class TrajCVaeDataset(Dataset):
    def __init__(self, root="dataset/train", T_in=200, T_out=600, K_fix=128):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files: raise FileNotFoundError(f"No npz in {root}")
        self.T_in, self.T_out, self.K = T_in, T_out, K_fix
        self.c_rbf, self.h_rbf = make_rbf(K_fix)
        self.dt = 1.0 / (T_out - 1)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        data        = np.load(self.files[i])
        y_world     = data["traj"].astype(np.float64)

        y_in        = resample_norm_safe(y_world, self.T_in)        # (T_in, 2) float 32
        c_vec       = make_context(y_world)                         # (4, )
        if ("wx" in data) and ("wy" in data) and ("K" in data) and int(data["K"])==self.K:
            wx = data["wx"].astype(np.float32)
            wy = data["wy"].astype(np.float32)
        else:
            y_fit = resample_norm_safe(y_world, self.T_out).astype(np.float64)
            wx, wy, _, _ = fit_weights_2d(y_fit, self.dt, self.c_rbf, self.h_rbf)
            wx = wx.astype(np.float32)
            wy = wy.astype(np.float32)



        return torch.from_numpy(y_in), torch.from_numpy(c_vec), \
               torch.from_numpy(wx), torch.from_numpy(wy)
    
def kld_gauss(mu, logvar):
    # mean over batch : -0.5 * sum(1 + log(sigma)^2 - mu^2 - sigma^2)
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def train(root="dataset/train", epochs=60, bs=128, lr=5e-4,
          T_in=200, T_out=600, K_fix=128, z_dim=32, beta=1e-3,
          num_workers=0, wd=0.0, clip=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrajCVaeDataset(root, T_in, T_out, K_fix)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=num_workers, pin_memory=True)
    
    C_dim   = 4
    model   = CVAEWeights(T_in=T_in, K_fix=K_fix, C_dim=C_dim, z_dim=z_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mse     = nn.MSELoss()

    model.train()
    for ep in range(1, epochs+1):
        tot = 0.0
        for y_in, c_vec, wx_gt, wy_gt, in dl:
            y_in    = y_in.to(device)               # (B, T_in, 2)
            c_vec   = c_vec.to(device)              # (B, 4)
            wx_gt   = wx_gt.to(device)              # (B, K)
            wy_gt   = wy_gt.to(device)              # (B, K)

            wx_pred, wy_pred, mu, logvar = model(y_in, c_vec, deterministic=False)
            loss_rec = mse(wx_pred, wx_gt) + mse(wy_pred, wy_gt)
            loss_kld = kld_gauss(mu, logvar)
            loss = loss_rec + beta * loss_kld

            opt.zero_grad()
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            tot += loss.item() * y_in.size(0)
        print(f"[ep {ep}] loss={tot/len(ds):.6f} (beta={beta})")

    Path("artifacts").mkdir(exist_ok=True)
    path = f"artifacts/cvae_weights_K{K_fix}_z{z_dim}.pth"
    torch.save(model.state_dict(), path)
    print(f"saved {path}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="dataset/train")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--T_in", type=int, default=200)
    ap.add_argument("--T_out", type=int, default=600)
    ap.add_argument("--Kfix", type=int, default=128)
    ap.add_argument("--z", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()

    train(root=args.root, epochs=args.epochs, bs=args.bs, lr=args.lr,
          T_in=args.T_in, T_out=args.T_out, K_fix=args.Kfix, z_dim=args.z,
          beta=args.beta, num_workers=args.workers, wd=args.wd, clip=args.clip)