import os, glob, numpy as np, torch
from network_pkg.admp_core.model.resolution_net import ResolutionMLP
from network_pkg.admp_core.core.basis.make_rbf  import make_rbf
from network_pkg.admp_core.core.dmp.dmp2d       import fit_weights_2d, rollout_2d
from network_pkg.admp_core.core.label.labeler   import nrmse

def resample_norm(y, T, eps=1e-6):
    idx = np.linspace(0, len(y) - 1, T).astype(int)
    z = y[idx]
    d = np.linalg.norm(z[-1] - z[0])
    if d < 1e-3:
        # if closed/semi-closed : scale with the length of bow
        diffs   = np.diff(z, axis=0)
        L       = np.sum(np.linalg.norm(diffs, axis=1))
        scale   = max(L, eps)
    else:
        scale   = d
    return (z - z[0]) / scale


def main(root="dataset/testbank", T_in=200, T_out=600, K_min=10, K_max=600, model_path="artifacts/resolution_mlp.pth"):
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = ResolutionMLP(T_in); model.load_state_dict(torch.load("artifacts/resolution_mlp.pth")); model.eval()

    model = ResolutionMLP(T_in).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    vals=[]
    for f in files:
        data = np.load(f)
        y = data["traj"]
        y_in = resample_norm(y, T_in)
        with torch.no_grad():
            r_pred = model(torch.tensor(y_in[None], dtype=torch.float32, device=device)).item()
        K_pred = int(np.clip(round(1.0/max(r_pred, 1e-4)), K_min, K_max))

        y_lab = resample_norm(y, T_out)
        dt = 1.0 / (T_out - 1)
        c, h = make_rbf(K_pred)
        wx, wy, y0, g = fit_weights_2d(y_lab, dt, c, h)
        yhat = rollout_2d(T_out, dt, c, h, wx, wy, y0, g)
        nr, r = nrmse(y_lab, yhat)

        K_gt = int(data["K"]) if "K" in data.files else None
        if K_gt is not None:
            print(f"{os.path.basename(f)}  K_pred={K_pred:4d}  K_gt={K_gt:4d}  nRMSE={nr:.4f}")
        else:
            print(f"{os.path.basename(f)}  K_pred={K_pred:4d}  nRMSE={nr:.4f}")

        vals.append(nr)

    if vals:
        print("p50 = ", np.percentile(vals, 50), "p90 = ", np.percentile(vals, 90), "max = ", np.max(vals))
    
    # print("p50=", np.percentile(vals, 50), "p90=", np.percentile(vals, 90), "max=", np.max(vals))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="dataset/testbank")
    ap.add_argument("--T_in", type=int, default=200)
    ap.add_argument("--T_out", type=int, default=600)
    ap.add_argument("--K_min", type=int, default=10)
    ap.add_argument("--K_max", type=int, default=600)
    ap.add_argument("--model_path", type=str, default="artifacts/resolution_mlp.pth")
    args = ap.parse_args()
    main(args.root, args.T_in, args.T_out, args.K_min, args.K_max, args.model_path)
