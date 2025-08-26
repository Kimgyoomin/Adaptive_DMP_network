# network_pkg/scripts/make_textbook.py
import os, argparse, numpy as np
from pathlib import Path

from network_pkg.admp_core.core.gen.bezier      import rand_bezier_fixed_dist
from network_pkg.admp_core.core.gen.test_curves import circle_arc, s_curve
from network_pkg.admp_core.core.basis.make_rbf  import make_rbf
# from network_pkg.admp_core.core.dmp.dmp2d       import fit_weights_2d
from network_pkg.admp_core.core.dmp.dmp2d       import inverse_dmp_2d


# -- util : Normalization resample ( trainig / label correction )
def resample_norm_safe(y, T, eps=1e-6):
    idx = np.linspace(0, len(y) - 1, T).astype(int)
    z   = y[idx].astype(np.float64)
    d   = np.linalg.norm(z[1] - z[0])
    if d < 1e-3:
        diffs   = np.diff(z, axis=0)
        L       = np.sum(np.linalg.norm(diffs, axis=1))
        d       = max(L, eps)
    return (z - z[0]) / d


def get_random_fixed(T=400, D=5.0, jitter=0.25, n_ctrl=2):
    return rand_bezier_fixed_dist(T=T, D=D, jitter=jitter, n_ctrl=n_ctrl)


def gen_testbank(T=400):
    # Use open-curve (without closed-curve)
    # 25.08.25
    # I think it's reasonable to use only open-curve 
    # When navigating through terrain, closed-curve should be devided into many open-curves
    return [circle_arc(T=T), s_curve(T=T)]      # only for open curve
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",     type=str,               default="dataset/train_force")
    ap.add_argument("--num",        type=int,               default=500)
    ap.add_argument("--T_out",      type=int,               default=600)                # >= 2 * K_max
    ap.add_argument("--D",          type=float,             default=5.0)
    # ap.add_argument("--Kmin",       type=int,               default=32)
    # ap.add_argument("--Kmax",       type=int,               default=2048)
    ap.add_argument("--Kfix",       type=int,               default=128)
    # ap.add_argument("--grid_pts",   type=int,               default=30)
    # ap.add_argument("--Kset",       type=int,   nargs="+",  default=[10, 100, 300, 600])
    # ap.add_argument("--lam",        type=float,             default=0.02)
    ap.add_argument("--store_w",    action="store_true",    help="label(w, c, h) store either")
    args = ap.parse_args()

    outdir  = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    c, h    = make_rbf(args.Kfix)
    dt      = 1.0 / (args.T_out - 1)

    for i in range(args.num):
        # for randomization jitter, n_ctrl random sweep
        jitter = np.clip(0.2 + 0.15*np.random.randn(), 0.05, 0.5)
        n_ctrl = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])

        # raw world-curve creation
        traj_world  = get_random_fixed(T=400, D=args.D, jitter=jitter, n_ctrl=n_ctrl)
        # at normalization coordinate , create label
        y_fit           = resample_norm_safe(traj_world, args.T_out)
        wx, wy, y0, g   = fit_weights_2d(y_fit, dt, c, h)

        # Simple resample error (At Normalization coordinate)
        # call $ rollout_2d and put nRMSE after import and Calculation
        # for here, storage only
        if args.store_w:
            np.savez_compressed(outdir/f"{i:06d}.npz",
            traj = traj_world, K = args.Kfix, wx=wx, wy=wy,
            c=c, h=h, y0=y0, g=g)

        else:
            np.savez_compressed(outdir/f"{i:06d}.npz", traj=traj_world)


        if i % 50 == 0:
            print(f"[{i} / {args.num}] done (Kfix={args.Kfix})")



    # --- test bank ---
    tb_dir = outdir.parent/"testbank"
    tb_dir.mkdir(parents=True, exist_ok=True)
    for j, t in enumerate(gen_testbank(T=800)):
        y_fit = resample_norm_safe(t, max(args.T_out, 800))
        wx, wy, y0, g = fit_weights_2d(y_fit, dt, c, h)
        np.savez_compressed(tb_dir/f"tb_{j:02d}.npz",
            traj=t, K=args.Kfix, wx=wx, wy=wy, c=c, h=h, y0=y0, g=g)
        print(f"[test {j}] prepared with Kfix={args.Kfix}")

if __name__ == "__main__":
    main()