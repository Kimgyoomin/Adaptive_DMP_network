import os, argparse, numpy as np
from pathlib import Path
# from network_pkg.admp_core.core.gen.bezier      import rand_bezier_single
from network_pkg.admp_core.core.gen.bezier      import rand_bezier_fixed_dist
from network_pkg.admp_core.core.gen.test_curves import circle_arc, s_curve
# from network_pkg.admp_core.core.label.labeler   import pick_K_and_weights
from network_pkg.admp_core.core.label.labeler   import pick_K_continuous

def get_random_fixed(T=400, D=5.0, jitter=0.25, n_ctrl=2):
    return rand_bezier_fixed_dist(T=T, D=D, jitter=jitter, n_ctrl=n_ctrl)

def gen_testbank(T=400):
    return [circle_arc(T=T), s_curve(T=T)]      # only for open curve
    
# def gen_random(T=400, jitter=0.25):
#     return rand_bezier_single(T=T, jitter=jitter)


# def gen_testbank(T=400):
#     return [circle_arc(T=T), lemniscate(T=T), s_curve(T=T)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",     type=str,               default="dataset/train")
    ap.add_argument("--num",        type=int,               default=500)
    ap.add_argument("--T_out",      type=int,               default=600)                # >= 2 * K_max
    ap.add_argument("--D",          type=float,             default=5.0)
    ap.add_argument("--Kmin",       type=int,               default=32)
    ap.add_argument("--Kmax",       type=int,               default=2048)
    ap.add_argument("--grid_pts",   type=int,               default=30)
    # ap.add_argument("--Kset",       type=int,   nargs="+",  default=[10, 100, 300, 600])
    ap.add_argument("--lam",        type=float,             default=0.02)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        # for randomization jitter, n_ctrl random sweep
        jitter = np.clip(0.2 + 0.15*np.random.randn(), 0.05, 0.5)
        n_ctrl = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
        traj = get_random_fixed(T=400, D=args.D, jitter=jitter, n_ctrl=n_ctrl)

        out  = pick_K_continuous(traj, dt=None, T_out=args.T_out,
                                 K_min = args.Kmin, K_max=args.Kmax,
                                 grid="log", num=args.grid_pts, lam=args.lam)
        np.savez_compressed(outdir/f"{i:06d}.npz",
            traj=traj, K=out["K"], wx=out["wx"], wy=out["wy"],
            c=out["c"], h=out["h"], y0=out["y0"], g=out["g"],
            nrmse=out["nrmse"], rmse=out["rmse"])
        
        if i % 50 == 0:
            print(f"[{i} / {args.num}] K*={out['K']}, nRMSE={out['nrmse']:.4f}")


    # Store example of test bench
    tb_dir = outdir.parent/"testbank"; tb_dir.mkdir(parents=True, exist_ok=True)
    for j, t in enumerate(gen_testbank(T=800)):
        out = pick_K_continuous(t, dt=None, T_out=max(args.T_out, 800), 
                                K_min=args.Kmin, K_max=args.Kmax,
                                grid="log", num=args.grid_pts, lam=args.lam)
        np.savez_compressed(tb_dir/f"tb_{j:02d}.npz", traj=t, **out)
        print(f"[test {j}] K*={out['K']}, nRMSE={out['nrmse']:.4f}")

if __name__ == "__main__":
    main()