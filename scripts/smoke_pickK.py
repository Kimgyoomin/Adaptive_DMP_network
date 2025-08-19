import numpy as np
from network_pkg.admp_core.core.gen.bezier      import rand_bezier_single
from network_pkg.admp_core.core.label.labeler   import pick_K_and_weights

def main():
    np.random.seed(0)
    traj = rand_bezier_single(T=400, jitter=0.25)
    out = pick_K_and_weights(traj, dt=None)
    print(f"K* = {out['K']}, nRMSE = {out['nrmse']:.4f}, RMSE={out['rmse']:.4f}")

if __name__ == "__main__":
    main()