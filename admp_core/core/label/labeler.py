import numpy as np
from network_pkg.admp_core.core.basis.make_rbf  import make_rbf
from network_pkg.admp_core.core.dmp.dmp2d       import fit_weights_2d, rollout_2d


def pick_K_continuous(traj_xy, dt=None, T_out = 600,
                      K_min = 32, K_max = 2048, grid="log", num=30,
                      lam=0.02, ay=25.0, by=6.25, ax=4.6052, ridge_lambda=0.0):
    """
    for dense K grid // penalized score = nRMSE + lam*sqrt(K/T_out) minimum K* choose
    - grid = "log" : log space (for complexity spectrum)
    - ridge_lambda >0 : Ridge for OLS (Numerical stability)
     """
    y = traj_xy
    if dt is None:
        dt = 1.0 / (T_out - 1)
    # Resample + scale regularization (closed safety)
    idx = np.linspace(0, len(y) - 1, T_out).astype(int)
    y = y[idx]
    d = np.linalg.norm(y[-1] - y[0])
    if d < 1e-3:
        diffs = np.diff(y, axis=0); L = np.sum(np.linalg.norm(diffs, axis=1))
        scale = max(L, 1e-6)
    else:
        scale = d
    y = (y - y[0]) / scale

    if grid == "log":
        Ks = np.unique(np.round(np.geomspace(K_min, K_max, num)).astype(int))
    else:
        Ks = np.arange(K_min, K_max+1)

    best = dict(score=np.inf)
    for K in Ks:
        c, h = make_rbf(K)
        wx, wy, y0, g = fit_weights_2d(y, dt, c, h, ay, by, ax)     # If needed, add ridge
        yhat = rollout_2d(T_out, dt, c, h, wx, wy, y0, g, ay, by, ax)
        err = np.sqrt(((y - yhat)**2).sum(axis=1)).mean() / (np.linalg.norm(y[-1]-y[0]) + 1e-12)  # nRMSE
        score = err + lam * np.sqrt(K / T_out)
        if score < best["score"]:
            best = dict(K=int(K), c=c, h=h, wx=wx, wy=wy, y0=y0, g=g,
                        nrmse=float(err), rmse=float(err), score=float(score))
    return best


def nrmse(y, yhat, eps=1e-9):
    rmse = np.sqrt(np.mean((yhat - y) ** 2))
    diag = np.linalg.norm(y[-1] - y[0]) + eps
    return rmse / (diag + eps), rmse

def resample_uniform(y, T):
    idx = np.linspace(0, len(y)-1, T).astype(int)
    return y[idx]

def normalize_by_start_goal(y, eps=1e-9):
    return (y - y[0]) / (np.linalg.norm(y[-1] - y[0]) + eps)

def pick_K_and_weights(traj_xy: np.ndarray, dt: float,
                       K_set=(10, 100, 300, 600, 1000, 2500, 3000, 10000),
                       ay = 25., by=6.25, ax=4.6052, T_out=200):
    """
    For each K, OLS fitting -> Rollout -> choose K* of minimum nRMSE
    you can get back dict: {K, wx, wy, c, h, y0, g, nrmse, rmse}
    """

    # 1) Resample & Regularization
    y = resample_uniform(traj_xy, T_out).astype(np.float64)
    y = normalize_by_start_goal(y)

    # 2) set dt automatically : Set runtime as 1.0 (align as standard DMP's time azis)
    if dt is None:
        dt = 1.0 / (T_out - 1)

    best = None

    for K in K_set:
        c, h = make_rbf(K, ax=ax)
        wx, wy, y0, g = fit_weights_2d(y, dt, c, h, ay, by, ax)
        yhat = rollout_2d(T_out, dt, c, h, wx, wy, y0, g, ay, by, ax)
        nr, r = nrmse(y, yhat)
        rec = (nr, r, K, (wx, wy, c, h, y0, g))
        if (best is None) or (nr < best[0]):
            best = rec

    nr, r, K, pack = best
    wx, wy, c, h, y0, g = pack
    return dict(K=K, wx=wx, wy=wy, c=c, h=h, y0=y0, g=g, nrmse=nr, rmse=r)