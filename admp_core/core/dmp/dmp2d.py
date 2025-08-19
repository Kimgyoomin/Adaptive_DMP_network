import numpy as np

def canonical_roll(T: int, dt: float, ax: float = 4.6052, tau: float = 1.0):
    x = np.ones(T, dtype=np.float64)
    for t in range(1, T):
        x[t] = x[t-1] + (-ax * x[t-1]) * (dt / tau)         # x_dot = -ax * x
    return np.clip(x, 0.0, 1.0)


def basis_activations(x, c, h):
    Psi     = np.exp(-(h[None, :] * (x[:, None] - c[None, :]) ** 2))        # [T, K]
    Psi_n   = Psi / (Psi.sum(axis = 1, keepdims=True) + 1e-12)
    return Psi_n

def diff1(y, dt):
    v       = np.zeros_like(y)            # Returns shape of zero mat y
    v[1:-1] = (y[2:] - y[:-2]) / (2*dt)
    v[0]    = (y[1] - y[0]) / dt
    v[-1]   = (y[-1] - y[-2]) / dt
    return v

def fit_weights_2d(traj_xy: np.ndarray, dt: float, c: np.ndarray, h: np.ndarray,
                   ay: float = 25.0, by: float = 6.25, ax: float = 4.6052):
    """
    Using OLS -> find 2D weights (wx, wy)
    - Input : traj_xyL [T, 2], (Regularization at caller)
    - f_target = y_ddot - ay*(by*(g - y) - y_dot)
    - \phi = \psi_n * x * (g - y0); w = argmin||\phi - f||_2^2 (for each axis)
    """
    T       = traj_xy.shape[0]
    x       = canonical_roll(T, dt, ax)
    Psi_n   = basis_activations(x, c, h)      # [T, K]
    y0      = traj_xy[0, :].copy()
    g       = traj_xy[-1, :].copy()

    vx      = diff1(traj_xy[:, 0], dt); axx = diff1(vx, dt)
    vy      = diff1(traj_xy[:, 1], dt); ayy = diff1(vy, dt)

    fx      = axx - ay*(by*(g[0] - traj_xy[:, 0]) - vx)
    fy      = ayy - ay*(by*(g[1] - traj_xy[:, 1]) - vy)

    front_x = x * (g[0] - y0[0])
    front_y = x * (g[1] - y0[1])

    Phi_x   = Psi_n * front_x[:, None]
    wx, *_  = np.linalg.lstsq(Phi_x, fx, rcond=None)

    Phi_y   = Psi_n * front_y[:, None]
    wy, *_  = np.linalg.lstsq(Phi_y, fy, rcond=None)

    return wx.astype(np.float64), wy.astype(np.float64), y0, g

def rollout_2d(T: int, dt: float, c: np.ndarray, h: np.ndarray,
               wx: np.ndarray, wy: np.ndarray, y0: np.ndarray, g: np.ndarray,
               ay: float = 25.0, by: float = 6.25, ax: float = 4.6052, tau: float = 1.0):
    x       = canonical_roll(T, dt, ax, tau)
    Psi_n   = basis_activations(x, c, h)
    f_x     = (Psi_n @ wx) * (x * (g[0] - y0[0]))
    f_y     = (Psi_n @ wy) * (x * (g[1] - y0[1]))

    y       = np.zeros((T, 2), dtype=np.float64)
    v       = np.zeros((T, 2), dtype=np.float64)

    y[0,:]  = y0
    for t in range(T-1):
        a_x = (ay*(by*(g[0] - y[t, 0]) - v[t,0]) + f_x[t]) / tau
        a_y = (ay*(by*(g[1] - y[t, 1]) - v[t,1]) + f_y[t]) / tau

        v[t+1, 0] = v[t, 0] + dt * a_x
        v[t+1, 1] = v[t, 1] + dt * a_y

        y[t+1, 0] = y[t, 0] + dt * (v[t+1, 0] / tau)
        y[t+1, 1] = y[t, 1] + dt * (v[t+1, 1] / tau)
    return y