import numpy as np
from math import comb

def rand_goal_on_circle(D=5.0):
    """
    For fixed distance from start goal(0,0) / for convenient
    """
    th = np.random.uniform(-np.pi, np.pi)
    return np.array([D*np.cos(th), D*np.sin(th)], dtype=np.float64)

def rand_bezier_fixed_dist(T=400, D=5.0, jitter=0.25, n_ctrl=2):
    """
    s = (0,0) -> fixed,  ||g|| = D -> fixed
    Bezier curve between s ~ g
    n_ctrl = 2 for now, -> 4th Bezier curve (middle control point 2)
    """
    s = np.zeros(2, dtype=np.float64)
    g = rand_goal_on_circle(D)

    # basic / orthogonal direction
    u = (g - s); L = np.linalg.norm(u) + 1e-12
    t_hat = u / L
    n_hat = np.array([-t_hat[1], t_hat[0]])

    # Set control Point : devide as direction of t_hat + random for orthogonal
    ctrl = [s]
    for k in range(1, n_ctrl + 1):
        alpha = k/(n_ctrl + 1)      # Devide equally between 0, 1
        along = s + alpha * (g - s)
        offset = (np.random.randn()*jitter) * L * n_hat * 0.3
        ctrl.append(along + offset)
    ctrl.append(g)
    ctrl = np.stack(ctrl, axis=0)               # (n_ctrl + 2, 2)

    # Sample as De Casteljau or Bernstein
    t = np.linspace(0, 1, T)
    # Bernstein polynomial
    n = len(ctrl) - 1

    B = np.stack([comb(n, i) * (t ** i) * ((1 - t) ** (n - i)) for i in range(n + 1)], axis=1)    # (T,n+1)
    y = B @ ctrl
    return y


def bezier(ctrl: np.ndarray, T: int):
    n = ctrl.shape[0] -1
    t = np.linspace(0.0, 1.0, T)
    B = np.stack([comb(n, i)*(t**i)*((1-t)**(n-i)) for i in range(n+1)], axis=1)
    traj = B @ ctrl # [T, 2]
    return traj

def rand_bezier_single(T=400, jitter=0.25):
    """
    Random Start/Goal + Middle Control Point 2 (3 Bezier)
    """
    s = np.random.uniform(0, 1, size=2)
    g = np.random.uniform(0, 1, size=2)
    m1 = s + (g-s) * np.random.rand() + jitter*np.random.randn(2)
    m2 = s + (g-s) * np.random.rand() + jitter*np.random.randn(2)
    ctrl = np.stack([s, m1, m2, g], axis=0)     # cubic (stable), [4,2]
    traj = bezier(ctrl, T)
    return traj