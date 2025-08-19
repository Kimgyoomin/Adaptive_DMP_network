import numpy as np

def make_rbf(K: int, ax: float = 4.6052):
    """
    Canonical x(t) = exp(-ax*t). t \in [0, 1] devide equally -> x-space center , width creation
    Width : h_i = 1 / (2 * (\delta c_i)^2 ) (Last -> reuse \delta c_{K-1} )
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    t   = np.linspace(0.0, 1.0, K)
    c   = np.exp(-ax * t)                         # x-space equally center
    dc  = np.diff(c)
    if K == 1:
        dc = np.array([1.0])
    else:
        dc = np.append(dc, dc[-1])                  # Reuse final interval
    h = 1.0 / (2.0 * (dc ** 2) + 1e-12)
    return c.astype(np.float64), h.astype(np.float64)