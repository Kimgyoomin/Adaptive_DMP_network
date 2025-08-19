import numpy as np

def circle_arc(T = 400, r = 0.5, center=(0.5, 0.5), theta0=0.0, dtheta = np.pi):
    t = np.linspace(0.0, 1.0, T)
    th = theta0 + dtheta * t
    cx, cy = center
    x = cx + r * np.cos(th); y = cy + r * np.sin(th)
    return np.stack([x, y], axis=1)


def lemniscate(T=400, a=0.45, center=(0.5, 0.5)):
    # Bernoulli lemniscate usage -> (x^2 + y^2)^2 = a^2 * (x^2 - y^2)
    t = np.linspace(-np.pi/2, np.pi/2, T)
    x = (a * np.sqrt(2) * np.cos(t)) / (np.sin(t) ** 2 + 1)
    y = (a * np.sqrt(2) * np.cos(t) * np.sin(t)) / (np.sin(t) ** 2 + 1)
    x = (x - x.min()) / (x.max() - x.min()); y = (y - y.min()) / (y.max() - y.min())
    cx, cy = center
    return np.stack([x*0.8 + cx - 0.4, y * 0.8 + cy - 0.4], axis=1)

def s_curve(T=400):
    t = np.linspace(0, 1, T)
    x = t; y = 0.5 + 0.35*np.tanh(8 *(t - 0.5))
    return np.stack([x, y], axis=1)

