import torch, torch.nn as nn

class ResolutionMLP(nn.Module):
    def __init__(self, T_in=200):
        super().__init__()
        D = 2 * T_in            # (x, y)
        self.net = nn.Sequential(
            nn.Linear(D, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.softplus = nn.Softplus()       # set >= 0

    def forward(self, traj_norm):       # traj_norm: (B, T_in, 2)
        x = traj_norm.reshape(traj_norm.size(0), -1)
        r_raw = self.net(x)
        r = self.softplus(r_raw) + 1e-4
        return r    # almost 1/K