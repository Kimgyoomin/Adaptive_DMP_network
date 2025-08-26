import torch, torch.nn as nn


class WeightMLP(nn.Module):
    def __init__(self, T_in: int, K_fix: int):
        super().__init__()
        D = 2*T_in
        H = 256
        self.K = K_fix
        self.backbone = nn.Sequential(
            nn.Linear(D, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
        )
        self.head_x = nn.Linear(H, K_fix)
        self.head_y = nn.Linear(H, K_fix)

    def forward(self, y_in):            # y_in : (B, T_in, 2)
        z = y_in.reshape(y_in.size(0), -1)
        h = self.backbone(z)
        return self.head_x(h), self.head_y(h)   # (B, K), (B, K)