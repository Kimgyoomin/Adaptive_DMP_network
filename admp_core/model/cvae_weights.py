import torch, torch.nn as nn

class Encoder1D(nn.Module):
    def __init__(self, T_in: int, C_dim: int, z_dim: int):
        super().__init__()
        # Input (B, T, 2) -> (B, 2, T) 1D-CNN Encoder
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)     # (B, 64, 1) -> summarize globally
        )
        self.fc = nn.Sequential(
            nn.Linear(64 + C_dim, 128), nn.ReLU()
        )
        self.mu         = nn.Linear(128, z_dim)
        self.logvar     = nn.Linear(128, z_dim)

    def forward(self, y_in, c):
        # y_in : (B, T, 2), c : (B, C_dim)
        h           = self.conv(y_in.transpose(1, 2)).squeeze(-1)     # (B, 64)
        hc          = torch.cat([h, c], dim=-1)
        h2          = self.fc(hc)
        mu, logvar  = self.mu(h2), self.logvar(h2)
        return mu, logvar
    



class DecoderMLP(nn.Module):
    def __init__(self, K_fix: int, C_dim: int, z_dim: int):
        super().__init__()
        H = 256
        self.net = nn.Sequential(
            nn.Linear(z_dim + C_dim, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
        )
        self.head_x = nn.Linear(H, K_fix)
        self.head_y = nn.Linear(H, K_fix)

    def forward(self, z, c):
        h = self.net(torch.cat([z, c], dim=-1))
        return self.head_x(h), self.head_y(h)
    


class CVAEWeights(nn.Module):
    def __init__(self, T_in: int, K_fix: int, C_dim: int=4, z_dim: int=32):
        super().__init__()
        self.enc    = Encoder1D(T_in, C_dim, z_dim)
        self.dec    = DecoderMLP(K_fix, C_dim, z_dim)


    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self, y_in, c, deterministic: bool = False):
        mu, logvar = self.enc(y_in, c)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        wx, wy = self.dec(z, c)
        return wx, wy, mu, logvar