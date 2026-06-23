import torch
import torch.nn as nn

def _group_norm(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)

def _spectral(layer):
    return nn.utils.spectral_norm(layer)

class Generator3D(nn.Module):
    def __init__(self, latent_dim=64, base_channels=256):
        super().__init__()
        self.base_channels = base_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels*4*4*4),
            nn.ReLU(True)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose3d(base_channels, base_channels//2, 4, 2, 1),
            _group_norm(base_channels//2),
            nn.ReLU(True),
            nn.ConvTranspose3d(base_channels//2, base_channels//4, 4, 2, 1),
            _group_norm(base_channels//4),
            nn.ReLU(True),
            nn.ConvTranspose3d(base_channels//4, base_channels//8, 4, 2, 1),
            _group_norm(base_channels//8),
            nn.ReLU(True),
            nn.ConvTranspose3d(base_channels//8, base_channels//16, 4, 2, 1),
            _group_norm(base_channels//16),
            nn.ReLU(True),
            nn.Conv3d(base_channels//16, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self,z):
        x=self.fc(z)
        x=x.view(z.size(0), self.base_channels, 4, 4, 4)
        return self.net(x)

class Discriminator3D(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            _spectral(nn.Conv3d(1, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2,True),
            nn.Dropout3d(0.1),
            _spectral(nn.Conv3d(base_channels, base_channels*2, 4, 2, 1)),
            _group_norm(base_channels*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout3d(0.1),
            _spectral(nn.Conv3d(base_channels*2, base_channels*4, 4, 2, 1)),
            _group_norm(base_channels*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout3d(0.1),
            _spectral(nn.Conv3d(base_channels*4, base_channels*8, 4, 2, 1)),
            _group_norm(base_channels*8),
            nn.LeakyReLU(0.2,True),
            nn.Flatten(),
            _spectral(nn.Linear(base_channels*8*4*4*4,1))
        )

    def forward(self,x):
        return self.net(x)
