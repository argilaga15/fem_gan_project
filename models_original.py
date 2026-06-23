import torch
import torch.nn as nn

class Generator3D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512*4*4*4),
            nn.ReLU(True)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose3d(512,256,4,2,1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256,128,4,2,1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128,64,4,2,1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64,32,4,2,1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32,1,3,padding=1),
            nn.Tanh()
        )

    def forward(self,z):
        x=self.fc(z)
        x=x.view(z.size(0),512,4,4,4)
        return self.net(x)

class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv3d(1,32,4,2,1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout3d(0.2),
            nn.Conv3d(32,64,4,2,1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2,True),
            nn.Dropout3d(0.2),
            nn.Conv3d(64,128,4,2,1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2,True),
            nn.Dropout3d(0.2),
            nn.Conv3d(128,256,4,2,1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2,True),
            nn.Flatten(),
            nn.Linear(256*4*4*4,1)
        )

    def forward(self,x):
        return self.net(x)
