import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

def preprocess(x):
    x=x.astype(np.float32)
    x=65535.0-x
    x=x/65535.0
    x=(x-0.5)/0.5
    return x

class TomographyDataset(Dataset):
    def __init__(self, volume, px=128, dsze=1.25, epoch_samples=10000):
        self.volume=volume
        self.px=px
        self.pxs=int(round(px*dsze))
        self.epoch_samples=epoch_samples
        self.h=self.pxs//2

        # Keep the crop fully inside the volume on every axis.
        self.xc=volume.shape[0]//2
        self.yc=volume.shape[1]//2
        self.z_margin=self.h+80

        self.L=int(volume.shape[2]-2*self.z_margin)

        # Preserve the original "ring" sampling idea, but cap it so that
        # x/y crops never run off the edge of the volume.
        safe_center=min(self.xc, self.yc)
        self.R=int(safe_center-100-round(self.pxs*0.7071))

        if self.L <= 0:
            raise ValueError(
                f"Volume depth {volume.shape[2]} is too small for a {self.px}px crop."
            )
        if self.R <= 0:
            raise ValueError(
                f"Volume spatial size {volume.shape[:2]} is too small for a {self.px}px crop."
            )

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, idx):
        ranh=self.z_margin+np.random.randint(self.L)
        ranr=np.random.randint(self.R)
        rant=np.random.rand()*2*np.pi
        cx=self.xc+round(ranr*np.cos(rant))
        cy=self.yc+round(ranr*np.sin(rant))
        h=self.h
        crop=self.volume[cx-h:cx+h, cy-h:cy+h, ranh-h:ranh+h]
        crop=zoom(crop, self.px/self.pxs, order=1)
        crop=preprocess(crop)
        return torch.from_numpy(crop).unsqueeze(0).float()
