import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from config import DSZE, EPOCH_SAMPLES, derive_dsze, derive_epoch_samples

def preprocess(x, invert=False):
    x = np.asarray(x)

    # Normalize to [0, 1] using the natural range of the input dtype.
    # Integer arrays are scaled from their full representable range.
    # Floating-point arrays are scaled from their observed range when
    # needed, while already-normalized inputs are left as-is.
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float64)
        x = (x - info.min) / (info.max - info.min)
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        if xmax > 1.0 or xmin < 0.0:
            span = xmax - xmin
            if span > 0:
                x = (x - xmin) / span
            else:
                x = np.zeros_like(x)
        else:
            x = np.clip(x, 0.0, 1.0)
    else:
        x = x.astype(np.float64)
        xmin = np.min(x)
        xmax = np.max(x)
        span = xmax - xmin
        if span > 0:
            x = (x - xmin) / span
        else:
            x = np.zeros_like(x)

    # Optionally keep the original inversion convention.
    if invert:
        x = 1.0 - x
    x = (x - 0.5) / 0.5
    return x.astype(np.float32, copy=False)

class TomographyDataset(Dataset):
    def __init__(self, volume, px, dsze=None, epoch_samples=None, invert=False):
        self.volume=volume
        self.px=px
        self.dsze=derive_dsze(px) if dsze is None else dsze
        self.epoch_samples=derive_epoch_samples(px) if epoch_samples is None else epoch_samples
        self.pxs=int(round(px*self.dsze))
        self.invert=invert
        self.h=self.pxs//2

        # Keep the crop fully inside the volume on every axis.
        self.xc=volume.shape[0]//2
        self.yc=volume.shape[1]//2
        self.z_margin=self.h+80

        # Compute a safe crop margin so every sample stays inside the volume.
        max_h = min(self.xc, self.yc, volume.shape[2] // 2 - 1)
        if self.h > max_h:
            raise ValueError(
                f"Volume {volume.shape} is too small for PX={self.px} with DSZE={self.dsze}."
            )

        self.h = min(self.h, max_h)
        self.pxs = self.h * 2
        self.z_margin = self.h + 1
        self.L = int(volume.shape[2] - 2 * self.z_margin)
        if self.L <= 0:
            raise ValueError(
                f"Volume depth {volume.shape[2]} is too small for PX={self.px} with DSZE={self.dsze}."
            )

        # Preserve the original "ring" sampling idea, but cap it so that
        # x/y crops never run off the edge of the volume.
        safe_center=min(self.xc, self.yc)
        self.R=int(safe_center - self.h - 1)
        if self.R <= 0:
            raise ValueError(
                f"Volume spatial size {volume.shape[:2]} is too small for PX={self.px} with DSZE={self.dsze}."
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
        crop=preprocess(crop, invert=self.invert)
        return torch.from_numpy(crop).unsqueeze(0).float()
