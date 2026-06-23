import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import loadmat

from config import *
from dataset import TomographyDataset
from models import Generator3D, Discriminator3D
from fem import FEMSupervisor
from utils import update_ema
from loadervol import load_volume

#print(torch.version.cuda) # Should show 12.4
#print(torch.cuda.is_available()) # Should be True if GPU is detected

#print("Torch version:", torch.__version__)
#print("CUDA compiled with Torch:", torch.version.cuda)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("device =", device)

def main():

    #volume=np.load('tomography.npy')   # replace with your volume
    volume = load_volume("./scans/A4-1_raw_sub_1.tif")

    ds=TomographyDataset(volume, px=PX, epoch_samples=EPOCH_SAMPLES)
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)

    G=Generator3D(LATENT_DIM).to(device)
    D=Discriminator3D().to(device)
    ema_G=copy.deepcopy(G).eval()

    optG=torch.optim.Adam(G.parameters(),lr=LR_G,betas=(BETA1,BETA2))
    optD=torch.optim.Adam(D.parameters(),lr=LR_D,betas=(BETA1,BETA2))

    criterion=torch.nn.BCEWithLogitsLoss()
    fem=FEMSupervisor()

    scaler=torch.amp.GradScaler('cuda', enabled=AMP and torch.cuda.is_available())

    for epoch in range(EPOCHS):
        for real in loader:
            real=real.to(device)
            bs=real.size(0)

            z=torch.randn(bs,LATENT_DIM,device=device)

            with torch.amp.autocast('cuda', enabled=AMP and torch.cuda.is_available()):
                fake=G(z)

                real_logits=D(real)
                fake_logits=D(fake.detach())

                d_loss=(
                    criterion(real_logits, torch.full_like(real_logits,0.9))
                    + criterion(fake_logits, torch.zeros_like(fake_logits))
                )

            optD.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(optD)

            with torch.amp.autocast('cuda', enabled=AMP and torch.cuda.is_available()):
                fake=G(z)
                adv_loss=criterion(D(fake), torch.ones(bs,1,device=device))
                fem_loss=fem.compute_loss(fake)
                g_loss=adv_loss+fem_loss

            optG.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()

            update_ema(ema_G,G,EMA_DECAY)

        torch.save({
            'epoch':epoch,
            'G':G.state_dict(),
            'D':D.state_dict(),
            'ema_G':ema_G.state_dict()
        }, f'checkpoint_{epoch}.pt')

        print(epoch, d_loss.item(), g_loss.item())

if __name__ == "__main__":
    main()
