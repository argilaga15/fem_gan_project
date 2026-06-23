import copy
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt

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

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    G=Generator3D(LATENT_DIM, output_size=PX).to(device)
    D=Discriminator3D().to(device)
    ema_G=copy.deepcopy(G).eval()

    optG=torch.optim.Adam(G.parameters(),lr=LR_G,betas=(BETA1,BETA2))
    optD=torch.optim.Adam(D.parameters(),lr=LR_D,betas=(BETA1,BETA2))

    criterion=torch.nn.BCEWithLogitsLoss()
    fem=FEMSupervisor()

    scaler=torch.amp.GradScaler('cuda', enabled=AMP and torch.cuda.is_available())
    fixed_z = torch.randn(1, LATENT_DIM, device=device)
    d_losses = []
    g_losses = []

    def save_loss_plot(epoch_num):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.plot(range(1, len(d_losses) + 1), d_losses, label='Discriminator loss')
        ax.plot(range(1, len(g_losses) + 1), g_losses, label='Generator loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training loss evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(checkpoint_dir / f'loss_evolution_epoch_{epoch_num}.png', bbox_inches='tight')
        plt.close(fig)

    for epoch in range(EPOCHS):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for real in loader:
            real=real.to(device)
            bs=real.size(0)
            num_batches += 1

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
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        epoch_d_loss /= max(num_batches, 1)
        epoch_g_loss /= max(num_batches, 1)
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)

        epoch_num = epoch + 1

        if epoch_num % 100 == 0:
            torch.save({
                'epoch':epoch_num,
                'G':G.state_dict(),
                'D':D.state_dict(),
                'ema_G':ema_G.state_dict()
            }, checkpoint_dir / f'checkpoint_{epoch_num}.pt')

        if epoch_num % 20 == 0:
            with torch.no_grad():
                generated = ema_G(fixed_z).detach().cpu()[0, 0].numpy()
            mid_z = generated.shape[2] // 2
            slice_img = generated[:, :, mid_z]

            fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
            ax.imshow(slice_img, cmap='gray')
            ax.set_title(f'Epoch {epoch_num} - central z slice')
            ax.axis('off')
            fig.tight_layout(pad=0)
            fig.savefig(checkpoint_dir / f'generated_epoch_{epoch_num}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            save_loss_plot(epoch_num)

        #print(epoch_num, epoch_d_loss, epoch_g_loss)

        print(
            f"Epoch {epoch_num:04d}/{EPOCHS} "
            f"D={epoch_d_loss:.4f} "
            f"G={epoch_g_loss:.4f}"
        )

    save_loss_plot(EPOCHS)
    (checkpoint_dir / 'loss_evolution.png').write_bytes(
        (checkpoint_dir / f'loss_evolution_epoch_{EPOCHS}.png').read_bytes()
    )

if __name__ == "__main__":
    main()
