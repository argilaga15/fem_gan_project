train.py assumes your tomography volume is stored as:

tomography.npy
tomography.mat
or
tomography.tif

with shape:

(X, Y, Z)
The FEM supervision is intentionally left as a placeholder in fem.py.
The project includes:
	Streaming cylindrical sampling dataset
	BCEWithLogits GAN loss
	AMP mixed precision
	EMA generator
	Checkpointing
	3D Generator / Discriminator
	MATLAB-equivalent preprocessing

Before running on real data, I would still make two domain-specific upgrades for materials generation:

Replace BatchNorm in the discriminator with Spectral Normalization.
Replace BCE loss with Hinge GAN loss (usually more stable for 3D microstructure synthesis).

Those changes are straightforward and generally improve training stability for volumetric materials datasets. 

Commit 23/6/2026

Updated train.py so the loss plot is also saved every 20 epochs.
What changed:
Every 20 epochs, it now saves:
checkpoints/generated_epoch_<N>.png
checkpoints/loss_evolution_epoch_<N>.png
At the end, it still saves the final plot as checkpoints/loss_evolution.png
I also kept the 100-epoch .pt checkpoint cadence unchanged.
One small note: I made the final loss_evolution.png a copy of the last periodic plot, so the end-of-training file always matches the latest plotted state.
If you want, I can also make the checkpoint interval configurable from config.py so you can change 20 and 100 without touching the training script.