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

For real data, to do two domain-specific upgrades for materials generation:

Replace BatchNorm in the discriminator with Spectral Normalization.
Replace BCE loss with Hinge GAN loss (usually more stable for 3D microstructure synthesis).

Commit 23/6/2026

Updated train.py so the loss plot is also saved every 20 epochs.
What changed:
Every 20 epochs, it now saves:
checkpoints/generated_epoch_<N>.png
checkpoints/loss_evolution_epoch_<N>.png
final loss_evolution.png is a copy of the last periodic plot, so the end-of-training file always matches the latest plotted state.

Data type support:
What changed:
preprocess() now handles:
unsigned integers like uint8, uint16, uint32
signed integers like int16, int32
floating-point arrays like float32 and float64
It normalizes to [0, 1] using the dtype’s natural range for integers.
For floats, it uses the observed range when values are outside [0, 1].
It still keeps  original inversion convention and final output range of [-1, 1].
preprocess(x, invert=True) now takes an optional invert flag.
TomographyDataset(..., invert=True) now stores that setting and passes it through.
The default behavior is unchanged, so your current pipeline still inverts by default.
