def derive_dsze(px, target_raw_px=80, min_dsze=0.5, max_dsze=2.0):
    if px <= 0:
        raise ValueError("PX must be positive.")
    dsze = target_raw_px / float(px)
    return max(min_dsze, min(max_dsze, dsze))


def derive_epoch_samples(px, base_px=64, base_samples=2000, min_samples=500, max_samples=5000):
    if px <= 0:
        raise ValueError("PX must be positive.")
    scale = base_px / float(px)
    samples = int(round(base_samples * scale))
    return max(min_samples, min(max_samples, samples))


PX = 64
DSZE = derive_dsze(PX)
LATENT_DIM = 64
BATCH_SIZE = 1
LR_G = 1e-4
LR_D = 1e-4
BETA1 = 0.0
BETA2 = 0.99
EPOCHS = 300
EMA_DECAY = 0.999
AMP = True
EPOCH_SAMPLES = derive_epoch_samples(PX)
