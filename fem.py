import torch

class FEMSupervisor:
    def __init__(self):
        self.enabled=False

    def compute_loss(self, fake_batch):
        if not self.enabled:
            return torch.tensor(0.0, device=fake_batch.device)
        raise NotImplementedError("Insert FEM implementation here.")
