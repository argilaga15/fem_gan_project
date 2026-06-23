import copy
import torch

def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.mul_(decay).add_(p.data, alpha=1-decay)

def make_ema(model):
    return copy.deepcopy(model).eval()
