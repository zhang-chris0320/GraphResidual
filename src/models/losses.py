import torch
from torch.nn import functional as F
def preserve_loss(z,e): return 1-F.cosine_similarity(z,e).mean()
def residual_loss(x): return x.square().mean()
