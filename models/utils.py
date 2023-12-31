"""
Model utils

"""
import torch.nn as nn
from copy import deepcopy as c


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    r"""clones model with N copy.
    Args:
        module (nn.Module): Torch model to copy.
        N (int): Number of copy time.
    """
    return nn.ModuleList(c(module) for _ in range(N))
