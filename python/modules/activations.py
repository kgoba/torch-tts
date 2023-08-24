import torch
import torch.nn as nn


def isru(x, alpha=1):
    return x / torch.sqrt(1 + alpha * (x * x))


def isrlu(x, alpha=1):
    return torch.where(x >= 0, x, x / torch.sqrt(1 + alpha * (x * x)))


class ISRU(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x / torch.sqrt(1 + self.alpha * (x * x))


class ISRLU(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, x / torch.sqrt(1 + self.alpha * (x * x)))
