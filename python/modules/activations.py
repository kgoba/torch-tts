import torch
import torch.nn as nn


def isru_sigmoid(x):
    return (1 + isru(x / 2, 1.0)) / 2


def isru(x, alpha: float = 1.0):
    return x / torch.sqrt(1 + alpha * (x * x))


def isrlu(x, alpha: float = 1.0):
    return torch.where(x >= 0, x, x / torch.sqrt(1 + alpha * (x * x)))


class ISRU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x / torch.sqrt(1 + self.alpha * (x * x))


class ISRLU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, x / torch.sqrt(1 + self.alpha * (x * x)))
