import torch
import torch.nn as nn
from src.transforms.dwt1d import DWT1D


class MSEWithHaarLoss(nn.Module):
    def __init__(self, haar_lambda=0.1, wavelet='haar'):
        super().__init__()
        self.haar_lambda = haar_lambda
        self.mse = nn.MSELoss()
        self.dwt = DWT1D(in_channels=1, wavelet=wavelet)

    def forward(self, pred, target):
        loss_t = self.mse(pred, target)
        loss_h = self.mse(self.dwt(pred), self.dwt(target))
        return loss_t + self.haar_lambda * loss_h
