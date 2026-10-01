import torch.nn as nn
from src.transforms.dwt1d import DWT1D, IDWT1D


class DWConvBlock(nn.Module):
    def __init__(self, in_channels, filters, num_layers=5, final_filters=None):
        super().__init__()
        layers = [nn.Conv1d(in_channels, filters, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Conv1d(filters, filters, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        if final_filters is not None:
            layers += [nn.Conv1d(filters, final_filters, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DW_CNN(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.conv1 = DWConvBlock(1, 64, num_layers=5)
        self.pool1 = DWT1D(in_channels=64, wavelet=wavelet)
        self.conv2 = DWConvBlock(128, 64, num_layers=5)
        self.pool2 = DWT1D(in_channels=64, wavelet=wavelet)
        self.latent = DWConvBlock(128, 64, num_layers=5, final_filters=128)
        self.up1 = IDWT1D(in_channels=128, wavelet=wavelet)
        self.conv4 = DWConvBlock(64, 64, num_layers=5, final_filters=128)
        self.up2 = IDWT1D(in_channels=128, wavelet=wavelet)
        self.out_block = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        latent = self.latent(p2)
        u1 = self.up1(latent)
        a1 = u1 + c2
        c4 = self.conv4(a1)
        u2 = self.up2(c4)
        a2 = u2 + c1
        out = self.out_block(a2)
        return out
