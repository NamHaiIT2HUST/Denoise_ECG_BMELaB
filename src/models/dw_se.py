import torch.nn as nn
from src.transforms.dwt1d import DWT1D, IDWT1D
from src.models.blocks import SE1D


class DWConvBlock(nn.Module):
    def __init__(self, in_channels, filters, num_layers=5, final_filters=None, use_se=False, se_r=8, final_activation=True):
        super().__init__()
        layers = [nn.Conv1d(in_channels, filters, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Conv1d(filters, filters, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        out_ch = filters
        if final_filters is not None:
            layers.append(nn.Conv1d(filters, final_filters, kernel_size=3, padding=1))
            if final_activation:
                layers.append(nn.ReLU(inplace=True))
            out_ch = final_filters
        self.net = nn.Sequential(*layers)
        self.se = SE1D(out_ch, r=se_r) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.net(x))


class OutputConvBlock(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=64, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class DW_SE(nn.Module):
    def __init__(self, wavelet='haar', se_r=8):
        super().__init__()
        self.conv1 = DWConvBlock(1, 64, num_layers=5, use_se=False, se_r=se_r)
        self.pool1 = DWT1D(in_channels=64, wavelet=wavelet)
        self.conv2 = DWConvBlock(128, 64, num_layers=5, use_se=True, se_r=se_r)
        self.pool2 = DWT1D(in_channels=64, wavelet=wavelet)
        self.latent = DWConvBlock(128, 64, num_layers=5, final_filters=128, use_se=True, se_r=se_r)
        self.up1 = IDWT1D(in_channels=128, wavelet=wavelet)
        self.conv4 = DWConvBlock(64, 64, num_layers=5, final_filters=128, use_se=True, se_r=se_r)
        self.up2 = IDWT1D(in_channels=128, wavelet=wavelet)
        self.out_block = OutputConvBlock()

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
        return self.out_block(a2)
