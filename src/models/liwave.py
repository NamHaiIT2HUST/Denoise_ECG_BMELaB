import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transforms.dwt1d import DWT1D, IDWT1D
from src.models.blocks import SE1D


class ResBlock1D(nn.Module):
    def __init__(self, channels, k=3, drop=0.0, use_bn=True):
        super().__init__()
        pad = (k - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, bias=not use_bn)
        self.bn1 = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, bias=not use_bn)
        self.bn2 = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class ResSEBlock1D(nn.Module):
    def __init__(self, channels, depth, use_se=False, se_reduction=8):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock1D(channels) for _ in range(depth)]) if depth > 0 else nn.Identity()
        self.se = SE1D(channels, r=se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.blocks(x))


class LiWave(nn.Module):
    def __init__(self, base=32, depth1=2, depth2=2, depth_mid=3, use_bn=True, use_se_mid=True, se_reduction=8, wavelet='haar'):
        super().__init__()
        C = base
        self.stem = nn.Sequential(nn.Conv1d(1, C, kernel_size=3, padding=1, bias=not use_bn), nn.BatchNorm1d(C) if use_bn else nn.Identity(), nn.SiLU(inplace=True))
        self.enc1 = ResSEBlock1D(C, depth1, use_se=False, se_reduction=se_reduction)
        self.dwt1 = DWT1D(in_channels=C, wavelet=wavelet)
        self.skip1 = ResSEBlock1D(C, 1, use_se=False, se_reduction=se_reduction)
        self.enc2 = ResSEBlock1D(C, depth1, use_se=False, se_reduction=se_reduction)
        self.dwt2 = DWT1D(in_channels=C, wavelet=wavelet)
        self.skip2 = ResSEBlock1D(C, 1, use_se=False, se_reduction=se_reduction)
        self.mid = ResSEBlock1D(C, depth_mid, use_se=use_se_mid, se_reduction=se_reduction)
        self.dec2 = ResSEBlock1D(C, depth2, use_se=False, se_reduction=se_reduction)
        self.idwt2 = IDWT1D(in_channels=2*C, wavelet=wavelet)
        self.dec1 = ResSEBlock1D(C, depth2, use_se=False, se_reduction=se_reduction)
        self.idwt1 = IDWT1D(in_channels=2*C, wavelet=wavelet)
        self.out_conv = nn.Conv1d(C, 1, kernel_size=3, padding=1)

    def _call_dwt(self, dwt, x):
        out = dwt(x)
        return torch.chunk(out, 2, dim=1)

    def _call_idwt(self, idwt, low, high):
        return idwt(torch.cat([low, high], dim=1))

    def forward(self, x):
        f0 = self.stem(x)
        e1 = self.enc1(f0)
        a1, d1 = self._call_dwt(self.dwt1, e1)
        s1 = self.skip1(d1)
        e2 = self.enc2(a1)
        a2, d2 = self._call_dwt(self.dwt2, e2)
        s2 = self.skip2(d2)
        b = self.mid(a2)
        d2l = self.dec2(b)
        u1 = self._call_idwt(self.idwt2, d2l, s2)
        d1l = self.dec1(u1)
        u0 = self._call_idwt(self.idwt1, d1l, s1)
        return self.out_conv(u0)
