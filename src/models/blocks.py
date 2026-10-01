import torch
import torch.nn as nn
import torch.nn.functional as F


class SE1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.fc1 = nn.Conv1d(ch, hid, kernel_size=1)
        self.fc2 = nn.Conv1d(hid, ch, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class SE2D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.fc1 = nn.Conv2d(ch, hid, kernel_size=1)
        self.fc2 = nn.Conv2d(hid, ch, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=(-2, -1), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class LiteResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.pw(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dw(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out
class ChannelAttention1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.mlp = nn.Sequential(
            nn.Linear(ch, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, ch)
        )

    def forward(self, x):
        avg_out = self.mlp(x.mean(dim=-1))
        max_out = self.mlp(x.amax(dim=-1))
        return torch.sigmoid(avg_out + max_out).unsqueeze(-1)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


class CBAM1D(nn.Module):
    def __init__(self, ch, r=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(ch, r)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
