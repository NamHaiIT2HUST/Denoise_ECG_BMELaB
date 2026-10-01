import torch
import torch.nn as nn

from src.models.blocks import SE1D, CBAM1D
from src.transforms.dwt1d import DWT1D, IDWT1D


class InvertedSEBlock1D(nn.Module):
    def __init__(self, channels, expansion=4, kernel_size=7, se_reduction=8):
        super().__init__()
        hidden = channels * expansion
        padding = (kernel_size - 1) // 2

        self.expand = nn.Conv1d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden)
        self.project = nn.Conv1d(hidden, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se = CBAM1D(channels, r=se_reduction)
        self.act = nn.Mish(inplace=True)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.expand(x)))
        out = self.act(self.bn2(self.depthwise(out)))
        out = self.bn3(self.project(out))
        out = self.se(out)
        return self.act(residual + self.gamma * out)


class DetailGateBlock1D(nn.Module):
    def __init__(self, channels, se_reduction=8):
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.gate = CBAM1D(channels, r=se_reduction)
        self.act = nn.Mish(inplace=True)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        detail = self.act(self.bn(self.pw(self.dw(x))))
        detail = self.gate(detail)
        return x + self.gamma * detail


class HaarSymLite(nn.Module):
    """
    Fast symmetric wavelet ECG denoiser.

    Default design:
    - base channels = 32
    - inverted depthwise-separable SE blocks
    - two-level DWT/IDWT encoder-decoder
    - gated high-frequency detail skips
    - residual output learning: y_hat = x + delta

    With base=32, expansion=4, mid_depth=3, the model has about 70k
    trainable parameters. The wavelet can be swept with PyWavelets names.
    """

    def __init__(
        self,
        base=32,
        expansion=4,
        mid_depth=3,
        se_reduction=8,
        wavelet="haar",
    ):
        super().__init__()
        c = base
        self.base = base
        self.expansion = expansion
        self.mid_depth = mid_depth
        self.wavelet = wavelet

        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.Mish(inplace=True),
        )
        self.enc1 = InvertedSEBlock1D(c, expansion=expansion, kernel_size=7, se_reduction=se_reduction)
        self.dwt1 = DWT1D(in_channels=c, wavelet=wavelet)
        self.skip1 = DetailGateBlock1D(c, se_reduction=se_reduction)

        self.enc2 = InvertedSEBlock1D(c, expansion=expansion, kernel_size=7, se_reduction=se_reduction)
        self.dwt2 = DWT1D(in_channels=c, wavelet=wavelet)
        self.skip2 = DetailGateBlock1D(c, se_reduction=se_reduction)

        self.mid = nn.Sequential(
            *[
                InvertedSEBlock1D(c, expansion=expansion, kernel_size=7, se_reduction=se_reduction)
                for _ in range(mid_depth)
            ]
        )

        self.dec2 = InvertedSEBlock1D(c, expansion=expansion, kernel_size=7, se_reduction=se_reduction)
        self.idwt2 = IDWT1D(in_channels=2 * c, wavelet=wavelet)
        self.dec1 = InvertedSEBlock1D(c, expansion=expansion, kernel_size=7, se_reduction=se_reduction)
        self.idwt1 = IDWT1D(in_channels=2 * c, wavelet=wavelet)
        self.out_conv = nn.Conv1d(c, 1, kernel_size=3, padding=1)

        nn.init.normal_(self.out_conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _split_wavelet(x):
        return torch.chunk(x, 2, dim=1)

    @staticmethod
    def _merge_wavelet(low, high):
        return torch.cat([low, high], dim=1)

    def forward(self, x):
        length = x.shape[-1]
        f0 = self.stem(x)

        e1 = self.enc1(f0)
        a1, d1 = self._split_wavelet(self.dwt1(e1))
        s1 = self.skip1(d1)

        e2 = self.enc2(a1)
        a2, d2 = self._split_wavelet(self.dwt2(e2))
        s2 = self.skip2(d2)

        b = self.mid(a2)
        low2 = self.dec2(b)
        u1 = self.idwt2(self._merge_wavelet(low2, s2))
        low1 = self.dec1(u1[..., :a1.shape[-1]])
        u0 = self.idwt1(self._merge_wavelet(low1, s1))

        delta = self.out_conv(u0[..., :length])
        return x + delta

    def bottleneck(self, x):
        """Tra ve dac trung o day (bottleneck) sau 2 tang DWT + mid. Shape (B, base, N/4).

        Dung cho nhanh phan loai (2 pha: denoise -> freeze -> classify).
        """
        f0 = self.stem(x)
        e1 = self.enc1(f0)
        a1, _ = self._split_wavelet(self.dwt1(e1))
        e2 = self.enc2(a1)
        a2, _ = self._split_wavelet(self.dwt2(e2))
        return self.mid(a2)
