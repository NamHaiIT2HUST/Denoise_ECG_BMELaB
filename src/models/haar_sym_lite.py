import torch
import torch.nn as nn

from src.models.blocks import ResSEBlock1D
from src.transforms.dwt1d import DWT1D, IDWT1D


class HaarSymLite(nn.Module):
    """
    Standard Symmetric Wavelet ECG Denoiser (Unified ResSE Architecture).
    
    Refined design following mentor's guidance:
    - Unified ResSEBlock1D (k1=7, k2=3, SiLU, SE) used across all stages.
    - Symmetric 2-level Haar DWT analysis and IDWT synthesis.
    - High-frequency detail subbands D1, D2 filtered via ResSEBlock1D skip connections.
    - Global residual addition: y_hat = x + delta.
    - Total parameters: 75,133 (~75.1K).
    """

    def __init__(
        self,
        base=32,
        mid_depth=1,
        se_reduction=8,
        wavelet="haar",
        k1=7,
        k2=3,
        **kwargs,
    ):
        super().__init__()
        c = base
        self.base = base
        self.mid_depth = mid_depth
        self.wavelet = wavelet

        # Stem: Conv 7x1 -> BN -> SiLU
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=k1, padding=k1 // 2, bias=False),
            nn.BatchNorm1d(c),
            nn.SiLU(inplace=True),
        )

        # Encoder 1: ResSE -> DWT1
        self.enc1 = ResSEBlock1D(c, k1=k1, k2=k2, r=se_reduction)
        self.dwt1 = DWT1D(in_channels=c, wavelet=wavelet)
        self.skip1 = ResSEBlock1D(c, k1=k1, k2=k2, r=se_reduction)

        # Encoder 2: ResSE -> DWT2
        self.enc2 = ResSEBlock1D(c, k1=k1, k2=k2, r=se_reduction)
        self.dwt2 = DWT1D(in_channels=c, wavelet=wavelet)
        self.skip2 = ResSEBlock1D(c, k1=k1, k2=k2, r=se_reduction)

        # Bottleneck
        self.mid = nn.Sequential(
            *[ResSEBlock1D(c, k1=k1, k2=k2, r=se_reduction) for _ in range(mid_depth)]
        )

        # Decoder 2: IDWT -> ResSE
        self.idwt2 = IDWT1D(in_channels=2 * c, wavelet=wavelet)
        self.dec2 = ResSEBlock1D(c, k1=k1, k2=k2, r=se_reduction)

        # Decoder 1: IDWT -> ResSE
        self.idwt1 = IDWT1D(in_channels=2 * c, wavelet=wavelet)
        self.dec1 = ResSEBlock1D(c, k1=k1, k2=k2, r=se_reduction)

        # Output projection
        self.out_conv = nn.Conv1d(c, 1, kernel_size=k1, padding=k1 // 2)
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

        # Level 1 Analysis
        e1 = self.enc1(f0)
        a1, d1 = self._split_wavelet(self.dwt1(e1))
        s1 = self.skip1(d1)

        # Level 2 Analysis
        e2 = self.enc2(a1)
        a2, d2 = self._split_wavelet(self.dwt2(e2))
        s2 = self.skip2(d2)

        # Bottleneck
        b = self.mid(a2)

        # Level 2 Synthesis (detail skip s2 filtered by ResSE)
        u1 = self.idwt2(self._merge_wavelet(b, s2))
        dec2_out = self.dec2(u1[..., :a1.shape[-1]])

        # Level 1 Synthesis (detail skip s1 filtered by ResSE)
        u0 = self.idwt1(self._merge_wavelet(dec2_out, s1))
        dec1_out = self.dec1(u0[..., :length])

        # Global residual learning
        delta = self.out_conv(dec1_out)
        return x + delta

    def bottleneck(self, x):
        """Extract bottleneck features for downstream classification."""
        f0 = self.stem(x)
        e1 = self.enc1(f0)
        a1, _ = self._split_wavelet(self.dwt1(e1))
        e2 = self.enc2(a1)
        a2, _ = self._split_wavelet(self.dwt2(e2))
        return self.mid(a2)
