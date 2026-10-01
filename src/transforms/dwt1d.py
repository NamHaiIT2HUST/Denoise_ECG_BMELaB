import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def _wavelet_filters(wavelet, device=None, dtype=None):
    wav = pywt.Wavelet(wavelet)
    dec_lo = torch.tensor(wav.dec_lo[::-1], device=device, dtype=dtype)
    dec_hi = torch.tensor(wav.dec_hi[::-1], device=device, dtype=dtype)
    rec_lo = torch.tensor(wav.rec_lo, device=device, dtype=dtype)
    rec_hi = torch.tensor(wav.rec_hi, device=device, dtype=dtype)
    return dec_lo, dec_hi, rec_lo, rec_hi


class DWT1D(nn.Module):
    def __init__(self, in_channels=1, wavelet="haar"):
        super().__init__()
        pywt.Wavelet(wavelet)
        self.in_channels = in_channels
        self.wavelet = wavelet

    def forward(self, x):
        b, c, length = x.shape
        dec_lo, dec_hi, _, _ = _wavelet_filters(self.wavelet, device=x.device, dtype=x.dtype)
        k = dec_lo.numel()

        if self.wavelet == "haar":
            if length % 2 != 0:
                x = F.pad(x, (0, 1), mode="replicate")
            h0 = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], device=x.device, dtype=x.dtype)
            h1 = torch.tensor([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], device=x.device, dtype=x.dtype)
            weight_low = h0.view(1, 1, 2).repeat(c, 1, 1)
            weight_high = h1.view(1, 1, 2).repeat(c, 1, 1)
            low = F.conv1d(x, weight_low, stride=2, padding=0, groups=c)
            high = F.conv1d(x, weight_high, stride=2, padding=0, groups=c)
            return torch.cat([low, high], dim=1)

        odd_pad = length % 2
        pad_total = max(k - 2, 0) + odd_pad
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = F.pad(x, (pad_left, pad_right), mode="reflect")
        weight_low = dec_lo.view(1, 1, k).repeat(c, 1, 1)
        weight_high = dec_hi.view(1, 1, k).repeat(c, 1, 1)
        low = F.conv1d(x, weight_low, stride=2, padding=0, groups=c)
        high = F.conv1d(x, weight_high, stride=2, padding=0, groups=c)
        return torch.cat([low, high], dim=1)


class IDWT1D(nn.Module):
    def __init__(self, in_channels=2, wavelet="haar"):
        super().__init__()
        pywt.Wavelet(wavelet)
        self.in_channels = in_channels
        self.wavelet = wavelet

    def forward(self, x):
        b, c2, length = x.shape
        assert c2 % 2 == 0, "IDWT expects concatenated low/high channels."
        c = c2 // 2
        low, high = torch.chunk(x, 2, dim=1)

        if self.wavelet == "haar":
            h0 = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], device=x.device, dtype=x.dtype)
            h1 = torch.tensor([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], device=x.device, dtype=x.dtype)
            up = torch.zeros((b, c, length * 2), device=x.device, dtype=x.dtype)
            up[:, :, ::2] = (low * h0[0]) + (high * h1[0])
            up[:, :, 1::2] = (low * h0[1]) + (high * h1[1])
            return up

        _, _, rec_lo, rec_hi = _wavelet_filters(self.wavelet, device=x.device, dtype=x.dtype)
        k = rec_lo.numel()
        interleaved = torch.stack([low, high], dim=2).reshape(b, 2 * c, length)
        weight = torch.zeros((2 * c, 1, k), device=x.device, dtype=x.dtype)
        for ch in range(c):
            weight[2 * ch, 0, :] = rec_lo
            weight[2 * ch + 1, 0, :] = rec_hi
        out = F.conv_transpose1d(interleaved, weight, stride=2, padding=0, groups=c)
        crop_left = max((k - 2) // 2, 0)
        return out[..., crop_left:crop_left + length * 2]
