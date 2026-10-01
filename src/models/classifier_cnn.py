"""Phan loai AAMI (da lead) tren tin hieu DA KHU NHIEU, RR co nhanh rieng.

x (B, n_leads, L) -> denoiser ap tung lead (freeze) -> CNN encoder -> SE -> pool
-> Linear (morph) ; RR (6) -> BN -> Linear (rr) ; ghep -> head -> n_classes.
"""
import torch
import torch.nn as nn

from src.models.blocks import SE1D
from src.models.quantum_head import QuantumClassifierHead


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.Mish(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        identity = x
        out = self.net(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class CNNEncoder1D(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3, bias=False),
            nn.BatchNorm1d(base),
            nn.Mish(inplace=True),
            nn.MaxPool1d(2)
        )
        self.layer1 = BasicBlock1D(base, base * 2, stride=2)
        self.layer2 = BasicBlock1D(base * 2, base * 4, stride=2)
        self.out_channels = base * 4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class DenoiseCNNClassifier(nn.Module):
    def __init__(self, denoiser=None, n_classes=4, head='quantum', encoding='amplitude',
                 pool_k=8, base=16, use_denoise=True, freeze_denoiser=True,
                 rr_dim=6, morph_dim=128, rr_hidden=32, n_leads=1, z_norm=True, dropout=0.3,
                 n_context=1):
        super().__init__()
        self.denoiser = denoiser
        self.use_denoise = use_denoise and (denoiser is not None)
        self.freeze_denoiser = freeze_denoiser
        self.n_leads = n_leads
        self.z_norm = z_norm                          # chuan hoa SAU denoiser
        if self.denoiser is not None and freeze_denoiser:
            for p in self.denoiser.parameters():
                p.requires_grad = False

        self.in_ch = n_leads * n_context          # kenh = lead x nhip ngu canh
        self.cnn = CNNEncoder1D(in_ch=self.in_ch, base=base)
        self.se = SE1D(self.cnn.out_channels)
        self.pool = nn.AdaptiveAvgPool1d(pool_k)
        self.rr_dim = rr_dim
        self.morph = nn.Sequential(nn.Linear(self.cnn.out_channels * pool_k, morph_dim),
                                   nn.GELU(), nn.Dropout(dropout))
        self.rr_mlp = nn.Sequential(nn.BatchNorm1d(rr_dim), nn.Linear(rr_dim, rr_hidden), nn.GELU())
        self.head = QuantumClassifierHead(in_features=morph_dim + rr_hidden, encoding=encoding,
                                          mode=head, n_classes=n_classes)

    def _denoise(self, x):                           # x: (B, n_leads, L) -> khu nhieu tung lead
        leads = []
        for l in range(x.size(1)):
            xl = x[:, l:l + 1, :]
            if self.freeze_denoiser:
                with torch.no_grad():
                    xl = self.denoiser(xl)
            else:
                xl = self.denoiser(xl)
            leads.append(xl)
        return torch.cat(leads, dim=1)

    def forward(self, x, rr=None):                   # x: (B, n_leads, L)
        if self.use_denoise:
            x = self._denoise(x)                     # denoiser nhan thang mV THO
        if self.z_norm:                              # chuan hoa SAU denoiser
            m = x.mean(dim=-1, keepdim=True)
            s = x.std(dim=-1, keepdim=True) + 1e-6
            x = (x - m) / s
        f = self.pool(self.se(self.cnn(x)))
        morph = self.morph(torch.flatten(f, 1))
        if rr is None:
            rr = torch.zeros(x.size(0), self.rr_dim, device=x.device, dtype=morph.dtype)
        rrf = self.rr_mlp(rr)
        return self.head(torch.cat([morph, rrf], dim=1))
