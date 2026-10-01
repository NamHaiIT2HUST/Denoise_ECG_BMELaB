"""PyTorch re-implementation of DeepFilter (Model I / LANL / dilated).

Romero, F. P., Piñol, D. C., & Vázquez-Seisdedos, C. R. (2021).
"DeepFilter: An ECG baseline wander removal filter using deep learning techniques."
Biomedical Signal Processing and Control, 70, 102992.

Ported by hand from the authors' Keras/TensorFlow source
(github.com/fperdigon/DeepFilter, deepFilter/dl_models.py,
`deep_filter_model_I_LANL_dilated`) to match this project's PyTorch pipeline.
This is a faithful re-implementation for benchmarking purposes, not the
authors' official code -- see ROADMAP.md #4.

Kept intentionally, including a quirk of the original: LANLFilter_module_dilated
splits `layers` into 6 branches via integer division, so a nominal width of
64/32/16 actually yields 60/30/12 channels. Reproduced as-is so the comparison
uses the architecture that was actually published, not a "corrected" variant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LANLModule(nn.Module):
    """Concatenation of linear + ReLU multi-kernel-size conv branches.

    dilation=1 -> 4 kernel sizes (3,5,9,15) x {linear, relu} = 8 branches (LANLFilter_module)
    dilation=3 -> 3 kernel sizes (5,9,15)   x {linear, relu} = 6 branches (LANLFilter_module_dilated)
    """

    def __init__(self, in_channels, layers, dilation=1):
        super().__init__()
        kernels = [3, 5, 9, 15] if dilation == 1 else [5, 9, 15]
        n_groups = len(kernels) * 2
        ch = layers // n_groups
        self.out_channels = ch * n_groups
        self.n_linear = len(kernels)

        convs = []
        for k in kernels:
            pad = dilation * (k - 1) // 2
            convs.append(nn.Conv1d(in_channels, ch, kernel_size=k, dilation=dilation, padding=pad))
        for k in kernels:
            pad = dilation * (k - 1) // 2
            convs.append(nn.Conv1d(in_channels, ch, kernel_size=k, dilation=dilation, padding=pad))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = []
        for i, conv in enumerate(self.convs):
            y = conv(x)
            if i >= self.n_linear:
                y = F.relu(y)
            outs.append(y)
        return torch.cat(outs, dim=1)


class DeepFilter(nn.Module):
    """Deep Filter Model I / LANL / dilated -- direct (non-residual) denoiser.

    Unlike haar_sym_lite, the original model regresses the clean signal
    directly (no learned residual add), so forward() returns the prediction
    as-is. Fully conv, stride=1, 'same' padding throughout -> shape-agnostic,
    works on this project's 8192-sample segments unchanged.
    """

    def __init__(self, dropout=0.4):
        super().__init__()
        stages = [
            (1, 64, 1), (None, 64, 3),
            (None, 32, 1), (None, 32, 3),
            (None, 16, 1), (None, 16, 3),
        ]
        blocks, norms = [], []
        in_ch = 1
        for _, layers, dilation in stages:
            blk = _LANLModule(in_ch, layers, dilation=dilation)
            blocks.append(blk)
            norms.append(nn.BatchNorm1d(blk.out_channels))
            in_ch = blk.out_channels
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv1d(in_ch, 1, kernel_size=9, padding=4)

    def forward(self, x):
        for blk, bn in zip(self.blocks, self.norms):
            x = blk(x)
            x = self.dropout(x)
            x = bn(x)
        return self.out_conv(x)
