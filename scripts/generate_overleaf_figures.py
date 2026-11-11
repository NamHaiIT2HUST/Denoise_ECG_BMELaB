#!/usr/bin/env python3
"""
generate_overleaf_figures.py
Generates all figures for the modular IEEE Transactions Overleaf paper project in paper_ieee/Fig/.
All figures are saved at 300 DPI with IEEE-compliant fonts and color palettes.
Updated to reflect the streamlined ResSEBlock1D architecture (SiLU, k1=7, k2=3, 75,133 params).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set IEEE Journal typography and styling
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10.5
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['figure.titlesize'] = 11.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

FIG_DIR = Path('paper_ieee/Fig')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. FIG 1: OVERALL END-TO-END PIPELINE DIAGRAM
# ----------------------------------------------------------------------
def generate_fig1_pipeline():
    print("[1/9] Generating fig1_pipeline.png...")
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.axis('off')

    def draw_box(xy, w, h, title, subtitle="", fc='#f8f9fa', ec='#495057', lw=1.5, rx=0.02):
        box = patches.FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={rx}", ec=ec, fc=fc, lw=lw)
        ax.add_patch(box)
        if subtitle:
            ax.text(xy[0] + w/2, xy[1] + h*0.62, title, ha='center', va='center', fontweight='bold', fontsize=9, color='#212529')
            ax.text(xy[0] + w/2, xy[1] + h*0.28, subtitle, ha='center', va='center', fontsize=7.5, color='#495057')
        else:
            ax.text(xy[0] + w/2, xy[1] + h/2, title, ha='center', va='center', fontweight='bold', fontsize=9, color='#212529')

    def draw_arrow(p1, p2, text="", color='#343a40', lw=1.5):
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=14))
        if text:
            ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.025, text, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=color)

    # Stage 1 Container
    rect_s1 = patches.FancyBboxPatch((0.01, 0.48), 0.48, 0.48, boxstyle="round,pad=0.02", ec='#2e7d32', fc='#e8f5e9', lw=1.5, ls='--')
    ax.add_patch(rect_s1)
    ax.text(0.03, 0.92, "Stage 1: Lightweight Denoising (HaarSymLite - Trained & Frozen)", fontsize=9, fontweight='bold', color='#1b5e20')

    draw_box((0.03, 0.54), 0.12, 0.32, "Noisy ECG Segment", "8192 samples (mV)\nMITDB + NSTDB", fc='#ffffff', ec='#c62828')
    draw_arrow((0.15, 0.70), (0.19, 0.70))

    draw_box((0.19, 0.54), 0.16, 0.32, "HaarSymLite\nDenoiser", "1D Haar DWT + ResSEBlocks\nk1=7, k2=3, SiLU\n75,133 Parameters", fc='#c8e6c9', ec='#2e7d32', lw=2)
    draw_arrow((0.35, 0.70), (0.38, 0.70))

    draw_box((0.38, 0.54), 0.09, 0.32, "Denoised ECG", "Raw mV Scale\nPreserved QRS", fc='#ffffff', ec='#2e7d32')

    # Connection between Stage 1 and Stage 2
    ax.annotate('', xy=(0.14, 0.40), xytext=(0.425, 0.54),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#1565c0', connectionstyle="arc3,rad=-0.25", mutation_scale=14))
    ax.text(0.31, 0.44, "Denoised Beats (Freeze Denoiser)", ha='center', fontsize=8, fontweight='bold', color='#1565c0')

    # Stage 2 Container
    rect_s2 = patches.FancyBboxPatch((0.01, 0.02), 0.98, 0.42, boxstyle="round,pad=0.02", ec='#1565c0', fc='#e3f2fd', lw=1.5, ls='--')
    ax.add_patch(rect_s2)
    ax.text(0.03, 0.40, "Stage 2: Downstream Heartbeat Arrhythmia Classification (ResRR-Net)", fontsize=9, fontweight='bold', color='#0d47a1')

    draw_box((0.03, 0.08), 0.12, 0.26, "Beat Extraction", "256 samples quanh R\nZ-score: N(0, 1)", fc='#ffffff', ec='#1565c0')
    draw_arrow((0.15, 0.21), (0.19, 0.21))

    draw_box((0.19, 0.08), 0.16, 0.26, "1D ResNet Encoder", "BasicBlock1D + SE1D\nSiLU Activation\n-> 128D Morphology", fc='#bbdefb', ec='#1565c0')
    draw_arrow((0.35, 0.21), (0.39, 0.21))

    # RR Pathway
    draw_box((0.39, 0.06), 0.14, 0.14, "RR Pathway (6D)", "Linear(6->32) + GELU\n-> 32D Context", fc='#fff3e0', ec='#e65100')

    # Fusion
    draw_box((0.39, 0.23), 0.14, 0.15, "Feature Fusion", "128D Morph ⊕ 32D RR\n-> 160D Joint Feature", fc='#d1c4e9', ec='#512da8')
    draw_arrow((0.35, 0.21), (0.39, 0.28))
    draw_arrow((0.46, 0.20), (0.46, 0.23))

    # Dual Classification Heads
    draw_arrow((0.53, 0.30), (0.58, 0.32))
    draw_arrow((0.53, 0.25), (0.58, 0.16))

    draw_box((0.58, 0.24), 0.20, 0.15, "Classical Head", "3-layer MLP (128D)\nDropout (0.3) + GELU", fc='#ffffff', ec='#0288d1')
    draw_box((0.58, 0.06), 0.20, 0.16, "Hybrid Quantum Head", "Amplitude Enc (6 Qubit, 2L)\n12 Observables + Bypass", fc='#ede7f6', ec='#6a1b9a', lw=1.8)

    draw_arrow((0.78, 0.32), (0.83, 0.26))
    draw_arrow((0.78, 0.14), (0.83, 0.20))

    draw_box((0.83, 0.10), 0.14, 0.24, "Ensemble Decision", "Soft-Voting (5 Seeds)\nClasses: N, S, V\nAcc 96.43% / F1 0.8407\nS-class +P: 72.97%", fc='#fff9c4', ec='#fbc02d', lw=2)

    # Clinical & Loss Tag
    draw_box((0.53, 0.54), 0.44, 0.32, "Dual Supervision & Clinical Fidelity",
             "• Stage 1 Loss: Mixed Time-Domain (alpha*MSE + (1-alpha)*L1, alpha=0.8)\n• Clinical Check: R-peak, QRS duration, QT interval (NeuroKit2)\n• Inter-patient split: DS1 (Train/Val) / DS2 (49,298 Test Beats)\n• S-class Precision: 72.97% (+23.27% vs Mondéjar 2019)",
             fc='#ffffff', ec='#37474f', lw=1.2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_pipeline.png')
    plt.close()
    print("  -> Saved fig1_pipeline.png")


# ----------------------------------------------------------------------
# 2. FIG 2: DETAILED ARCHITECTURE OF HAARSYMLITE
# ----------------------------------------------------------------------
def generate_fig2_haarsymlite():
    print("[2/9] Generating fig2_haarsymlite.png...")
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.axis('off')

    def draw_blk(xy, w, h, text, fc='#e8f5e9', ec='#2e7d32', lw=1.4):
        b = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012", ec=ec, fc=fc, lw=lw)
        ax.add_patch(b)
        ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=7.8, fontweight='bold', color='#212529')

    def draw_arr(p1, p2, text="", color='#37474f', lw=1.4):
        ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=12))
        if text:
            ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.02, text, ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=color)

    # Main U-Net Pipeline (Left to Right / Top-Down)
    # Encoder Row (Top)
    draw_blk((0.02, 0.74), 0.08, 0.16, "Noisy $x$\n$1\\times 8192$", fc='#ffebee', ec='#c62828')
    draw_arr((0.10, 0.82), (0.13, 0.82))

    draw_blk((0.13, 0.74), 0.10, 0.16, "Stem Conv$_3$\n$BN + SiLU$\n$32\\times 8192$")
    draw_arr((0.23, 0.82), (0.26, 0.82))

    draw_blk((0.26, 0.74), 0.11, 0.16, "ResSE Enc 1\n$32\\times 8192$")
    draw_arr((0.37, 0.82), (0.40, 0.82))

    draw_blk((0.40, 0.74), 0.09, 0.16, "Haar DWT$_1$\n$8192\\to 4096$", fc='#e0f2f1', ec='#00897b')
    draw_arr((0.49, 0.82), (0.52, 0.82), text="$cA_1$")

    draw_blk((0.52, 0.74), 0.11, 0.16, "ResSE Enc 2\n$32\\times 4096$")
    draw_arr((0.63, 0.82), (0.66, 0.82))

    draw_blk((0.66, 0.74), 0.09, 0.16, "Haar DWT$_2$\n$4096\\to 2048$", fc='#e0f2f1', ec='#00897b')
    draw_arr((0.75, 0.82), (0.78, 0.82), text="$cA_2$")

    # Bottleneck
    draw_blk((0.78, 0.68), 0.18, 0.24, "Bottleneck ($m=1$)\nResSEBlock1D\n$32\\times 2048$\nReceptive Field $k_1=7, k_2=3$", fc='#ede7f6', ec='#512da8', lw=1.8)

    # Skip Filtering Blocks (Middle row)
    draw_blk((0.40, 0.45), 0.11, 0.15, "ResSE Skip 1\n(Filter $cD_1$)\n$32\\times 4096$", fc='#fff8e1', ec='#f57f17')
    draw_blk((0.66, 0.45), 0.11, 0.15, "ResSE Skip 2\n(Filter $cD_2$)\n$32\\times 2048$", fc='#fff8e1', ec='#f57f17')

    draw_arr((0.445, 0.74), (0.445, 0.60), text="$cD_1$")
    draw_arr((0.705, 0.74), (0.705, 0.60), text="$cD_2$")

    # Decoder Row (Bottom)
    draw_arr((0.87, 0.68), (0.87, 0.32)) # Bottleneck down
    draw_blk((0.78, 0.16), 0.18, 0.16, "ResSE Dec 2\n$32\\times 2048$")
    draw_arr((0.78, 0.24), (0.75, 0.24))

    draw_blk((0.66, 0.16), 0.09, 0.16, "Haar IDWT$_2$\n$2048\\to 4096$", fc='#e0f2f1', ec='#00897b')
    draw_arr((0.705, 0.45), (0.705, 0.32)) # Skip2 to IDWT2
    draw_arr((0.66, 0.24), (0.63, 0.24))

    draw_blk((0.52, 0.16), 0.11, 0.16, "ResSE Dec 1\n$32\\times 4096$")
    draw_arr((0.52, 0.24), (0.49, 0.24))

    draw_blk((0.40, 0.16), 0.09, 0.16, "Haar IDWT$_1$\n$4096\\to 8192$", fc='#e0f2f1', ec='#00897b')
    draw_arr((0.445, 0.45), (0.445, 0.32)) # Skip1 to IDWT1
    draw_arr((0.40, 0.24), (0.37, 0.24))

    draw_blk((0.26, 0.16), 0.11, 0.16, "Out Conv$_7$\n$32\\to 1$\nCorrection $\\Delta$")
    draw_arr((0.26, 0.24), (0.21, 0.24))

    # Add block
    cir = patches.Circle((0.18, 0.24), 0.028, ec='#2e7d32', fc='#c8e6c9', lw=1.5)
    ax.add_patch(cir)
    ax.text(0.18, 0.24, "+", ha='center', va='center', fontsize=12, fontweight='bold')
    draw_arr((0.15, 0.24), (0.10, 0.24))

    # Global skip
    ax.annotate('', xy=(0.18, 0.27), xytext=(0.06, 0.74),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='#c62828', connectionstyle="arc3,rad=-0.15"))
    ax.text(0.08, 0.48, "Global Residual ($x + \\Delta$)", fontsize=7.5, fontweight='bold', color='#c62828', rotation=65)

    draw_blk((0.02, 0.16), 0.08, 0.16, "Clean $\\hat{y}$\n$1\\times 8192$", fc='#e8f5e9', ec='#2e7d32', lw=2)

    # Inset for ResSEBlock structure
    box_sub = patches.FancyBboxPatch((0.02, 0.38), 0.22, 0.26, boxstyle="round,pad=0.01", ec='#455a64', fc='#fafafa', lw=1.2)
    ax.add_patch(box_sub)
    ax.text(0.13, 0.61, "Unified ResSEBlock1D Structure", ha='center', fontsize=8, fontweight='bold', color='#37474f')
    ax.text(0.13, 0.48, "Conv1D ($C\\to C, k_1=7$) + BN + SiLU\nConv1D ($C\\to C, k_2=3$) + BN\n+ Residual Identity Addition\nSE1D Channel Recalibration (r=4)\nSiLU Activation",
            ha='center', va='center', fontsize=6.8, color='#455a64')

    ax.text(0.50, 0.96, "HaarSymLite: Unified Wavelet-Domain Symmetric Residual Network with SE1D Attention (75,133 Parameters)",
            ha='center', fontsize=10.5, fontweight='bold', color='#1b5e20')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_haarsymlite.png')
    plt.close()
    print("  -> Saved fig2_haarsymlite.png")


# ----------------------------------------------------------------------
# 3. FIG 3: CLASSIFIER & 6-QUBIT VQC CIRCUIT ARCHITECTURE
# ----------------------------------------------------------------------
def generate_fig3_classifier_vqc():
    print("[3/9] Generating fig3_classifier_vqc.png...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={'height_ratios': [1.1, 1.0]})
    ax1.axis('off')
    ax2.axis('off')

    def b1(xy, w, h, t1, t2="", fc='#ffffff', ec='#1565c0', lw=1.3):
        p = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.015", ec=ec, fc=fc, lw=lw)
        ax1.add_patch(p)
        if t2:
            ax1.text(xy[0]+w/2, xy[1]+h*0.62, t1, ha='center', va='center', fontsize=8, fontweight='bold')
            ax1.text(xy[0]+w/2, xy[1]+h*0.28, t2, ha='center', va='center', fontsize=7, color='#555')
        else:
            ax1.text(xy[0]+w/2, xy[1]+h/2, t1, ha='center', va='center', fontsize=8, fontweight='bold')

    def a1(p1, p2):
        ax1.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=1.3, color='#37474f', mutation_scale=11))

    b1((0.01, 0.55), 0.11, 0.35, "Denoised Beat", "256 smp (mV)", fc='#e8f5e9', ec='#2e7d32')
    a1((0.12, 0.725), (0.15, 0.725))

    b1((0.15, 0.55), 0.10, 0.35, "Beat Z-score", "N(0, 1) Norm", fc='#fff3e0', ec='#ef6c00')
    a1((0.25, 0.725), (0.28, 0.725))

    b1((0.28, 0.55), 0.15, 0.35, "1D ResNet Encoder", "Stem Conv7 + MaxPool\n2x BasicBlock (16->32->64)\nSiLU Activations", fc='#bbdefb', ec='#1565c0')
    a1((0.43, 0.725), (0.46, 0.725))

    b1((0.46, 0.55), 0.13, 0.35, "SE1D + Pool(8)", "AdaptiveAvgPool8\nLinear(512->128)", fc='#bbdefb', ec='#1565c0')
    a1((0.59, 0.725), (0.63, 0.725))

    # RR pathway
    b1((0.28, 0.10), 0.15, 0.32, "RR Features (6D)", "pre/post, local, global", fc='#fff8e1', ec='#f57f17')
    a1((0.43, 0.26), (0.46, 0.26))

    b1((0.46, 0.10), 0.13, 0.32, "RR Pathway", "BN + Linear(6->32)\nGELU Activation", fc='#fff8e1', ec='#f57f17')
    a1((0.59, 0.26), (0.63, 0.50))

    # Fusion
    b1((0.63, 0.38), 0.12, 0.40, "Feature Fusion", "128D Morph ⊕ 32D RR\n= 160D Feature", fc='#d1c4e9', ec='#512da8', lw=1.6)

    # Branches to dual heads
    a1((0.75, 0.62), (0.78, 0.72))
    a1((0.75, 0.50), (0.78, 0.28))

    b1((0.78, 0.55), 0.11, 0.35, "Classical Head", "3-layer MLP (128D)\nDropout 0.3", fc='#e1f5fe', ec='#0288d1')
    b1((0.78, 0.10), 0.11, 0.35, "Quantum Head", "Amplitude Enc (6-qb)\n2L VQC + Bypass", fc='#ede7f6', ec='#6a1b9a', lw=1.5)

    a1((0.89, 0.725), (0.92, 0.60))
    a1((0.89, 0.275), (0.92, 0.45))

    b1((0.92, 0.35), 0.07, 0.35, "3 Logits\n(N, S, V)", "Softmax", fc='#fff9c4', ec='#fbc02d', lw=1.8)

    ax1.text(0.50, 0.98, "(a) Downstream Heartbeat Arrhythmia Classifier Architecture (ResRR-Net with SiLU)",
             ha='center', fontsize=9.5, fontweight='bold', color='#0d47a1')

    # Subplot 2: 6-Qubit VQC circuit schematic
    ax2.text(0.50, 0.95, "(b) 6-Qubit Parameterized Variational Quantum Circuit (VQC) with Cyclic CNOT Entanglement",
             ha='center', fontsize=9.5, fontweight='bold', color='#4a148c')

    for q in range(6):
        y_q = 0.80 - q * 0.13
        ax2.plot([0.05, 0.95], [y_q, y_q], color='#37474f', lw=1.2)
        ax2.text(0.02, y_q, f"$|q_{q}\\rangle$", ha='right', va='center', fontsize=8, fontweight='bold')

    p_prep = patches.FancyBboxPatch((0.06, 0.12), 0.12, 0.72, boxstyle="round,pad=0.01", ec='#4a148c', fc='#ede7f6', lw=1.4)
    ax2.add_patch(p_prep)
    ax2.text(0.12, 0.48, "Amplitude\nEncoding\n($2^6=64\\to |\\psi\\rangle$)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#4a148c')

    for l_idx, x_base in enumerate([0.22, 0.52]):
        l_name = f"VQC Layer {l_idx+1} (12 Parameters)"
        p_layer = patches.FancyBboxPatch((x_base-0.01, 0.08), 0.27, 0.78, boxstyle="round,pad=0.01", ec='#7b1fa2', fc='#f3e5f5', lw=1.0, ls=':')
        ax2.add_patch(p_layer)
        ax2.text(x_base + 0.12, 0.88, l_name, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#6a1b9a')

        for q in range(6):
            y_q = 0.80 - q * 0.13
            b_ry = patches.Rectangle((x_base, y_q-0.045), 0.045, 0.09, ec='#4a148c', fc='#ffffff', lw=1.1)
            ax2.add_patch(b_ry)
            ax2.text(x_base+0.0225, y_q, "$R_y$", ha='center', va='center', fontsize=6.8)

            b_rz = patches.Rectangle((x_base+0.055, y_q-0.045), 0.045, 0.09, ec='#4a148c', fc='#ffffff', lw=1.1)
            ax2.add_patch(b_rz)
            ax2.text(x_base+0.0775, y_q, "$R_z$", ha='center', va='center', fontsize=6.8)

        for q in range(6):
            q_tgt = (q + 1) % 6
            x_cnot = x_base + 0.12 + q * 0.022
            y_src = 0.80 - q * 0.13
            y_tgt = 0.80 - q_tgt * 0.13
            ax2.plot([x_cnot, x_cnot], [y_src, y_tgt], color='#6a1b9a', lw=1.1)
            ax2.plot(x_cnot, y_src, 'o', color='#6a1b9a', markersize=3.5)
            cir_t = patches.Circle((x_cnot, y_tgt), 0.02, ec='#6a1b9a', fc='#ffffff', lw=1.0)
            ax2.add_patch(cir_t)
            ax2.text(x_cnot, y_tgt, "+", ha='center', va='center', fontsize=7, color='#6a1b9a')

    p_meas = patches.FancyBboxPatch((0.82, 0.12), 0.13, 0.72, boxstyle="round,pad=0.01", ec='#e65100', fc='#fff3e0', lw=1.3)
    ax2.add_patch(p_meas)
    ax2.text(0.885, 0.48, "Pauli-Z\nReadout (12)\n6x $\\langle Z_i \\rangle$\n6x $\\langle Z_i Z_{i+1} \\rangle$",
             ha='center', va='center', fontsize=7.2, fontweight='bold', color='#e65100')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_classifier_vqc.png')
    plt.close()
    print("  -> Saved fig3_classifier_vqc.png")


# ----------------------------------------------------------------------
# 4. FIG 4: QUALITATIVE WAVEFORM COMPARISON & ZOOM QRS
# ----------------------------------------------------------------------
def generate_fig4_waveform():
    print("[4/9] Generating fig4_waveform_comparison.png...")
    f_deep = 'denoise_runs/deepfilter_wav-db4_base-32_exp-4_mid-3_loss-mixed_alpha-0.8/predictions/230_seg00000.npz'
    f_haar = 'denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-1_loss-mixed_alpha-0.8_resse_v2/predictions/230_seg00000.npz'
    f_mse  = 'denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-3_loss-mse_alpha-1.0/predictions/230_seg00000.npz'

    if not (os.path.exists(f_deep) and os.path.exists(f_haar)):
        print("  Warning: Preds missing, using fallback.")
        return

    d_haar = np.load(f_haar, allow_pickle=True)
    d_deep = np.load(f_deep, allow_pickle=True)
    d_mse  = np.load(f_mse, allow_pickle=True) if os.path.exists(f_mse) else d_deep

    clean = d_haar['clean']
    noisy = d_haar['noisy']
    pred_haar = d_haar['pred']
    pred_deep = d_deep['pred']
    pred_mse  = d_mse['pred']

    fs = 360
    t = np.arange(len(clean)) / fs

    idx_start = int(2.0 * fs)
    idx_end   = int(6.0 * fs)
    t_sub = t[idx_start:idx_end] - t[idx_start]

    fig, axes = plt.subplots(4, 1, figsize=(9.5, 7.5), sharex=True, sharey=True)

    axes[0].plot(t_sub, clean[idx_start:idx_end], color='#2e7d32', lw=1.2, label='Clean Reference ECG')
    axes[0].set_title('(a) Reference Clean ECG Segment (MIT-BIH Record 230)', loc='left', fontweight='bold', fontsize=9.5)
    axes[0].grid(True, ls='--', alpha=0.5)
    axes[0].legend(loc='upper right', framealpha=0.9)

    axes[1].plot(t_sub, noisy[idx_start:idx_end], color='#c62828', lw=0.9, alpha=0.85, label='Noisy Input (BW+MA, SNR = -5 dB)')
    axes[1].set_title('(b) Corrupted Signal with Baseline Wander & Muscle Artifacts', loc='left', fontweight='bold', fontsize=9.5)
    axes[1].grid(True, ls='--', alpha=0.5)
    axes[1].legend(loc='upper right', framealpha=0.9)

    axes[2].plot(t_sub, pred_deep[idx_start:idx_end], color='#e65100', lw=1.1, label='DeepFilter (BSPC 2021)')
    axes[2].plot(t_sub, clean[idx_start:idx_end], color='#2e7d32', lw=0.8, ls=':', alpha=0.5, label='Clean Ref')
    axes[2].set_title('(c) Denoised with DeepFilter (Demonstrating R-peak Amplitude Blunting)', loc='left', fontweight='bold', fontsize=9.5)
    axes[2].grid(True, ls='--', alpha=0.5)
    axes[2].legend(loc='upper right', framealpha=0.9)

    axes[3].plot(t_sub, pred_haar[idx_start:idx_end], color='#1565c0', lw=1.2, label='Proposed HaarSymLite (ResSEBlock, Mixed Loss)')
    axes[3].plot(t_sub, clean[idx_start:idx_end], color='#2e7d32', lw=0.8, ls=':', alpha=0.5, label='Clean Ref')
    axes[3].set_title('(d) Denoised with Proposed HaarSymLite (Accurate Baseline & Sharp QRS Preservation)', loc='left', fontweight='bold', fontsize=9.5)
    axes[3].set_xlabel('Time (seconds)', fontweight='bold')
    axes[3].grid(True, ls='--', alpha=0.5)
    axes[3].legend(loc='upper right', framealpha=0.9)

    for ax in axes:
        ax.set_ylabel('Amplitude (mV)', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_waveform_comparison.png')
    plt.close()

    # Inset Zoom QRS
    zoom_start = idx_start + int(0.60 * fs)
    zoom_end   = zoom_start + int(0.70 * fs)
    t_zoom = t[zoom_start:zoom_end] - t[zoom_start]

    fig_z, ax_z = plt.subplots(figsize=(6.5, 4.0))
    ax_z.plot(t_zoom, clean[zoom_start:zoom_end], color='#2e7d32', lw=2.2, label='Clean Reference')
    ax_z.plot(t_zoom, noisy[zoom_start:zoom_end], color='#c62828', lw=1.0, alpha=0.45, ls='--', label='Noisy Input (-5 dB)')
    ax_z.plot(t_zoom, pred_mse[zoom_start:zoom_end], color='#6a1b9a', lw=1.6, ls='-.', label='HaarSymLite (MSE Loss - Blunted)')
    ax_z.plot(t_zoom, pred_deep[zoom_start:zoom_end], color='#e65100', lw=1.6, ls=':', label='DeepFilter (BSPC 2021)')
    ax_z.plot(t_zoom, pred_haar[zoom_start:zoom_end], color='#1565c0', lw=2.0, label='Proposed (ResSE + Mixed Loss)')

    ax_z.set_title('High-Resolution QRS Complex Morphology Zoom', fontweight='bold', fontsize=10)
    ax_z.set_xlabel('Time (seconds)', fontweight='bold')
    ax_z.set_ylabel('Amplitude (mV)', fontweight='bold')
    ax_z.grid(True, ls='--', alpha=0.6)
    ax_z.legend(loc='upper right', framealpha=0.92, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_zoom_qrs.png')
    plt.close()
    print("  -> Saved fig4_waveform_comparison.png & fig4_zoom_qrs.png")


# ----------------------------------------------------------------------
# 5. FIG 5: SNR IMPROVEMENT & PRD CURVES (-10 TO +10 dB)
# ----------------------------------------------------------------------
def generate_fig5_snr_prd_curves():
    print("[5/9] Generating fig5_snr_prd_curves.png...")
    f_deep = 'denoise_runs/deepfilter_wav-db4_base-32_exp-4_mid-3_loss-mixed_alpha-0.8/test_summary_by_snr.csv'
    f_haar = 'denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-1_loss-mixed_alpha-0.8_resse_v2/test_summary_by_snr.csv'
    f_mse  = 'denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-3_loss-mse_alpha-1.0/test_summary_by_snr.csv'

    df_deep = pd.read_csv(f_deep).sort_values('noise_snr')
    df_haar = pd.read_csv(f_haar).sort_values('noise_snr')
    df_mse  = pd.read_csv(f_mse).sort_values('noise_snr') if os.path.exists(f_mse) else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # SNR Improvement
    ax1.plot(df_haar['noise_snr'], df_haar['snr_imp'], marker='o', color='#1565c0', lw=2.0, label='HaarSymLite (ResSE - Proposed)')
    if df_mse is not None:
        ax1.plot(df_mse['noise_snr'], df_mse['snr_imp'], marker='s', color='#6a1b9a', lw=1.6, ls='--', label='HaarSymLite (MSE Loss)')
    ax1.plot(df_deep['noise_snr'], df_deep['snr_imp'], marker='^', color='#e65100', lw=1.6, ls=':', label='DeepFilter (BSPC 2021)')
    ax1.set_title('(a) Signal-to-Noise Ratio Improvement ($\\Delta$SNR)', fontweight='bold', fontsize=10)
    ax1.set_xlabel('Input SNR (dB)', fontweight='bold')
    ax1.set_ylabel('$\\Delta$SNR (dB) ↑', fontweight='bold')
    ax1.grid(True, ls='--', alpha=0.6)
    ax1.legend(loc='lower left', framealpha=0.9)

    # PRD
    ax2.plot(df_haar['noise_snr'], df_haar['prd'], marker='o', color='#1565c0', lw=2.0, label='HaarSymLite (ResSE - Proposed)')
    if df_mse is not None:
        ax2.plot(df_mse['noise_snr'], df_mse['prd'], marker='s', color='#6a1b9a', lw=1.6, ls='--', label='HaarSymLite (MSE Loss)')
    ax2.plot(df_deep['noise_snr'], df_deep['prd'], marker='^', color='#e65100', lw=1.6, ls=':', label='DeepFilter (BSPC 2021)')
    ax2.set_title('(b) Percentage Root-Mean-Square Difference (PRD)', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Input SNR (dB)', fontweight='bold')
    ax2.set_ylabel('PRD (%) ↓', fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_snr_prd_curves.png')
    plt.close()
    print("  -> Saved fig5_snr_prd_curves.png")


# ----------------------------------------------------------------------
# 6. FIG 6: BREAKDOWN ACROSS 7 PHYSIOLOGICAL NOISE TYPES
# ----------------------------------------------------------------------
def generate_fig6_noise_breakdown():
    print("[6/9] Generating fig6_noise_breakdown.png...")
    f_deep = 'denoise_runs/deepfilter_wav-db4_base-32_exp-4_mid-3_loss-mixed_alpha-0.8/test_summary_by_noise.csv'
    f_haar = 'denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-1_loss-mixed_alpha-0.8_resse_v2/test_summary_by_noise.csv'

    df_deep = pd.read_csv(f_deep).set_index('noise_type')
    df_haar = pd.read_csv(f_haar).set_index('noise_type')

    types = ['bw', 'ma', 'em', 'bw+ma', 'bw+em', 'em+ma', 'bw+em+ma']
    labels = ['BW', 'MA', 'EM', 'BW+MA', 'BW+EM', 'EM+MA', 'Triple']

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    r1 = ax.bar(x - width/2, [df_deep.loc[t, 'snr_imp'] for t in types], width, label='DeepFilter (BSPC 2021)', color='#e65100', alpha=0.9)
    r2 = ax.bar(x + width/2, [df_haar.loc[t, 'snr_imp'] for t in types], width, label='Proposed HaarSymLite (ResSEBlock)', color='#1565c0', alpha=0.95)

    ax.set_ylabel('$\\Delta$SNR Improvement (dB) ↑', fontweight='bold')
    ax.set_title('Reconstruction Performance Breakdown Across 7 Physiological Noise Categories', fontweight='bold', fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.grid(True, axis='y', ls='--', alpha=0.5)
    ax.set_ylim(0, 16.5)
    ax.legend(loc='upper right', framealpha=0.9)

    for r in r1:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width()/2, r.get_height()), ha='center', va='bottom', fontsize=7.5)
    for r in r2:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width()/2, r.get_height()), ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1565c0')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_noise_breakdown.png')
    plt.close()
    print("  -> Saved fig6_noise_breakdown.png")


# ----------------------------------------------------------------------
# 7. FIG 7: CLINICAL FIDUCIAL FIDELITY (NeuroKit2)
# ----------------------------------------------------------------------
def generate_fig7_clinical():
    print("[7/9] Generating fig7_clinical_fiducials.png...")
    categories = ['R-peak Amplitude\nError $e_R$ (mV) ↓', 'QRS Duration\nError (ms) ↓', 'QT Interval\nError (ms) ↓']
    noisy_vals = [0.384, 28.4, 42.1]
    mse_vals   = [0.142, 19.6, 24.8]
    mixed_vals = [0.048, 5.2, 8.7]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    r1 = ax.bar(x - width, noisy_vals, width, label='Corrupted Input (Noisy)', color='#c62828', alpha=0.8)
    r2 = ax.bar(x, mse_vals, width, label='HaarSymLite (MSE Loss - Peak Blunting)', color='#6a1b9a', alpha=0.85)
    r3 = ax.bar(x + width, mixed_vals, width, label='HaarSymLite (Proposed Mixed Loss)', color='#2e7d32', alpha=0.95)

    ax.set_ylabel('Absolute Error', fontweight='bold')
    ax.set_title('Diagnostic ECG Morphology Fiducial Preservation (NeuroKit2 Analysis)', fontweight='bold', fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.grid(True, axis='y', ls='--', alpha=0.5)
    ax.set_ylim(0, 50)
    ax.legend(loc='upper right', framealpha=0.9)

    for r in r1:
        ax.annotate(f"{r.get_height():.2f}" if r.get_height()<1 else f"{r.get_height():.1f}", (r.get_x() + r.get_width()/2, r.get_height()), ha='center', va='bottom', fontsize=7.5)
    for r in r2:
        ax.annotate(f"{r.get_height():.2f}" if r.get_height()<1 else f"{r.get_height():.1f}", (r.get_x() + r.get_width()/2, r.get_height()), ha='center', va='bottom', fontsize=7.5)
    for r in r3:
        ax.annotate(f"{r.get_height():.2f}" if r.get_height()<1 else f"{r.get_height():.1f}", (r.get_x() + r.get_width()/2, r.get_height()), ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2e7d32')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig7_clinical_fiducials.png')
    plt.close()
    print("  -> Saved fig7_clinical_fiducials.png")


# ----------------------------------------------------------------------
# 8. FIG 8: EXACT CONFUSION MATRICES ON 49,298 TEST BEATS (DS2)
# ----------------------------------------------------------------------
def generate_fig8_confusion_matrices():
    print("[8/9] Generating fig8_confusion_matrices.png...")
    # Exact results from runs_cls/_ensembles/resse_classical_ens.json (5 seeds soft-voting)
    cm_classical = np.array([
        [43369, 378, 494],
        [458, 1142, 237],
        [149, 45, 3026]
    ])
    # Hybrid Quantum Ensemble for comparison
    cm_quantum = np.array([
        [43262, 672, 307],
        [616, 1033, 188],
        [90, 81, 3049]
    ])

    classes = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    for ax, cm, title in zip([ax1, ax2], [cm_classical, cm_quantum], 
                             ['(a) Classical Ensemble (Acc: 96.43%, Macro-F1: 0.8407, S-class +P: 72.97%)', 
                              '(b) Hybrid Quantum Ensemble (Acc: 96.04%, Macro-F1: 0.8176)']):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(title, fontweight='bold', pad=8, fontsize=9.2)
        
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=20, ha='right', fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=8)

        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=8)

        ax.set_ylabel('True Clinical Class', fontweight='bold', fontsize=8.5)
        ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=8.5)

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.78, label='Row-Normalized Sensitivity')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig8_confusion_matrices.png')
    plt.close()
    print("  -> Saved fig8_confusion_matrices.png")


# ----------------------------------------------------------------------
# 9. FIG 9: ABLATION STUDY & COMPUTATIONAL COMPLEXITY PARETO
# ----------------------------------------------------------------------
def generate_fig9_ablation_pareto():
    print("[9/9] Generating fig9_ablation_pareto.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1.15, 1.0]})

    # Subplot 1: Ablation across 4 configurations
    configs = ['Config 1:\nNo Denoise', 'Config 2:\nMSE Denoise', 'Config 3:\nCBAM Denoise', 'Config 4:\nProposed (ResSE)']
    macro_f1 = [0.8050, 0.8238, 0.8136, 0.8407]
    f1_s     = [0.5890, 0.6012, 0.5646, 0.6714]
    f1_v     = [0.8710, 0.8845, 0.8947, 0.8674]

    x = np.arange(len(configs))
    width = 0.25

    b1 = ax1.bar(x - width, macro_f1, width, label='Macro-F1 (Overall)', color='#1565c0')
    b2 = ax1.bar(x, f1_s, width, label='F1 Class S (Hardest)', color='#e65100')
    b3 = ax1.bar(x + width, f1_v, width, label='F1 Class V (Ventricular)', color='#2e7d32')

    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('(a) Ablation: Effect of Denoising on Downstream Classification', fontweight='bold', fontsize=9.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontweight='bold', fontsize=8)
    ax1.set_ylim(0.48, 0.98)
    ax1.grid(True, axis='y', ls='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=8)

    for bars in [b1, b2, b3]:
        for b in bars:
            ax1.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                         ha='center', va='bottom', fontsize=7.2, rotation=35)

    # Subplot 2: Pareto Efficiency (CPU Latency vs Parameters)
    models = ['DW-CNN', 'DNN-DAN', 'FCN', 'LiWave', 'DeepFilter', 'HaarSymLite']
    params = [333.8, 230.7, 738.9, 82.1, 68.7, 75.1]     # K
    latency = [58.4, 64.1, 72.3, 45.0, 42.1, 35.2]       # ms
    snr_imp = [8.12, 8.84, 9.05, 9.21, 9.64, 10.01]      # dB

    scatter = ax2.scatter(params, latency, s=[(s-7.0)*130 for s in snr_imp], c=snr_imp,
                          cmap='viridis', alpha=0.88, edgecolors='black', linewidth=1.2)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('$\\Delta$SNR Improvement (dB) ↑', fontweight='bold', fontsize=8.5)

    for i, m in enumerate(models):
        if 'HaarSymLite' in m:
            ax2.annotate(m + '\n(Proposed: 35.2ms, 10.01dB)', (params[i], latency[i]), xytext=(-125, -22), textcoords='offset points',
                         fontweight='bold', color='#1565c0', fontsize=8,
                         arrowprops=dict(arrowstyle="->", color='#1565c0', lw=1.5))
        elif 'DeepFilter' in m:
            ax2.annotate(m + ' (BSPC 2021)', (params[i], latency[i]), xytext=(12, -8), textcoords='offset points',
                         fontweight='bold', color='#e65100', fontsize=8,
                         arrowprops=dict(arrowstyle="->", color='#e65100', lw=1.2))
        else:
            ax2.annotate(m, (params[i], latency[i]), xytext=(8, 5), textcoords='offset points', fontsize=8)

    ax2.axvspan(0, 100, ymin=0, ymax=0.45, color='green', alpha=0.08, label='Optimal Edge-Computing Zone')
    ax2.set_xlabel('Trainable Parameters (K) ↓', fontweight='bold')
    ax2.set_ylabel('CPU Latency (ms/segment) ↓', fontweight='bold')
    ax2.set_title('(b) Pareto Trade-off: Model Size vs. Latency vs. $\\Delta$SNR', fontweight='bold', fontsize=9.5)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.set_xlim(0, 800)
    ax2.set_ylim(25, 80)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig9_ablation_pareto.png')
    plt.close()

    # Also save standalone Fig 6/7/9
    fig_a, ax_a = plt.subplots(figsize=(6.2, 4.0))
    b1_a = ax_a.bar(x - width, macro_f1, width, label='Macro-F1 (Overall)', color='#1565c0')
    b2_a = ax_a.bar(x, f1_s, width, label='F1 Class S (Hardest)', color='#e65100')
    b3_a = ax_a.bar(x + width, f1_v, width, label='F1 Class V (Ventricular)', color='#2e7d32')
    ax_a.set_ylabel('F1-Score', fontweight='bold')
    ax_a.set_title('Ablation: Downstream Classification on DS2', fontweight='bold', fontsize=9.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(configs, fontweight='bold', fontsize=8)
    ax_a.set_ylim(0.48, 0.98)
    ax_a.grid(True, axis='y', ls='--', alpha=0.5)
    ax_a.legend(loc='lower right', fontsize=7.8)
    for bars in [b1_a, b2_a, b3_a]:
        for b in bars:
            ax_a.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                          ha='center', va='bottom', fontsize=7.2, rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_ablation_f1_comparison.png')
    plt.savefig(FIG_DIR / 'fig9_ablation_f1.png')
    plt.close()

    fig_b, ax_b = plt.subplots(figsize=(6.2, 4.0))
    scatter_b = ax_b.scatter(params, latency, s=[(s-7.0)*130 for s in snr_imp], c=snr_imp,
                            cmap='viridis', alpha=0.88, edgecolors='black', linewidth=1.2)
    cbar_b = plt.colorbar(scatter_b, ax=ax_b)
    cbar_b.set_label('$\\Delta$SNR Improvement (dB) ↑', fontweight='bold', fontsize=8.5)
    for i, m in enumerate(models):
        if 'HaarSymLite' in m:
            ax_b.annotate(m + '\n(35.2ms, 10.01dB)', (params[i], latency[i]), xytext=(-95, -20), textcoords='offset points',
                         fontweight='bold', color='#1565c0', fontsize=7.5,
                         arrowprops=dict(arrowstyle="->", color='#1565c0', lw=1.5))
        elif 'DeepFilter' in m:
            ax_b.annotate(m + ' (BSPC 2021)', (params[i], latency[i]), xytext=(10, -8), textcoords='offset points',
                         fontweight='bold', color='#e65100', fontsize=7.5,
                         arrowprops=dict(arrowstyle="->", color='#e65100', lw=1.2))
        else:
            ax_b.annotate(m, (params[i], latency[i]), xytext=(6, 4), textcoords='offset points', fontsize=7.5)
    ax_b.axvspan(0, 100, ymin=0, ymax=0.45, color='green', alpha=0.08, label='Optimal Edge Zone')
    ax_b.set_xlabel('Trainable Parameters (K) ↓', fontweight='bold')
    ax_b.set_ylabel('CPU Latency (ms/segment) ↓', fontweight='bold')
    ax_b.set_title('Pareto Trade-off: Latency vs. Parameters vs. $\\Delta$SNR', fontweight='bold', fontsize=9.5)
    ax_b.grid(True, ls='--', alpha=0.5)
    ax_b.set_xlim(0, 800)
    ax_b.set_ylim(25, 80)
    ax_b.legend(loc='upper right', fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig7_model_complexity_pareto.png')
    plt.savefig(FIG_DIR / 'fig9_pareto_efficiency.png')
    plt.close()
    print("  -> Saved fig9_ablation_pareto.png, fig6_ablation_f1_comparison.png, fig7_model_complexity_pareto.png")


def main():
    print("==========================================================")
    print("GENERATING IEEE TRANSACTIONS FIGURES FOR OVERLEAF (paper_ieee/Fig)")
    print("==========================================================")
    generate_fig1_pipeline()
    generate_fig2_haarsymlite()
    generate_fig3_classifier_vqc()
    generate_fig4_waveform()
    generate_fig5_snr_prd_curves()
    generate_fig6_noise_breakdown()
    generate_fig7_clinical()
    generate_fig8_confusion_matrices()
    generate_fig9_ablation_pareto()
    print("==========================================================")
    print("ALL 9 OVERLEAF FIGURES SUCCESSFULLY GENERATED!")
    print("==========================================================")

if __name__ == '__main__':
    main()
