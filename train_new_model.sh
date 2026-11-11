#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  BAT DAU HUAN LUYEN HAARSYMLITE V2 (Unified ResSE Architecture)"
echo "============================================================"
echo "GPU: NVIDIA Quadro RTX 3000"
echo "Config: base=32, mid_depth=1 (75,133 parameters)"
echo "Features: Conv 7x1 + 3x1, ResSE Filtered Detail Skips, SiLU"
echo "Loss: Mixed Loss (0.8 MSE + 0.2 L1)"
echo ""

python -m src.train \
    --config configs/default.yaml \
    --model haar_sym_lite \
    --base 32 \
    --mid_depth 1 \
    --loss mixed \
    --alpha 0.8 \
    --run_suffix resse_v2

echo ""
echo "============================================================"
echo "  HOAN TAT HUAN LUYEN! CHECKPOINT DUOC LUU TAI:"
echo "  denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-1_loss-mixed_alpha-0.8_resse_v2"
echo "============================================================"
