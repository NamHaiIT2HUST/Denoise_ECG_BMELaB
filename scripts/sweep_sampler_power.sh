#!/usr/bin/env bash
# Quet sampler_power 5 seed x 4 muc (thay bang 1-seed da bi huy - xem ROADMAP.md #1).
# Usage: bash scripts/sweep_sampler_power.sh [duong_dan_denoise_run]
set -e
export PYTHONPATH=.

DENOISE_RUN="${1:-denoise_runs/haar_sym_lite_wav-db4_base-32_exp-4_mid-5_loss-mixed_alpha-0.8_optimal}"
OUT_ROOT="runs_cls"
DEVICE="${DEVICE:-cuda}"

for P in 0.6 0.75 0.9 1.0; do
  PTAG=$(awk "BEGIN{printf \"%d\", $P*100}")
  for SEED in 0 1 2 3 4; do
    OUT="$OUT_ROOT/sp${PTAG}_seed${SEED}"
    if [ -f "$OUT/metrics.json" ]; then
      echo "[bo qua] $OUT da co ket qua"
      continue
    fi
    echo ">>> sampler_power=$P seed=$SEED -> $OUT"
    python -m src.train_classifier \
      --config configs/default.yaml \
      --denoise_run "$DENOISE_RUN" \
      --out_dir "$OUT" \
      --epochs 50 --batch_size 64 \
      --head classical \
      --sampler_power "$P" \
      --seed "$SEED" \
      --dropout 0.3 \
      --device "$DEVICE"
  done
done
echo "=== DONE sweep_sampler_power ==="
