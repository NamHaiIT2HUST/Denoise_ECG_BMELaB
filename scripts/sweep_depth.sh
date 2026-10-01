#!/usr/bin/env bash
set -e

for D in 1 2 3 4 5; do
  python -m src.train \
    --config configs/default.yaml \
    --model haar_sym_lite \
    --wavelet haar \
    --loss mse \
    --base 32 \
    --expansion 4 \
    --mid_depth "$D"
done
