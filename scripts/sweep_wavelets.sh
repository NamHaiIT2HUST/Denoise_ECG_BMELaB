#!/usr/bin/env bash
set -e

for W in haar db2 db4 sym4 coif1 bior2.2; do
  python -m src.train \
    --config configs/default.yaml \
    --model haar_sym_lite \
    --wavelet "$W" \
    --loss mse \
    --base 32 \
    --expansion 4 \
    --mid_depth 3
done
