#!/usr/bin/env bash
# Train moi baseline voi CA HAI loss (mse va mixed alpha=0.8), cung wavelet db4,
# giao thuc cong bang mo ta o PROJECT_OVERVIEW.md muc 3.6 -> lay ket qua tot nhat/baseline.
set -e
for MODEL in dw_cnn dw_se dnn_dan fcn liwave deepfilter; do
  python -m src.train --config configs/default.yaml --model $MODEL --wavelet db4 --loss mse
  python -m src.train --config configs/default.yaml --model $MODEL --wavelet db4 --loss mixed --alpha 0.8
done
