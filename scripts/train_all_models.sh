#!/usr/bin/env bash
set -e

bash scripts/train_baselines.sh
python -m src.train --config configs/default.yaml --model haar_sym_lite --wavelet haar --loss mse --base 32 --expansion 4 --mid_depth 3
