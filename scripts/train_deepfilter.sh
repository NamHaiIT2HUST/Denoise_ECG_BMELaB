#!/usr/bin/env bash
set -e
echo "Starting DeepFilter baseline training..."
python -m src.train --config configs/default.yaml --model deepfilter --wavelet db4 --loss mse
python -m src.train --config configs/default.yaml --model deepfilter --wavelet db4 --loss mixed --alpha 0.8
echo "DeepFilter training complete!"
