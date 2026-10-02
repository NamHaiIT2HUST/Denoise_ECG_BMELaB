#!/usr/bin/env bash
set -e

echo "=== Training Classical ResNet 5 seeds (NO DENOISE) ==="
for s in 0 1 2 3 4; do
  python -m src.train_classifier \
      --config configs/default.yaml \
      --no_denoise \
      --out_dir runs_cls/resnet_classical_no_denoise_seed$s \
      --head classical \
      --device cuda --seed $s
done

echo "=== Training Quantum ResNet 5 seeds (NO DENOISE) ==="
for s in 0 1 2 3 4; do
  python -m src.train_classifier \
      --config configs/default.yaml \
      --no_denoise \
      --out_dir runs_cls/resnet_quantum_no_denoise_seed$s \
      --head quantum \
      --device cuda --seed $s
done
