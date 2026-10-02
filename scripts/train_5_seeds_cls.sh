#!/usr/bin/env bash
set -e

# optimal denoise run
DENOISE_RUN="denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-3_loss-mse_alpha-1.0"

echo "=== Training Classical ResNet 5 seeds ==="
for s in 0 1 2 3 4; do
  python -m src.train_classifier \
      --config configs/default.yaml \
      --denoise_run $DENOISE_RUN \
      --out_dir runs_cls/resnet_classical_seed$s \
      --head classical \
      --device cuda --seed $s
done

echo "=== Training Quantum ResNet 5 seeds ==="
for s in 0 1 2 3 4; do
  python -m src.train_classifier \
      --config configs/default.yaml \
      --denoise_run $DENOISE_RUN \
      --out_dir runs_cls/resnet_quantum_seed$s \
      --head quantum \
      --device cuda --seed $s
done

echo "=== Ensembling Classical ==="
python scripts/ensemble_cls.py \
    --config configs/default.yaml \
    --denoise_run $DENOISE_RUN \
    --head classical \
    --run_dirs runs_cls/resnet_classical_seed0 runs_cls/resnet_classical_seed1 runs_cls/resnet_classical_seed2 runs_cls/resnet_classical_seed3 runs_cls/resnet_classical_seed4

echo "=== Ensembling Quantum ==="
python scripts/ensemble_cls.py \
    --config configs/default.yaml \
    --denoise_run $DENOISE_RUN \
    --head quantum \
    --run_dirs runs_cls/resnet_quantum_seed0 runs_cls/resnet_quantum_seed1 runs_cls/resnet_quantum_seed2 runs_cls/resnet_quantum_seed3 runs_cls/resnet_quantum_seed4

echo "All Done!"
