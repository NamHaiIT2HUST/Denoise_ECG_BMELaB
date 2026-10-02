#!/usr/bin/env bash
set -e

NEW_DENOISE_RUN="denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-3_loss-mixed_alpha-0.8_cbam_mish"

echo "=== 1. Danh gia mo hinh Denoise (Mixed Loss + CBAM1D) ==="
python -m src.eval --config configs/default.yaml --run_glob "$NEW_DENOISE_RUN"

echo "=== 2. Train 5 Seeds Classifier (Classical) voi Denoise moi ==="
for s in 0 1 2 3 4; do
  python -m src.train_classifier \
      --config configs/default.yaml \
      --denoise_run "$NEW_DENOISE_RUN" \
      --out_dir runs_cls/final_classical_seed$s \
      --head classical \
      --device cuda --seed $s
done

echo "=== 3. Train 5 Seeds Classifier (Quantum) voi Denoise moi ==="
for s in 0 1 2 3 4; do
  python -m src.train_classifier \
      --config configs/default.yaml \
      --denoise_run "$NEW_DENOISE_RUN" \
      --out_dir runs_cls/final_quantum_seed$s \
      --head quantum \
      --device cuda --seed $s
done

echo "=== 4. Ensemble Classical ==="
PYTHONPATH=. python scripts/ensemble_cls.py \
    --config configs/default.yaml \
    --denoise_run "$NEW_DENOISE_RUN" \
    --head classical \
    --run_dirs runs_cls/final_classical_seed0 runs_cls/final_classical_seed1 runs_cls/final_classical_seed2 runs_cls/final_classical_seed3 runs_cls/final_classical_seed4

echo "=== 5. Ensemble Quantum ==="
PYTHONPATH=. python scripts/ensemble_cls.py \
    --config configs/default.yaml \
    --denoise_run "$NEW_DENOISE_RUN" \
    --head quantum \
    --run_dirs runs_cls/final_quantum_seed0 runs_cls/final_quantum_seed1 runs_cls/final_quantum_seed2 runs_cls/final_quantum_seed3 runs_cls/final_quantum_seed4

echo "TAT CA DA HOAN TAT! CHUC MUNG DU AN THANH CONG!"
