#!/usr/bin/env bash
set -e

DENOISE_RUN="denoise_runs/haar_sym_lite_wav-haar_base-32_exp-4_mid-1_loss-mixed_alpha-0.8_resse_v2"

echo "============================================================"
echo "  1. DANH GIA MO HINH DENOISE MOI (Delta SNR, PRD, RMSE)"
echo "============================================================"
python -m src.eval --config configs/default.yaml --run_glob "$DENOISE_RUN"

echo ""
echo "============================================================"
echo "  2. TRAIN 5 SEEDS CLASSIFIER (ResRR-Net voi SiLU)"
echo "============================================================"
for s in 0 1 2 3 4; do
    echo "--- Dang train Seed $s ---"
    python -m src.train_classifier \
        --config configs/default.yaml \
        --denoise_run "$DENOISE_RUN" \
        --out_dir "runs_cls/resse_classical_seed$s" \
        --head classical \
        --device cuda --seed "$s"
done

echo ""
echo "============================================================"
echo "  3. ENSEMBLE KET QUA PHAN LOAI (DS2 TEST SET - 49,298 BEATS)"
echo "============================================================"
PYTHONPATH=. python scripts/ensemble_cls.py \
    --config configs/default.yaml \
    --denoise_run "$DENOISE_RUN" \
    --head classical \
    --run_dirs runs_cls/resse_classical_seed0 runs_cls/resse_classical_seed1 runs_cls/resse_classical_seed2 runs_cls/resse_classical_seed3 runs_cls/resse_classical_seed4

echo ""
echo "============================================================"
echo "  HOAN TAT TOAN BO PIPELINE!"
echo "============================================================"
