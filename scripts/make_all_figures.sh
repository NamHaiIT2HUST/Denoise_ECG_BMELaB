#!/usr/bin/env bash
set -e

python -m src.visualize --config configs/default.yaml --run_glob "runs/*" --segment_index 0
python scripts/plot_results.py --summary runs/summary_all_models.csv --noise_snr runs/all_models_by_noise_snr.csv --out_dir figures/metrics
