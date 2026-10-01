#!/usr/bin/env bash
set -e
python -m src.eval --config configs/default.yaml --run_glob "runs/*"
python scripts/summarize_runs.py --run_glob "runs/*" --out runs/summary_all_models.csv
python scripts/merge_all_segment_csv.py --run_glob "runs/*" --out runs/all_segment_metrics.csv
python scripts/merge_noise_snr_summary.py --run_glob "runs/*" --out runs/all_models_by_noise_snr.csv
