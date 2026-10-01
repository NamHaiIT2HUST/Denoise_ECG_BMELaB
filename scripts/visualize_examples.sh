#!/usr/bin/env bash
set -e
python -m src.visualize --config configs/default.yaml --run_glob "runs/*" --segment_index 0
