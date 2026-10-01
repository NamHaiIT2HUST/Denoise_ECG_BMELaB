#!/usr/bin/env bash
set -e

bash scripts/train_all_models.sh
bash scripts/sweep_alpha.sh
bash scripts/sweep_depth.sh
bash scripts/sweep_wavelets.sh
bash scripts/eval_all.sh
bash scripts/make_all_figures.sh
