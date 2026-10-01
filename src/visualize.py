import argparse
from pathlib import Path
import glob
import numpy as np

from src.utils.io import load_yaml
from src.utils.plots import save_waveform_plot, save_error_plot, save_all_models_comparison


def plot_single_run_examples(run_dir: Path, segment_index=0):
    pred_dir = run_dir / 'predictions'
    files = sorted(pred_dir.glob('*.npz'))
    if not files:
        return
    idx = min(max(segment_index, 0), len(files) - 1)
    npz = np.load(files[idx], allow_pickle=True)
    save_waveform_plot(npz['clean'], npz['noisy'], npz['pred'], run_dir / 'figures' / 'waveforms' / 'example.png', title=run_dir.name)
    save_error_plot(npz['clean'], npz['noisy'], npz['pred'], run_dir / 'figures' / 'waveforms' / 'error.png', title=run_dir.name)


def plot_all_model_comparison(run_dirs, out_root, segment_index=0):
    picks = {}
    clean = noisy = None
    for run_dir in run_dirs:
        pred_files = sorted((run_dir / 'predictions').glob('*.npz'))
        if not pred_files:
            continue
        idx = min(max(segment_index, 0), len(pred_files) - 1)
        npz = np.load(pred_files[idx], allow_pickle=True)
        if clean is None:
            clean = npz['clean']
            noisy = npz['noisy']
        picks[run_dir.name] = npz['pred']
    if picks:
        save_all_models_comparison(clean, noisy, picks, out_root / 'comparison_all_models' / 'same_segment.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run_glob', required=True)
    parser.add_argument('--segment_index', type=int, default=0)
    args = parser.parse_args()
    load_yaml(args.config)
    run_dirs = [Path(p) for p in sorted(glob.glob(args.run_glob))]
    out_root = Path('figures')
    for run_dir in run_dirs:
        plot_single_run_examples(run_dir, segment_index=args.segment_index)
    plot_all_model_comparison(run_dirs, out_root, segment_index=args.segment_index)
    print('Saved visualizations.')


if __name__ == '__main__':
    main()
