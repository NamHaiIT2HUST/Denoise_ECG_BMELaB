from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt


def save_waveform_plot(clean, noisy, pred, out_path, title='Waveform comparison'):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(clean, label='Clean', linewidth=1.2)
    plt.plot(noisy, label='Noisy', linewidth=0.9, alpha=0.8)
    plt.plot(pred, label='Pred', linewidth=1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_error_plot(clean, noisy, pred, out_path, title='Error comparison'):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(clean - noisy, label='Noisy error', linewidth=0.9, alpha=0.8)
    plt.plot(clean - pred, label='Prediction error', linewidth=1.0)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_feature_map_grid(feat, out_path, title='Feature maps', max_channels=16):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(feat)
    if arr.ndim == 4:
        arr = arr[0]
    c = min(arr.shape[0], max_channels)
    cols = 4
    rows = math.ceil(c / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        axes[i].axis('off')
    for i in range(c):
        axes[i].imshow(arr[i], aspect='auto', origin='lower', cmap='magma')
        axes[i].set_title(f'ch {i}')
        axes[i].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_all_models_comparison(clean, noisy, model_preds, out_path, title='All models on same segment'):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(model_preds) + 2
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    axes[0].plot(clean, linewidth=1.0)
    axes[0].set_title('Clean')
    axes[1].plot(noisy, linewidth=0.9, color='tab:red')
    axes[1].set_title('Noisy')
    for idx, (name, pred) in enumerate(model_preds.items(), start=2):
        axes[idx].plot(pred, linewidth=0.9)
        axes[idx].set_title(name)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_metric_bar(df, metric, out_path, higher_is_better=False, title=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = df.sort_values(metric, ascending=not higher_is_better)
    labels = data['run_name'] if 'run_name' in data.columns else data['model_name']
    plt.figure(figsize=(max(8, 0.35 * len(data)), 4))
    plt.bar(range(len(data)), data[metric].values, color='white', edgecolor='black', linewidth=1.2)
    plt.xticks(range(len(data)), labels, rotation=75, ha='right', fontsize=8)
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_pivot_heatmap(df, row, col, value, out_path, title=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot_table(index=row, columns=col, values=value, aggfunc='mean')
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(pivot.columns)), max(4, 0.35 * len(pivot.index))))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_title(title or f'{value} by {row} and {col}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
