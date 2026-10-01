"""Ve UMAP cua khong gian dac trung hoc duoc (cho hinh trong paper).

Chieu xuong 2D cac vector dac trung truoc head phan loai, to mau theo lop N/S/V.
Ho tro 3 che do so sanh:
  --mode denoise   : dac trung khi CO khu nhieu  vs  KHONG khu nhieu
  --mode head      : dac trung nhanh classical   vs  nhanh quantum (12 observable)
  --mode single    : chi mot cau hinh

Vi du:
  PYTHONPATH=. python scripts/plot_umap.py --config configs/default.yaml \
      --denoise_run <run_optimal> --run_dir runs_cls/q09_seed0 \
      --mode denoise --out figures/umap_denoise.png
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.utils.io import load_yaml
from src.models.classifier_cnn import DenoiseCNNClassifier
from src.data.dataset_cls import ECGBeatClsDataset
from src.train_classifier import load_denoiser

CLASS_NAMES = ['N', 'S', 'V']
COLORS = ['#4C72B0', '#DD8452', '#55A868']


def build_model(cfg, ck_path, denoiser, head, encoding, use_denoise, device):
    c = cfg['classification']
    m = DenoiseCNNClassifier(
        denoiser=denoiser, n_classes=int(c.get('n_classes', 3)), head=head,
        encoding=encoding, pool_k=8, use_denoise=use_denoise,
        n_leads=int(c.get('n_leads', 1)), n_context=int(c.get('n_context', 1)),
        z_norm=bool(c.get('z_norm', True)))
    if ck_path is not None and Path(ck_path).exists():
        ck = torch.load(ck_path, map_location='cpu')
        m.load_state_dict(ck['model_state_dict'], strict=False)
    return m.to(device).eval()


@torch.no_grad()
def extract_features(model, loader, device, max_n=4000):
    """Lay vector dac trung hop nhat u = [morph || rr] (dau vao cua head)."""
    feats, labels = [], []
    for b in loader:
        x, rr = b['x'].to(device), b['rr'].to(device)
        if model.use_denoise:
            x = model._denoise(x)
        if model.z_norm:
            mu = x.mean(dim=-1, keepdim=True)
            sd = x.std(dim=-1, keepdim=True) + 1e-6
            x = (x - mu) / sd
        f = model.pool(model.se(model.cnn(x)))
        morph = model.morph(torch.flatten(f, 1))
        u = torch.cat([morph, model.rr_mlp(rr)], dim=1)
        feats.append(u.cpu().numpy()); labels.append(b['label'].numpy())
        if sum(len(a) for a in labels) >= max_n:
            break
    return np.vstack(feats)[:max_n], np.concatenate(labels)[:max_n]


def umap_2d(X, seed=42):
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        print('[canh bao] chua cai umap-learn -> dung t-SNE thay the (pip install umap-learn)')
        return TSNE(n_components=2, init='pca', random_state=seed).fit_transform(X)


def panel(ax, Z, y, title):
    for c in range(len(CLASS_NAMES)):
        mk = y == c
        ax.scatter(Z[mk, 0], Z[mk, 1], s=4, alpha=.55, c=COLORS[c],
                   label=CLASS_NAMES[c], linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--denoise_run', required=True)
    ap.add_argument('--run_dir', required=True, help='thu muc chua model_best.pt')
    ap.add_argument('--run_dir_b', default=None, help='cau hinh thu 2 de so sanh')
    ap.add_argument('--mode', default='denoise', choices=['denoise', 'head', 'single'])
    ap.add_argument('--head', default='quantum', choices=['quantum', 'classical'])
    ap.add_argument('--encoding', default='amplitude')
    ap.add_argument('--split', default='test')
    ap.add_argument('--max_n', type=int, default=4000)
    ap.add_argument('--out', default='figures/umap.png')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    manifest = Path(cfg['data']['processed_root']) / 'manifest_cls.csv'
    ds = ECGBeatClsDataset(manifest, split=args.split, use_noisy=True, z_norm=False)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    den = load_denoiser(args.denoise_run)

    panels = []
    if args.mode == 'denoise':
        for use_dn, tag in [(True, 'With denoising'), (False, 'Without denoising')]:
            m = build_model(cfg, Path(args.run_dir) / 'model_best.pt', den,
                            args.head, args.encoding, use_dn, device)
            X, y = extract_features(m, dl, device, args.max_n)
            panels.append((umap_2d(X), y, tag))
    elif args.mode == 'head':
        assert args.run_dir_b, '--mode head can them --run_dir_b'
        for rd, hd, tag in [(args.run_dir, 'quantum', 'Quantum head'),
                            (args.run_dir_b, 'classical', 'Classical head')]:
            m = build_model(cfg, Path(rd) / 'model_best.pt', den, hd,
                            args.encoding, True, device)
            X, y = extract_features(m, dl, device, args.max_n)
            panels.append((umap_2d(X), y, tag))
    else:
        m = build_model(cfg, Path(args.run_dir) / 'model_best.pt', den,
                        args.head, args.encoding, True, device)
        X, y = extract_features(m, dl, device, args.max_n)
        panels.append((umap_2d(X), y, 'Learned feature space'))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (Z, y, tag) in zip(axes, panels):
        panel(ax, Z, y, tag)
    axes[-1].legend(markerscale=3, fontsize=9, loc='best', frameon=True)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('Saved', out)


if __name__ == '__main__':
    main()
