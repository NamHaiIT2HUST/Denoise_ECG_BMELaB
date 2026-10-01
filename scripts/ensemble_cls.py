"""Ensemble nhieu seed model (trung binh softmax) -> mot ket qua on dinh, phuong sai thap.

Vi du:
  PYTHONPATH=. python scripts/ensemble_cls.py --config configs/default.yaml \
      --denoise_run <run_optimal> --head quantum --encoding amplitude \
      --run_dirs runs_cls/q_seed0 runs_cls/q_seed1 runs_cls/q_seed2
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

from src.utils.io import load_yaml
from src.models.classifier_cnn import DenoiseCNNClassifier
from src.data.dataset_cls import ECGBeatClsDataset
from src.train_classifier import load_denoiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--denoise_run', default=None)
    ap.add_argument('--run_dirs', nargs='+', required=True)
    ap.add_argument('--head', default='quantum', choices=['quantum', 'classical'])
    ap.add_argument('--encoding', default='amplitude', choices=['amplitude', 'angle'])
    ap.add_argument('--pool_k', type=int, default=8)
    ap.add_argument('--no_denoise', action='store_true')
    ap.add_argument('--out', default=None, help='luu ket qua ra file .json')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ccfg = cfg.get('classification', {})
    n_classes = int(ccfg.get('n_classes', 5))
    n_leads = int(ccfg.get('n_leads', 1))
    n_context = int(ccfg.get('n_context', 1))
    z_norm = bool(ccfg.get('z_norm', True))
    manifest = Path(cfg['data']['processed_root']) / 'manifest_cls.csv'
    use_denoise = (not args.no_denoise) and args.denoise_run is not None
    den = load_denoiser(args.denoise_run) if use_denoise else None

    ds = ECGBeatClsDataset(manifest, split='test', use_noisy=True, z_norm=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    models = []
    for rd in args.run_dirs:
        m = DenoiseCNNClassifier(denoiser=den, n_classes=n_classes, head=args.head,
                                 encoding=args.encoding, pool_k=args.pool_k,
                                 use_denoise=use_denoise, n_leads=n_leads,
                                 n_context=n_context, z_norm=z_norm)
        ck = torch.load(Path(rd) / 'model_best.pt', map_location='cpu')
        m.load_state_dict(ck['model_state_dict'])
        m.eval(); models.append(m)

    ys, ps, snrs = [], [], []
    with torch.no_grad():
        for b in dl:
            probs = sum(F.softmax(m(b['x'], b['rr']), dim=1) for m in models) / len(models)
            ps += probs.argmax(1).numpy().tolist()
            ys += b['label'].numpy().tolist()
            snrs += [float(s) for s in b['meta']['noise_snr']]
    ys, ps = np.array(ys), np.array(ps)
    labels = list(range(n_classes))
    f1 = f1_score(ys, ps, average=None, labels=labels, zero_division=0).round(4)
    res = {
        'n_models': len(models), 'head': args.head,
        'denoise_active': bool(use_denoise),
        'run_dirs': list(args.run_dirs),
        'accuracy': round(accuracy_score(ys, ps), 4),
        'macro_f1': round(f1_score(ys, ps, average='macro', labels=labels, zero_division=0), 4),
        'macro_f1_nsv': round(float(np.mean(f1[:3])), 4),
        'per_class_f1': f1.tolist(),
        'per_class_Se': recall_score(ys, ps, average=None, labels=labels, zero_division=0).round(4).tolist(),
        'per_class_PP': precision_score(ys, ps, average=None, labels=labels, zero_division=0).round(4).tolist(),
        'confusion': confusion_matrix(ys, ps, labels=labels).tolist(),
    }
    print(f"ENSEMBLE n={res['n_models']} ({res['head']}) | DENOISE_ACTIVE={res['denoise_active']}")
    print('accuracy      =', res['accuracy'])
    print('macro-F1 (all)=', res['macro_f1'])
    print('macro-F1 (NSV)=', res['macro_f1_nsv'])
    print('per-class F1  =', res['per_class_f1'])
    print('per-class Se  =', res['per_class_Se'])
    print('per-class +P  =', res['per_class_PP'])
    print('confusion     =', res['confusion'])
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print('Saved ->', out)


if __name__ == '__main__':
    main()
