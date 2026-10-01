"""Train phan loai AAMI (mac dinh 3 lop N/S/V) tren tin hieu DA KHU NHIEU.

Denoiser (freeze, tien xu ly) -> CNN encoder -> SE -> pool -> noi RR
-> head (classical MLP / quantum VQC + bypass) -> n_classes.
Mat can bang xu ly bang WeightedRandomSampler voi `--sampler_power` (lever chinh).

Mac dinh = cau hinh tot nhat (macro-F1 0.8195): sampler_power 0.9, dropout 0.3,
khong augment, khong prior-tune, denoiser freeze.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils.io import load_yaml, load_json, save_json
from src.utils.seed import seed_everything
from src.models import build_model
from src.models.classifier_cnn import DenoiseCNNClassifier
from src.data.dataset_cls import ECGBeatClsDataset


def load_denoiser(run_dir):
    run = Path(run_dir)
    cfg = load_json(run / 'config_effective.json')
    m, d = cfg.get('model', {}), cfg.get('data', {})
    base = int(m.get('base', 32))
    model = build_model(m.get('name', 'haar_sym_lite'), wavelet=d.get('wavelet', 'haar'),
                        base=base, expansion=int(m.get('expansion', 4)),
                        mid_depth=int(m.get('mid_depth', 3)), se_reduction=int(m.get('se_reduction', 8)))
    ck = torch.load(run / 'checkpoint_best.pt', map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    return model


def collect_logits(model, loader, device):
    """Tra ve (logits, labels, snrs) de tune hieu chinh prior."""
    model.eval()
    L, Y, S = [], [], []
    with torch.no_grad():
        for b in loader:
            L.append(model(b['x'].to(device), b['rr'].to(device)).cpu())
            Y.append(b['label'])
            S += [float(s) for s in b['meta']['noise_snr']]
    return torch.cat(L), torch.cat(Y), np.array(S)


def tune_prior_bias(logits, labels, n_classes, grid=np.arange(0.0, 1.001, 0.01)):
    """Tim he so tau cho logit-adjustment: logit_c += tau * log(1/prior_c).

    Chon tau toi da hoa macro-F1 tren VAL (khong dung test) -> can bang Se/+P.
    """
    from sklearn.metrics import f1_score
    counts = np.bincount(labels.numpy(), minlength=n_classes).astype(np.float64)
    prior = counts / max(counts.sum(), 1)
    adj = np.log(1.0 / np.maximum(prior, 1e-12))
    best_tau, best_f1 = 0.0, -1.0
    for tau in grid:
        pred = (logits + torch.tensor(tau * adj, dtype=logits.dtype)).argmax(1).numpy()
        f1 = f1_score(labels.numpy(), pred, average='macro',
                      labels=list(range(n_classes)), zero_division=0)
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), float(f1)
    return best_tau, best_f1, adj


def run_eval(model, loader, device, n_classes, bias=None):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, confusion_matrix)
    logits, labels, snrs = collect_logits(model, loader, device)
    if bias is not None:
        logits = logits + torch.tensor(bias, dtype=logits.dtype)
    ys, ps = labels.numpy(), logits.argmax(1).numpy()
    labels = list(range(n_classes))
    out = {
        'accuracy': float(accuracy_score(ys, ps)),
        'macro_f1': float(f1_score(ys, ps, average='macro', labels=labels, zero_division=0)),
        'macro_precision': float(precision_score(ys, ps, average='macro', labels=labels, zero_division=0)),
        'macro_recall': float(recall_score(ys, ps, average='macro', labels=labels, zero_division=0)),
        'weighted_f1': float(f1_score(ys, ps, average='weighted', labels=labels, zero_division=0)),
        'per_class_f1': f1_score(ys, ps, average=None, labels=labels, zero_division=0).round(4).tolist(),
        'per_class_Se': recall_score(ys, ps, average=None, labels=labels, zero_division=0).round(4).tolist(),
        'per_class_PP': precision_score(ys, ps, average=None, labels=labels, zero_division=0).round(4).tolist(),
        'confusion': confusion_matrix(ys, ps, labels=labels).tolist(),
    }
    per_snr = {}
    for s in sorted(set(snrs.tolist())):
        mk = snrs == s
        per_snr[str(s)] = float(f1_score(ys[mk], ps[mk], average='macro', labels=labels, zero_division=0))
    out['macro_f1_by_snr'] = per_snr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--denoise_run', default=None, help='run denoiser (bo trong neu --no_denoise)')
    ap.add_argument('--out_dir', default='runs_cls/run')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--samples_per_epoch', type=int, default=16000, help='so mau/epoch qua WeightedRandomSampler')
    # sampler_power: ti le lay mau S:N = (n_N/n_S)^p. 0.9 la diem toi uu (xem bang ablation trong CLASSIFICATION.md)
    ap.add_argument('--sampler_power', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=None, help='ghi de seed cfg (cho multi-seed)')
    ap.add_argument('--head', default='quantum', choices=['quantum', 'classical'])
    ap.add_argument('--encoding', default='amplitude', choices=['amplitude', 'angle'])
    ap.add_argument('--pool_k', type=int, default=8)
    ap.add_argument('--no_denoise', action='store_true', help='ablation: phan loai tren beat NHIEU')
    ap.add_argument('--vqc_lr_scale', type=float, default=0.2, help='LR cho VQC = lr*scale (nho de on dinh)')
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.3)
    # Hai tuy chon duoi DA THU va lam KEM di -> mac dinh TAT (giu de tai lap ablation)
    ap.add_argument('--augment', action='store_true', help='bat data augmentation (thu nghiem: lam kem di)')
    ap.add_argument('--prior_tune', action='store_true', help='hieu chinh prior tren val (thu nghiem: trung tinh/kem)')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(args.seed if args.seed is not None else cfg['seed'])
    device = torch.device(args.device)
    ccfg = cfg.get('classification', {})
    n_classes = int(ccfg.get('n_classes', 5))
    n_leads = int(ccfg.get('n_leads', 1))
    n_context = int(ccfg.get('n_context', 1))
    z_norm = bool(ccfg.get('z_norm', True))
    manifest = Path(cfg['data']['processed_root']) / 'manifest_cls.csv'
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    use_denoise = not args.no_denoise
    # An toan: neu muon khu nhieu ma KHONG truyen --denoise_run -> bao loi ngay,
    # tranh truong hop model am tham chay khong khu nhieu ma log van bao co.
    if use_denoise and not args.denoise_run:
        raise SystemExit('LOI: thieu --denoise_run. Dung --no_denoise neu that su '
                         'muon phan loai tren beat NHIEU.')
    denoiser = load_denoiser(args.denoise_run) if use_denoise else None
    model = DenoiseCNNClassifier(
        denoiser=denoiser, n_classes=n_classes, head=args.head, encoding=args.encoding,
        pool_k=args.pool_k, use_denoise=use_denoise, freeze_denoiser=True,
        n_leads=n_leads, z_norm=z_norm, dropout=args.dropout, n_context=n_context).to(device)
    # In TRANG THAI THAT cua model (model.use_denoise), khong phai co dong lenh
    print(f'n_classes={n_classes} | n_context={n_context} | head={args.head} | '
          f'encoding={args.encoding} | DENOISE_ACTIVE={model.use_denoise} | '
          f'sampler_power={args.sampler_power} | dropout={args.dropout} | augment={args.augment}')
    if model.use_denoise:
        print(f'  denoise_run = {args.denoise_run}')

    ds_tr = ECGBeatClsDataset(manifest, split='train', use_noisy=True, z_norm=False,
                              augment=args.augment)
    ds_val = ECGBeatClsDataset(manifest, split='val', use_noisy=True, z_norm=False)
    ds_te = ECGBeatClsDataset(manifest, split='test', use_noisy=True, z_norm=False)

    # Can bang bang WeightedRandomSampler (oversample lop hiem, khong vut du lieu)
    labels = ds_tr.df['label'].to_numpy()
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    class_w = 1.0 / np.power(np.maximum(counts, 1), args.sampler_power)
    sample_w = class_w[labels]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=min(args.samples_per_epoch, len(ds_tr)), replacement=True)
    print('So beat/lop (train):', counts.astype(int).tolist())

    # num_workers=0: tranh loi Windows multiprocessing 'spawn' spawn nham sai python.exe
    # (tung gay NotImplementedError khi unpickle StringDtype do lech phien ban pandas
    # giua tien trinh chinh va worker con) - xem ROADMAP.md.
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer: VQC dung LR thap hon cho on dinh (denoiser luon freeze)
    vqc_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad or n.startswith('denoiser.'):
            continue
        (vqc_params if '.vqc.' in n else other_params).append(p)
    groups = [{'params': other_params, 'lr': args.lr}]
    if vqc_params:
        groups.append({'params': vqc_params, 'lr': args.lr * args.vqc_lr_scale})
    opt = torch.optim.Adam(groups, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    best = {'macro_f1': -1, 'state': None, 'epoch': -1}
    wait = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        if denoiser is not None:
            model.denoiser.eval()                       # denoiser luon o eval (freeze)
        tot = 0.0
        for b in dl_tr:
            opt.zero_grad()
            loss = crit(model(b['x'].to(device), b['rr'].to(device)), b['label'].to(device))
            loss.backward(); opt.step()
            tot += loss.item()
        val = run_eval(model, dl_val, device, n_classes)
        print(f"Epoch {ep}: loss={tot/max(1,len(dl_tr)):.4f} | val_macroF1={val['macro_f1']:.4f} acc={val['accuracy']:.4f}")
        if val['macro_f1'] > best['macro_f1']:
            best = {'macro_f1': val['macro_f1'],
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'epoch': ep}
            torch.save({'model_state_dict': best['state'], 'epoch': ep}, out_dir / 'model_best.pt')
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print('Early stopping.'); break

    if best['state'] is not None:
        model.load_state_dict(best['state'])

    # --- Hieu chinh prior: tune tau tren VAL (khong dung test) ---
    bias = None
    if args.prior_tune:
        vl, vy, _ = collect_logits(model, dl_val, device)
        tau, vf1, adj = tune_prior_bias(vl, vy, n_classes)
        bias = tau * adj
        base_val = run_eval(model, dl_val, device, n_classes)['macro_f1']
        print(f'\nPrior-tune tren VAL: tau={tau:.2f} | val macroF1 {base_val:.4f} -> {vf1:.4f}')
        print('  bias =', np.round(bias, 3).tolist())

    test = run_eval(model, dl_te, device, n_classes, bias=bias)
    # Luu DAY DU cau hinh hieu luc -> truy nguoc duoc moi run (tranh nham lan ve sau)
    save_json({'best_epoch': best['epoch'], 'val_macro_f1': best['macro_f1'],
               'n_classes': n_classes,
               'prior_bias': (None if bias is None else bias.tolist()),
               'config': {
                   'denoise_active': bool(model.use_denoise),
                   'denoise_run': args.denoise_run,
                   'head': args.head, 'encoding': args.encoding,
                   'sampler_power': args.sampler_power,
                   'samples_per_epoch': args.samples_per_epoch,
                   'dropout': args.dropout, 'weight_decay': args.weight_decay,
                   'lr': args.lr, 'batch_size': args.batch_size,
                   'augment': bool(args.augment), 'prior_tune': bool(args.prior_tune),
                   'seed': (args.seed if args.seed is not None else cfg['seed']),
                   'n_context': n_context, 'n_leads': n_leads, 'z_norm': z_norm,
                   'train_counts': counts.astype(int).tolist(),
               },
               'test': test}, out_dir / 'metrics.json')
    from src.preprocess.beats import class_names
    names = class_names(n_classes)
    nsv = float(np.mean(test['per_class_f1'][:3]))
    print('\n===== TEST =====')
    print('accuracy      :', round(test['accuracy'], 4))
    print('weighted-F1   :', round(test['weighted_f1'], 4))
    print('macro-F1      :', round(test['macro_f1'], 4))
    print('macro-F1 (NSV):', round(nsv, 4))
    print('macro-prec/rec:', round(test['macro_precision'], 4), '/', round(test['macro_recall'], 4))
    print('classes       :', names)
    print('per-class F1  :', test['per_class_f1'])
    print('per-class Se  :', test['per_class_Se'])
    print('per-class +P  :', test['per_class_PP'])
    print('confusion     :', test['confusion'])
    print('Saved ->', out_dir / 'metrics.json')


if __name__ == '__main__':
    main()
