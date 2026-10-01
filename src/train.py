import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_yaml, save_json
from src.utils.seed import seed_everything
from src.models import build_model
from src.data.dataset_1d import ECG1DDataset
from src.losses.composite import MixedMSEL1Loss
from src.losses.haar import MSEWithHaarLoss
from src.engine.trainer import Trainer


def build_loss(loss_name, alpha=1.0, haar_lambda=0.0, wavelet='haar'):
    if loss_name == 'mse':
        return torch.nn.MSELoss()
    if loss_name == 'huber':
        return torch.nn.HuberLoss()
    if loss_name == 'mixed':
        return MixedMSEL1Loss(alpha=alpha)
    if loss_name in {'mse_haar', 'mse_wavelet'}:
        return MSEWithHaarLoss(haar_lambda=haar_lambda, wavelet=wavelet)
    raise ValueError(loss_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', default=None)
    parser.add_argument('--loss', default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--wavelet', default=None)
    parser.add_argument('--base', type=int, default=None)
    parser.add_argument('--expansion', type=int, default=None)
    parser.add_argument('--mid_depth', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--run_suffix', default='')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg['seed'])
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    model_cfg = cfg['model']
    loss_cfg = cfg['loss']

    model_name = args.model or model_cfg['name']
    loss_name = args.loss or loss_cfg['name']
    alpha = args.alpha if args.alpha is not None else loss_cfg.get('alpha', 1.0)
    wavelet = args.wavelet or data_cfg.get('wavelet', 'haar')
    base = args.base if args.base is not None else model_cfg.get('base', 32)
    expansion = args.expansion if args.expansion is not None else model_cfg.get('expansion', 4)
    mid_depth = args.mid_depth if args.mid_depth is not None else model_cfg.get('mid_depth', 3)
    epochs = args.epochs if args.epochs is not None else train_cfg['epochs']

    run_name = (
        f"{model_name}_wav-{wavelet}_base-{base}_exp-{expansion}_mid-{mid_depth}"
        f"_loss-{loss_name}_alpha-{alpha}"
    )
    if args.run_suffix:
        run_name = f"{run_name}_{args.run_suffix}"
    run_dir = Path(cfg['output_root']) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_effective = dict(cfg)
    cfg_effective['model'] = dict(model_cfg)
    cfg_effective['model'].update({
        'name': model_name,
        'base': base,
        'expansion': expansion,
        'mid_depth': mid_depth,
        'se_reduction': model_cfg.get('se_reduction', 8),
    })
    cfg_effective['data'] = dict(data_cfg)
    cfg_effective['data']['wavelet'] = wavelet
    cfg_effective['loss'] = dict(loss_cfg)
    cfg_effective['loss'].update({'name': loss_name, 'alpha': alpha})
    cfg_effective['train'] = dict(train_cfg)
    cfg_effective['train']['epochs'] = epochs
    save_json(cfg_effective, run_dir / 'config_effective.json')

    device = torch.device(train_cfg['device'] if torch.cuda.is_available() and train_cfg['device'] == 'cuda' else 'cpu')

    manifest_csv = Path(data_cfg['processed_root']) / 'manifest.csv'
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Missing {manifest_csv}. Reuse the already prepared data/processed folder "
            "from the previous project, or run scripts/prepare_data.sh only if you really need to rebuild it."
        )

    ds_train = ECG1DDataset(manifest_csv, split='train')
    ds_val = ECG1DDataset(manifest_csv, split='val')

    dl_train = DataLoader(ds_train, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    dl_val = DataLoader(ds_val, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

    model = build_model(
        model_name,
        wavelet=wavelet,
        se_reduction=model_cfg.get('se_reduction', 8),
        base=base,
        expansion=expansion,
        mid_depth=mid_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    criterion = build_loss(loss_name, alpha=alpha, haar_lambda=loss_cfg.get('haar_lambda', 0.0), wavelet=wavelet)

    trainer = Trainer(model, optimizer, criterion, device, run_dir)
    trainer.fit(dl_train, dl_val, epochs=epochs, patience=train_cfg['patience'])
    print(f'Saved run to {run_dir}')


if __name__ == '__main__':
    main()
