import argparse
from pathlib import Path
import glob
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_yaml, load_json
from src.models import build_model
from src.data.dataset_1d import ECG1DDataset
from src.engine.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run_glob', required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    manifest_csv = Path(data_cfg['processed_root']) / 'manifest.csv'
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing {manifest_csv}. Reuse the existing preprocessed data folder before evaluation.")
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() and train_cfg['device'] == 'cuda' else 'cpu')

    run_dirs = sorted(glob.glob(args.run_glob))
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        name = run_dir.name
        cfg_path = run_dir / 'config_effective.json'
        run_cfg = load_json(cfg_path) if cfg_path.exists() else {}
        parts = {kv.split('-')[0]: '-'.join(kv.split('-')[1:]) for kv in name.split('_') if '-' in kv}
        model_name = run_cfg.get('model', {}).get('name', name.split('_wav')[0])
        wavelet = run_cfg.get('data', {}).get('wavelet', parts.get('wav', 'haar'))
        base = int(run_cfg.get('model', {}).get('base', parts.get('base', 32)))
        expansion = int(run_cfg.get('model', {}).get('expansion', parts.get('exp', 4)))
        mid_depth = int(run_cfg.get('model', {}).get('mid_depth', parts.get('mid', 3)))
        se_reduction = int(run_cfg.get('model', {}).get('se_reduction', cfg.get('model', {}).get('se_reduction', 8)))
        ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location='cpu')
        model = build_model(
            model_name,
            wavelet=wavelet,
            base=base,
            expansion=expansion,
            mid_depth=mid_depth,
            se_reduction=se_reduction,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        ds_test = ECG1DDataset(manifest_csv, split='test')
        dl_test = DataLoader(ds_test, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])
        evaluator = Evaluator(model, device, run_dir, model_name, wavelet=wavelet)
        evaluator.evaluate(dl_test)
        print(f'Evaluated {run_dir}')


if __name__ == '__main__':
    main()
