from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.utils.metrics import rmse, mae, prd, cosine_similarity, snr_out, snr_improvement


class Evaluator:
    def __init__(self, model, device, run_dir, model_name, wavelet='haar'):
        self.model = model
        self.device = device
        self.run_dir = Path(run_dir)
        self.model_name = model_name
        self.wavelet = wavelet
        (self.run_dir / 'predictions').mkdir(parents=True, exist_ok=True)

    def evaluate(self, loader):
        self.model.eval()
        rows = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc='Testing')):
                x = batch['x'].to(self.device)
                pred = self.model(x).cpu().numpy()
                metas = batch['meta']

                y = batch['y'].cpu().numpy()
                noisy = batch['x'].cpu().numpy()
                bs = pred.shape[0]
                for i in range(bs):
                    clean1d = y[i, 0]
                    pred1d = pred[i, 0]
                    noisy1d = noisy[i, 0]
                    row = {
                        'model_name': self.model_name,
                        'wavelet': self.wavelet,
                        'record_id': metas['record_id'][i],
                        'segment_id': int(metas['segment_id'][i]),
                        'noise_type': metas['noise_type'][i],
                        'noise_snr': float(metas['noise_snr'][i]),
                        'rmse': rmse(clean1d, pred1d),
                        'mae': mae(clean1d, pred1d),
                        'prd': prd(clean1d, pred1d),
                        'cosine': cosine_similarity(clean1d, pred1d),
                        'snr_out': snr_out(clean1d, pred1d),
                        'snr_imp': snr_improvement(clean1d, noisy1d, pred1d),
                    }
                    np.savez_compressed(self.run_dir / 'predictions' / f"{row['record_id']}_seg{row['segment_id']:05d}.npz", clean=clean1d, noisy=noisy1d, pred=pred1d)
                    rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.run_dir / 'test_segment_metrics.csv', index=False)
        df.groupby('noise_type').mean(numeric_only=True).reset_index().to_csv(self.run_dir / 'test_summary_by_noise.csv', index=False)
        df.groupby('noise_snr').mean(numeric_only=True).reset_index().to_csv(self.run_dir / 'test_summary_by_snr.csv', index=False)
        df.groupby(['noise_type', 'noise_snr']).mean(numeric_only=True).reset_index().to_csv(self.run_dir / 'test_summary_by_noise_snr.csv', index=False)
        return df
