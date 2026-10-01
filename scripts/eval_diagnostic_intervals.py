import argparse
import glob
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import neurokit2 as nk

from src.utils.io import load_yaml, load_json
from src.models import build_model
from src.data.dataset_1d import ECG1DDataset

def get_intervals(signal, fs=360):
    try:
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
        # Delineate waves
        _, waves_peak = nk.ecg_delineate(signal, rpeaks['ECG_R_Peaks'], sampling_rate=fs, method="dwt", show=False)
        
        # Calculate intervals
        qt_intervals = []
        qrs_durations = []
        
        q_peaks = waves_peak.get('ECG_Q_Peaks', [])
        s_peaks = waves_peak.get('ECG_S_Peaks', [])
        t_offsets = waves_peak.get('ECG_T_Offsets', [])
        
        for q, s, t_off in zip(q_peaks, s_peaks, t_offsets):
            if not np.isnan(q) and not np.isnan(s):
                qrs_durations.append((s - q) / fs * 1000) # in ms
            if not np.isnan(q) and not np.isnan(t_off):
                qt_intervals.append((t_off - q) / fs * 1000) # in ms
                
        return np.nanmean(qt_intervals) if qt_intervals else np.nan, np.nanmean(qrs_durations) if qrs_durations else np.nan
    except Exception as e:
        return np.nan, np.nan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest_csv = Path(cfg['data']['processed_root']) / 'manifest.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fs = cfg['data']['fs']

    run_dir = Path(args.run_dir)
    run_cfg = load_json(run_dir / 'config_effective.json')
    model_name = run_cfg.get('model', {}).get('name', 'haar_sym_lite')
    
    ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location='cpu')
    model = build_model(
        model_name,
        wavelet=run_cfg['data'].get('wavelet', 'haar'),
        base=run_cfg['model'].get('base', 32),
        expansion=run_cfg['model'].get('expansion', 4),
        mid_depth=run_cfg['model'].get('mid_depth', 3),
        se_reduction=run_cfg['model'].get('se_reduction', 8),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ds_test = ECG1DDataset(manifest_csv, split='test')
    
    results = []
    
    print(f"Evaluating {run_dir} for diagnostic intervals...")
    with torch.no_grad():
        for i in tqdm(range(min(500, len(ds_test)))): # Evaluate first 500 segments to save time
            sample = ds_test[i]
            noisy = sample['x'].to(device).unsqueeze(0)
            clean = sample['y'].numpy().flatten()
            meta = sample['meta']
            
            y_hat = model(noisy).cpu().numpy().flatten()
            noisy_np = noisy.cpu().numpy().flatten()
            
            ref_qt, ref_qrs = get_intervals(clean, fs)
            noisy_qt, noisy_qrs = get_intervals(noisy_np, fs)
            denoised_qt, denoised_qrs = get_intervals(y_hat, fs)
            
            results.append({
                'noise_type': meta['noise_type'],
                'noise_snr': meta['noise_snr'],
                'ref_qt': ref_qt,
                'ref_qrs': ref_qrs,
                'noisy_qt_err': np.abs(noisy_qt - ref_qt) if not np.isnan(ref_qt) else np.nan,
                'noisy_qrs_err': np.abs(noisy_qrs - ref_qrs) if not np.isnan(ref_qrs) else np.nan,
                'denoised_qt_err': np.abs(denoised_qt - ref_qt) if not np.isnan(ref_qt) else np.nan,
                'denoised_qrs_err': np.abs(denoised_qrs - ref_qrs) if not np.isnan(ref_qrs) else np.nan,
            })
            
    df = pd.DataFrame(results).dropna()
    summary = df.groupby(['noise_type', 'noise_snr']).mean().reset_index()
    print("\n--- Diagnostic Intervals Evaluation ---")
    print(summary[['noise_type', 'noise_snr', 'noisy_qt_err', 'denoised_qt_err', 'noisy_qrs_err', 'denoised_qrs_err']])
    
    out_csv = run_dir / 'diagnostic_intervals_summary.csv'
    summary.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

if __name__ == '__main__':
    main()
