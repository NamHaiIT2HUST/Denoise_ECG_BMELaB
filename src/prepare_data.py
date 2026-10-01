import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from src.utils.io import load_yaml, save_json
from src.preprocess.bandpass import bandpass_filter
from src.preprocess.segments import sliding_segments, keep_energy_middle_percentile
from src.preprocess.noise_mixer import make_noisy_segment
from src.preprocess.split_records import list_mitdb_records, split_records


def read_record_first_channel(record_path, channel_idx=0):
    record = wfdb.rdrecord(str(record_path))
    sig = record.p_signal[:, channel_idx]
    return sig.astype(np.float64), int(record.fs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg['data']

    dataset_root = Path(data_cfg['dataset_root'])
    mitdb_dir = dataset_root / data_cfg['mitdb_dir']
    nstdb_dir = dataset_root / data_cfg['nstdb_dir']
    out_root = Path(data_cfg['processed_root'])
    seg_dir = out_root / 'segments'
    seg_dir.mkdir(parents=True, exist_ok=True)

    records = list_mitdb_records(mitdb_dir)
    splits = split_records(records)
    save_json(splits, out_root / 'splits.json')

    noise_bank = {}
    for name in ['bw', 'ma', 'em']:
        sig, fs = read_record_first_channel(nstdb_dir / name, channel_idx=0)
        noise_bank[name] = sig

    rows = []
    seg_id_global = 0
    train_snrs = data_cfg['noise_train_snrs']
    test_snrs = data_cfg['noise_test_snrs']
    noise_groups = data_cfg['noise_groups']

    for split_name, recs in splits.items():
        for rec_id in tqdm(recs, desc=f'Preparing {split_name}'):
            sig, fs = read_record_first_channel(mitdb_dir / rec_id, channel_idx=data_cfg['channel_idx'])
            sig = bandpass_filter(sig, fs=fs, low=data_cfg['bandpass_low'], high=data_cfg['bandpass_high'])
            segments = sliding_segments(sig, segment_length=data_cfg['segment_length'], overlap_ratio=data_cfg['overlap_ratio'])
            segments, energy_meta = keep_energy_middle_percentile(segments, low_pct=data_cfg['energy_percentile_low'], high_pct=data_cfg['energy_percentile_high'])
            snr_list = train_snrs if split_name in ['train', 'val'] else test_snrs
            for local_seg_id, (start, seg) in enumerate(segments):
                snr = float(snr_list[local_seg_id % len(snr_list)])
                group = noise_groups[local_seg_id % len(noise_groups)]
                noisy, added_noise, noise_type = make_noisy_segment(seg, noise_bank, target_snr_db=snr, group_name=group, seed=seg_id_global + 42)
                npz_path = seg_dir / f'{rec_id}_seg{local_seg_id:05d}.npz'
                np.savez_compressed(npz_path, clean=seg.astype(np.float32), noisy=noisy.astype(np.float32), added_noise=added_noise.astype(np.float32))
                rows.append({
                    'record_id': rec_id,
                    'segment_id': local_seg_id,
                    'global_segment_id': seg_id_global,
                    'start': start,
                    'split': split_name,
                    'noise_type': noise_type,
                    'noise_snr': snr,
                    'npz_path': str(npz_path),
                })
                seg_id_global += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'manifest.csv', index=False)
    print(f'Saved manifest to {out_root / "manifest.csv"}')


if __name__ == '__main__':
    main()
