"""Sinh dataset phan loai beat (Normal/Abnormal) tu MIT-BIH Arrhythmia.

Moi beat -> luu (clean, noisy, label) + ghi manifest_cls.csv.
Chia inter-patient DS1/DS2 (de Chazal); val tach tu DS1.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from src.utils.io import load_yaml
from src.preprocess.bandpass import bandpass_filter
from src.preprocess.beats import (
    extract_beats, read_beat_annotations, split_of, aami_label, DS1, DS2, PACED_RECORDS,
)
from src.preprocess.noise_mixer import make_noisy_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    dcfg = cfg['data']
    ccfg = cfg.get('classification', {})

    beat_len = int(ccfg.get('beat_len', 256))
    n_classes = int(ccfg.get('n_classes', 5))
    n_leads = int(ccfg.get('n_leads', 1))
    n_context = int(ccfg.get('n_context', 1))
    val_records = ccfg.get('val_records', ['201', '203', '205', '207', '208'])
    train_snrs = dcfg['noise_train_snrs']
    test_snrs = dcfg['noise_test_snrs']
    groups = dcfg['noise_groups']

    dataset_root = Path(dcfg['dataset_root'])
    mitdb = dataset_root / dcfg['mitdb_dir']
    nstdb = dataset_root / dcfg['nstdb_dir']
    out_root = Path(dcfg['processed_root'])
    beat_dir = out_root / 'beats'
    beat_dir.mkdir(parents=True, exist_ok=True)

    # nap nhieu thuc NSTDB
    noise_bank = {}
    for n in ['bw', 'ma', 'em']:
        rec = wfdb.rdrecord(str(nstdb / n))
        noise_bank[n] = rec.p_signal[:, 0].astype(np.float64)

    # --- Chon val: mac dinh CO DINH (tai lap duoc). Bat auto_val de tu chon. ---
    if not ccfg.get('auto_val', False):
        print('VAL CO DINH (auto_val=false) =', val_records)
    if ccfg.get('auto_val', False):
        cnt = {}
        for rid in DS1:
            if rid in PACED_RECORDS:
                continue
            try:
                sm, sy = read_beat_annotations(mitdb / rid)
            except Exception:
                continue
            c = np.zeros(n_classes, dtype=np.int64)
            for s in sy:
                l = aami_label(s, n_classes)
                if l is not None:
                    c[l] += 1
            cnt[rid] = c
        total = np.sum(list(cnt.values()), axis=0).astype(np.float64)
        # Muc tieu val ~12% moi lop, TRAN 20% -> giu toi da lop hiem cho TRAIN
        target = np.maximum(0.12 * total, np.minimum(60.0, total))
        cap = 0.20 * total
        chosen, acc = [], np.zeros(n_classes, dtype=np.float64)
        remaining = sorted(cnt)          # LIST da sap xep -> tat dinh (set() cho thu tu ngau nhien)
        # Greedy: moi buoc chon record LAP DUOC NHIEU THIEU HUT NHAT (moi lop deu co mat)
        while remaining and np.any(acc < target):
            best, best_score = None, -1.0
            for rid in remaining:
                c = cnt[rid].astype(np.float64)
                if np.any(acc + c > cap):           # bo qua record lam vuot tran
                    continue
                deficit = np.maximum(target - acc, 0)
                # diem = phan thieu hut duoc lap, chuan hoa theo do hiem cua lop
                score = float(np.sum(np.minimum(c, deficit) / np.maximum(total, 1)))
                if score > best_score:
                    best, best_score = rid, score
            if best is None or best_score <= 0:
                break
            chosen.append(best); acc += cnt[best].astype(np.float64); remaining.remove(best)
        if chosen and np.all(acc > 0):              # chi dung neu MOI lop deu co mat trong val
            val_records = chosen
        else:
            print('CANH BAO: auto-val khong du moi lop -> dung val_records trong config')
        print('AUTO val_records =', val_records)
        print('  val/total moi lop =', np.round(acc / np.maximum(total, 1), 3).tolist())

    rows = []
    gid = 0
    for rec_id in tqdm(DS1 + DS2, desc='Beats'):
        if rec_id in PACED_RECORDS:
            continue
        rec = wfdb.rdrecord(str(mitdb / rec_id))
        n_ch = min(n_leads, rec.p_signal.shape[1])
        sigs = []
        for c in range(n_ch):
            s = rec.p_signal[:, c].astype(np.float64)
            sigs.append(bandpass_filter(s, fs=rec.fs, low=dcfg['bandpass_low'], high=dcfg['bandpass_high']))
        sigs = np.stack(sigs, axis=0)              # (n_leads, L)
        samples, symbols = read_beat_annotations(mitdb / rec_id)
        beats = extract_beats(sigs, samples, symbols, beat_len=beat_len, n_classes=n_classes,
                              fs=rec.fs, n_context=n_context)
        split = split_of(rec_id, val_records)
        snr_list = test_snrs if split == 'test' else train_snrs

        for bi, (r_s, beat, sym, lbl, rr) in enumerate(beats):
            snr = float(snr_list[bi % len(snr_list)])
            grp = groups[bi % len(groups)]
            noisy = np.stack([make_noisy_segment(beat[l], noise_bank, target_snr_db=snr,
                                                 group_name=grp, seed=gid + 42 + l)[0]
                              for l in range(beat.shape[0])], axis=0)   # (n_leads, L)
            ntype = 'mix'
            npz_path = beat_dir / f'{rec_id}_b{bi:05d}.npz'
            np.savez_compressed(npz_path,
                                clean=beat.astype(np.float32),
                                noisy=noisy.astype(np.float32),
                                rr=rr.astype(np.float32),
                                label=np.int64(lbl))
            rows.append({
                'record_id': rec_id, 'beat_index': bi, 'r_sample': r_s, 'split': split,
                'symbol': sym, 'label': int(lbl), 'noise_type': ntype, 'noise_snr': snr,
                'npz_path': str(npz_path),
            })
            gid += 1

    df = pd.DataFrame(rows)
    out_csv = out_root / 'manifest_cls.csv'
    df.to_csv(out_csv, index=False)
    print(f'Saved {out_csv} | tong beat: {len(df)}')
    print('Phan bo theo (split, label 0=Normal/1=Abnormal):')
    print(df.groupby(['split', 'label']).size())


if __name__ == '__main__':
    main()
