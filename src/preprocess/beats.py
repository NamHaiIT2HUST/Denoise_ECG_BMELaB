"""Beat-level annotation, AAMI mapping (2/4/5 lop), RR features, tach beat."""
import numpy as np

# Ky hieu MIT-BIH -> lop AAMI 5: 0=N,1=S,2=V,3=F,4=Q
AAMI5 = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4,
}

PACED_RECORDS = {'102', '104', '107', '217'}

DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
       '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
       '233', '234']


def class_names(n_classes):
    if n_classes == 2:
        return ['Normal', 'Abnormal']
    if n_classes == 3:
        return ['N', 'S', 'V']              # bo F va Q (hiem + mo ho hinh thai)
    if n_classes == 4:
        return ['N', 'S', 'V', 'F']         # bo Q
    return ['N', 'S', 'V', 'F', 'Q']


def aami_label(symbol, n_classes=5):
    c = AAMI5.get(symbol)
    if c is None:
        return None
    if n_classes == 2:
        return 0 if c == 0 else 1
    if n_classes == 3:
        return None if c >= 3 else c        # loai F va Q
    if n_classes == 4:
        return None if c == 4 else c        # loai Q
    return c


def read_beat_annotations(record_path):
    import wfdb
    ann = wfdb.rdann(str(record_path), 'atr')
    return ann.sample, ann.symbol


def extract_beats(sigs, samples, symbols, beat_len=256, n_classes=5, fs=360, n_context=1):
    """Tra ve list (r_sample, beat[n_leads*n_context, beat_len], symbol, label, rr[6]).

    n_context (le, vd 3): xep chong nhip TRUOC / HIEN TAI / SAU theo kenh ->
    model thay duoc ngu canh nhip (chia khoa nhan dien lop S den som).
    sigs: (n_leads, L) hoac (L,).
    """
    sigs = np.atleast_2d(sigs)                     # (n_leads, L)
    L = sigs.shape[1]
    half = beat_len // 2
    beats = [(int(s), sym, aami_label(sym, n_classes)) for s, sym in zip(samples, symbols)]
    beats = [(s, sym, lbl) for (s, sym, lbl) in beats if lbl is not None]
    rs = [s for (s, _, _) in beats]
    alld = np.diff(rs) if len(rs) > 1 else np.array([fs])
    glob = float(np.mean(alld)) if len(alld) else fs
    glob = glob if glob > 1 else fs
    out = []
    for i, (s, sym, lbl) in enumerate(beats):
        a, b = s - half, s - half + beat_len
        if a < 0 or b > L:
            continue
        pre = (rs[i] - rs[i - 1]) if i > 0 else (rs[i + 1] - rs[i] if i + 1 < len(rs) else fs)
        post = (rs[i + 1] - rs[i]) if i + 1 < len(rs) else pre
        j0 = max(1, i - 10)
        diffs = [rs[k] - rs[k - 1] for k in range(j0, i + 1)] if i > 0 else [pre]
        local = float(np.mean(diffs)) if diffs else pre
        local = local if local > 1 else (pre if pre > 1 else fs)
        # 6 dac trung RR (theo de Chazal): timing la chia khoa cho lop S
        rr = np.array([pre / fs, post / fs, pre / local, post / local,
                       pre / glob, post / glob], dtype=np.float32)
        # Ngu canh: xep chong n_context nhip lien ke (le, canh giua nhip hien tai)
        half_ctx = n_context // 2
        chans = []
        for k in range(-half_ctx, half_ctx + 1):
            j = min(max(i + k, 0), len(beats) - 1)          # bien: lap lai nhip dau/cuoi
            rj = beats[j][0]
            aj, bj = rj - half, rj - half + beat_len
            if aj < 0 or bj > L:                            # neu tran bien -> dung nhip hien tai
                aj, bj = a, b
            chans.append(sigs[:, aj:bj])
        beat = np.concatenate(chans, axis=0).astype(np.float32)   # (n_leads*n_context, beat_len)
        out.append((s, beat, sym, lbl, rr))
    return out


def split_of(record_id, val_records):
    if record_id in DS2:
        return 'test'
    if record_id in set(val_records):
        return 'val'
    return 'train'
