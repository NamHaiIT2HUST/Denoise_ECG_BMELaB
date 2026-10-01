import numpy as np


def sliding_segments(x, segment_length=8192, overlap_ratio=0.5):
    hop = int(segment_length * (1.0 - overlap_ratio))
    hop = max(1, hop)
    out = []
    for start in range(0, len(x) - segment_length + 1, hop):
        out.append((start, x[start:start + segment_length]))
    return out


def keep_energy_middle_percentile(segments, low_pct=5, high_pct=95):
    energies = np.array([np.sum(seg ** 2) for _, seg in segments], dtype=np.float64)
    lo = np.percentile(energies, low_pct)
    hi = np.percentile(energies, high_pct)
    kept = [(s, seg) for (s, seg), e in zip(segments, energies) if lo <= e <= hi]
    return kept, {'energy_low': float(lo), 'energy_high': float(hi)}
