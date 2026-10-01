import numpy as np


def _scale_noise_to_snr(clean, noise, target_snr_db, eps=1e-12):
    clean_power = np.mean(clean ** 2) + eps
    noise_power = np.mean(noise ** 2) + eps
    target_noise_power = clean_power / (10 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return noise * scale


def random_crop_or_tile(noise, target_len, rng):
    if len(noise) >= target_len:
        start = rng.integers(0, len(noise) - target_len + 1)
        return noise[start:start + target_len]
    reps = int(np.ceil(target_len / len(noise)))
    tiled = np.tile(noise, reps)
    start = rng.integers(0, len(tiled) - target_len + 1)
    return tiled[start:start + target_len]


def sample_noise_group(noise_bank, group_name, rng):
    keys = ['bw', 'ma', 'em']
    if group_name == 'single':
        chosen = [rng.choice(keys)]
    elif group_name == 'double':
        chosen = list(rng.choice(keys, size=2, replace=False))
    elif group_name == 'triple':
        chosen = keys
    else:
        raise ValueError(group_name)
    return chosen


def make_noisy_segment(clean, noise_bank, target_snr_db, group_name, seed=42):
    rng = np.random.default_rng(seed)
    names = sample_noise_group(noise_bank, group_name, rng)
    noise = np.zeros_like(clean, dtype=np.float64)
    for name in names:
        n = random_crop_or_tile(noise_bank[name], len(clean), rng)
        noise += n.astype(np.float64)
    noise_scaled = _scale_noise_to_snr(clean, noise, target_snr_db)
    noisy = clean + noise_scaled
    noise_type = '+'.join(sorted(names))
    return noisy.astype(np.float32), noise_scaled.astype(np.float32), noise_type
