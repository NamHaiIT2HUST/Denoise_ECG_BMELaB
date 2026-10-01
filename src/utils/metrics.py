import numpy as np


def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(x, y):
    return float(np.mean(np.abs(x - y)))


def prd(clean, pred, eps=1e-12):
    num = np.sqrt(np.sum((clean - pred) ** 2))
    den = np.sqrt(np.sum(clean ** 2)) + eps
    return float(100.0 * num / den)


def cosine_similarity(x, y, eps=1e-12):
    x = x.reshape(-1)
    y = y.reshape(-1)
    den = (np.linalg.norm(x) * np.linalg.norm(y)) + eps
    return float(np.dot(x, y) / den)


def snr_out(clean, pred, eps=1e-12):
    num = np.sum(clean ** 2)
    den = np.sum((clean - pred) ** 2) + eps
    return float(10.0 * np.log10(num / den + eps))


def snr_in(clean, noisy, eps=1e-12):
    num = np.sum(clean ** 2)
    den = np.sum((clean - noisy) ** 2) + eps
    return float(10.0 * np.log10(num / den + eps))


def snr_improvement(clean, noisy, pred):
    return snr_out(clean, pred) - snr_in(clean, noisy)

