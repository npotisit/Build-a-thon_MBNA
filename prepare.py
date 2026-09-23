"""Shared Lead I preparation. Used identically for training and for scoring."""
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from fractions import Fraction

FS = 250          # competition sampling rate
N = 2500          # 10 seconds at 250 Hz

def _clean(x):
    x = np.asarray(x, dtype=float).ravel()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():                                   # fill gaps by straight-line interpolation
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x

def prepare_ecg(x, fs=FS, norm="robust", band=(0.5, 40.0)):
    """Raw Lead I at any sampling rate -> 2,500 samples at 250 Hz, filtered and scaled.
    Shorter recordings are zero padded at the end; longer ones are center cropped to 10 s."""
    x = _clean(x)
    if fs != FS:
        fr = Fraction(FS, int(round(fs))).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator)
    if len(x) > 3 * FS // 2:                        # filter needs a bit of signal to work with
        hi = min(band[1], 0.45 * FS)
        b, a = butter(3, [band[0] / (FS / 2), hi / (FS / 2)], btype="band")
        x = filtfilt(b, a, x)
    if len(x) > N:
        s = (len(x) - N) // 2
        x = x[s:s + N]
    x = x - np.median(x)
    if norm == "robust":
        scale = 1.4826 * np.median(np.abs(x))       # robust spread, ignores spikes
    else:
        scale = x.std()
    x = x / scale if scale > 1e-8 else x
    x = np.clip(x, -50, 50)                         # tame extreme artifacts
    out = np.zeros(N, dtype=np.float32)
    out[:len(x)] = x
    return out

# ---------- heartbeat features (used by the baseline and the feature neural net) ----------
PRE, POST = int(0.25 * FS), int(0.45 * FS)

def r_peaks(x):
    d = np.gradient(x) ** 2
    w = int(0.12 * FS)
    env = np.convolve(d, np.ones(w) / w, mode="same")
    cand, pr = find_peaks(env, distance=int(0.33 * FS), height=0)
    if len(cand) == 0:
        return cand
    h = np.sort(pr["peak_heights"])
    pk = cand[pr["peak_heights"] > 0.3 * np.median(h[len(h) // 2:])]
    g, w = np.abs(np.gradient(x)), int(0.06 * FS)
    return np.unique([max(p - w, 0) + np.argmax(g[max(p - w, 0):p + w]) for p in pk])

def beat_vector(x):
    """Median heartbeat (shape, 175 samples) + 6 summary numbers. x = output of prepare_ecg."""
    nz = np.flatnonzero(x)
    x = x[: nz.max() + 1] if len(nz) else x
    pk = r_peaks(x)
    pk = pk[(pk > PRE) & (pk < len(x) - POST)]
    if len(pk) < 2:
        return np.zeros(PRE + POST + 6, dtype=np.float32)
    beats = np.array([x[p - PRE:p + POST] for p in pk])
    med = np.median(beats, axis=0)
    med = med - np.median(med[: int(0.1 * FS)])
    shape = med / (np.abs(med).max() + 1e-8)
    rr = np.diff(pk) / FS
    t_seg = shape[PRE + int(0.15 * FS): PRE + int(0.42 * FS)]
    summ = [60 / np.median(rr) / 100,
            np.log1p(rr.std() * 1000) / 5,
            shape[PRE - 6:PRE + 4].max(), shape[PRE - 4:PRE + 10].min(),
            t_seg[np.argmax(np.abs(t_seg))],
            np.median([np.corrcoef(b, med)[0, 1] for b in beats])]
    return np.concatenate([shape, summ]).astype(np.float32)
