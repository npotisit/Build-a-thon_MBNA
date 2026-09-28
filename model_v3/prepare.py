"""Shared Lead I preparation. Used identically for training and for scoring."""
# WHAT THIS FILE DOES
# Every ECG, whether from a hospital machine or an Apple Watch, goes through this file
# before a model sees it. It turns messy raw signals into one standard format, so the
# models compare heart SHAPES, not differences between devices.
#   prepare_ecg  : cleans one raw signal -> 2,500 numbers (10 s at 250 Hz). Used by BOTH models.
#   beat_vector  : summarizes a cleaned signal -> 180 numbers. Used by the feature net only.

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from fractions import Fraction

FS = 250          # competition sampling rate
N = 2500          # 10 seconds at 250 Hz

def _clean(x):
    # Turn the input into a flat list of decimal numbers.
    x = np.asarray(x, dtype=float).ravel()
    # Find missing or broken values (NaN or infinity), e.g. from a dropped sensor reading.
    bad = ~np.isfinite(x)
    if bad.all():
        # Nothing usable at all: return a flat line instead of crashing.
        return np.zeros_like(x)
    if bad.any():                                   # fill gaps by straight-line interpolation
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x

def prepare_ecg(x, fs=FS, norm="robust", band=(0.5, 40.0)):
    """Raw Lead I at any sampling rate -> 2,500 samples at 250 Hz, filtered and scaled.
    Shorter recordings are zero padded at the end; longer ones are center cropped to 10 s."""
    # Step 1: fix missing values.
    x = _clean(x)

    # Step 2: RESAMPLE to 250 readings per second.
    # Hospital ECGs record at 500 per second, the Apple Watch at 512. Converting both to 250
    # means "one second" is the same number of points for every device.
    if fs != FS:
        fr = Fraction(FS, int(round(fs))).limit_denominator(1000)   # e.g. 500 -> 250 is ratio 1/2
        x = resample_poly(x, fr.numerator, fr.denominator)

    # Step 3: FILTER, keeping only frequencies between 0.5 and 40 Hz (a "band pass").
    #   Below 0.5 Hz: slow drift of the baseline, caused by breathing or body movement.
    #   Above 40 Hz:  electrical noise and muscle twitches.
    # The heartbeat itself lives in between. filtfilt runs the filter forward and backward so
    # the waves are not shifted in time.
    if len(x) > 3 * FS // 2:                        # filter needs a bit of signal to work with
        hi = min(band[1], 0.45 * FS)
        b, a = butter(3, [band[0] / (FS / 2), hi / (FS / 2)], btype="band")
        x = filtfilt(b, a, x)

    # Step 4: TRIM to 10 seconds. If longer, keep the middle 10 s.
    # (score_pair.py handles long recordings smarter by splitting them into windows first.)
    if len(x) > N:
        s = (len(x) - N) // 2
        x = x[s:s + N]

    # Step 5: RESCALE so only the shape matters, not the device's voltage.
    # Like standardizing a variable in regression: subtract the center, divide by the spread.
    # We use the median and the median absolute deviation (MAD) instead of the mean and
    # standard deviation, because one big spike in the signal would distort the mean and SD.
    # 1.4826 converts MAD to the same scale as a standard deviation.
    x = x - np.median(x)
    if norm == "robust":
        scale = 1.4826 * np.median(np.abs(x))       # robust spread, ignores spikes
    else:
        scale = x.std()
    x = x / scale if scale > 1e-8 else x            # skip dividing if the signal is flat
    x = np.clip(x, -50, 50)                         # tame extreme artifacts

    # Step 6: PAD short recordings with zeros at the end so every output is exactly 2,500 long.
    out = np.zeros(N, dtype=np.float32)
    out[:len(x)] = x
    return out

# ---------- heartbeat features (used by the baseline and the feature neural net) ----------
# Each heartbeat is cut out from 0.25 s before the main spike (R peak) to 0.45 s after it.
# That window holds the P wave (before), the QRS spike, and the T wave (after).
PRE, POST = int(0.25 * FS), int(0.45 * FS)          # 62 points before, 112 after = 174 points per beat

def r_peaks(x):
    """Find the R peaks: the tall, sharp spike of each heartbeat."""
    # The R peak is where the signal changes fastest. The slope (gradient), squared,
    # is large there and small elsewhere.
    d = np.gradient(x) ** 2
    # Smooth that with a short moving average (0.12 s) to get one bump per heartbeat.
    w = int(0.12 * FS)
    env = np.convolve(d, np.ones(w) / w, mode="same")
    # Candidate bumps must be at least 0.33 s apart (a heart can't beat faster than ~180 bpm).
    cand, pr = find_peaks(env, distance=int(0.33 * FS), height=0)
    if len(cand) == 0:
        return cand
    # Drop small bumps (noise): keep bumps taller than 30% of a typical big bump.
    h = np.sort(pr["peak_heights"])
    pk = cand[pr["peak_heights"] > 0.3 * np.median(h[len(h) // 2:])]
    # Fine tune: within 0.06 s of each bump, move to the point of steepest slope.
    g, w = np.abs(np.gradient(x)), int(0.06 * FS)
    return np.unique([max(p - w, 0) + np.argmax(g[max(p - w, 0):p + w]) for p in pk])

def beat_vector(x):
    """Median heartbeat (shape, 174 samples) + 6 summary numbers = 180. x = output of prepare_ecg."""
    # Ignore zero padding at the end so it isn't mistaken for signal.
    nz = np.flatnonzero(x)
    x = x[: nz.max() + 1] if len(nz) else x
    # Find heartbeats, keeping only ones with a full window around them.
    pk = r_peaks(x)
    pk = pk[(pk > PRE) & (pk < len(x) - POST)]
    if len(pk) < 2:
        # Too few beats found (very noisy or very short signal): return all zeros.
        return np.zeros(PRE + POST + 6, dtype=np.float32)

    # Stack every heartbeat on top of each other and take the MEDIAN beat.
    # The median cancels out noise that only affects one or two beats.
    beats = np.array([x[p - PRE:p + POST] for p in pk])
    med = np.median(beats, axis=0)
    med = med - np.median(med[: int(0.1 * FS)])     # set the flat start of the beat to zero
    shape = med / (np.abs(med).max() + 1e-8)        # scale so the tallest point is 1

    # Six summary numbers (each roughly scaled to a similar size so no one number dominates):
    rr = np.diff(pk) / FS                           # seconds between beats
    t_seg = shape[PRE + int(0.15 * FS): PRE + int(0.42 * FS)]     # where the T wave sits
    summ = [60 / np.median(rr) / 100,               # 1. heart rate (beats per minute / 100)
            np.log1p(rr.std() * 1000) / 5,          # 2. heart rate variability (log scaled)
            shape[PRE - 6:PRE + 4].max(),           # 3. height of the R peak area
            shape[PRE - 4:PRE + 10].min(),          # 4. depth of the dip right after it (S wave)
            t_seg[np.argmax(np.abs(t_seg))],        # 5. T wave height (can be up or down)
            np.median([np.corrcoef(b, med)[0, 1] for b in beats])]   # 6. beat consistency (0 to 1)
    # Final fingerprint: 174 shape points + 6 summaries = 180 numbers.
    return np.concatenate([shape, summ]).astype(np.float32)
