"""Score how likely two Lead I ECGs come from the same patient.

Usage (demo on the public SJLIFE data):
    python score_pair.py --demo path/to/SJLIFE-data
Usage in code:
    from score_pair import MatchModel
    model = MatchModel()                           # loads the model files next to this script
    score = model.score(ecg_a, fs_a, ecg_b, fs_b)  # float in [0, 1]

Inputs can be any length and sampling rate. Windowing (how long recordings are handled):
  * Recordings of about 10 s or less are used as they are (zero padded if short).
  * Longer recordings (for example a 30 s Apple Watch ECG): if over 20 s, the first 1.5 s is
    skipped (the Apple Watch often shows a spike while the finger settles on the crown), then
    the MIDDLE 10 s is used (default, windows="middle").
  * Optional windows="three": score the first, middle and last 10 s and average them.
    We tested both on held-out patients; the middle window scored as well or better.

Ensemble: final score = w * mean(CNN scores) + (1 - w) * feature net score.
Settings come from config.json in the model folder if present; otherwise
feature_nn.h5 + siamese_cnn.h5 with w = 0.5.
"""
# WHAT THIS FILE DOES
# This is the file the judges run. It loads our trained models and turns two raw ECGs into
# one score between 0 and 1 (higher = more likely the same person).
#   raw ECG A, raw ECG B
#     -> split long recordings into 10 s windows       (split_windows)
#     -> clean each window                             (prepare_ecg, in prepare.py)
#     -> feature net scores the heartbeat summaries    (beat_vector + feature_nn.h5)
#     -> CNN(s) score the full cleaned signals         (siamese_cnn*.h5)
#     -> blend the two: w * CNN + (1 - w) * feature net, averaged over all window pairs

import os, json, glob, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide TensorFlow's startup messages
import numpy as np
import keras
from prepare import prepare_ecg, beat_vector

HERE = os.path.dirname(os.path.abspath(__file__))   # the folder this file lives in

def split_windows(x, fs, mode="middle"):
    """Raw 1D signal -> list of raw windows of about 10 s each."""
    x = np.asarray(x, dtype=float).ravel()
    L = int(round(10 * fs))                          # number of readings in 10 seconds
    # Short recordings (up to 11 s): use as they are, one window.
    if len(x) <= int(1.1 * L):
        return [x]
    # Long recordings (over 20 s, like the 30 s watch ECG): skip the first 1.5 s,
    # where the watch often shows a spike from the finger touching the crown.
    if len(x) > 2 * L:
        x = x[int(round(1.5 * fs)):]
    # "middle" mode: just the middle 10 s.
    if mode == "middle" or len(x) <= L:
        s = max(0, (len(x) - L) // 2)
        return [x[s:s + L]]
    # "three" mode (optional): first, middle, and last 10 s, averaged. Luke's suggestion; in our
    # test it tied the middle window on AUROC, so the default stays "middle".
    return [x[s:s + L] for s in (0, (len(x) - L) // 2, len(x) - L)]

class MatchModel:
    def __init__(self, folder=HERE, windows="middle"):
        # Optional config.json says which model files to load and how to blend them.
        cfg_path = os.path.join(folder, "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        # compile=False: we only use the models to predict, not to train, so skip training setup.
        load = lambda f: keras.models.load_model(os.path.join(folder, f), compile=False)
        # Model 1, the feature net: compares heartbeat summaries (180 numbers each).
        self.feature_nn = load(cfg.get("feature_nn", "feature_nn.h5"))
        # Model 2, the Siamese CNN(s): compare the full cleaned 10 s signals.
        # If several are listed (e.g. 3 trained with different random starts), all are used.
        cnn_files = cfg.get("cnns") or sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "siamese_cnn*.h5")))
        self.cnns = [load(f) for f in cnn_files]
        # w = how much of the final vote the CNN gets (0.8 means 80% CNN, 20% feature net).
        self.w = float(cfg.get("w", 0.5))
        self.windows = cfg.get("windows", windows)

    def score(self, ecg_a, fs_a, ecg_b, fs_b):
        """ecg_*: 1D array of raw Lead I samples (any units, any length).
        fs_*: sampling rate in Hz. Returns a float in [0, 1]; higher = more likely the same patient."""
        # Split each recording into windows and clean each window.
        A = [prepare_ecg(x, fs_a) for x in split_windows(ecg_a, fs_a, self.windows)]
        B = [prepare_ecg(x, fs_b) for x in split_windows(ecg_b, fs_b, self.windows)]
        # Pair every window of A with every window of B
        # (a 10 s hospital ECG vs a 30 s watch ECG gives 1 x 3 = 3 pairs).
        pa = np.array([a for a in A for _ in B]); pb = np.array([b for _ in A for b in B])

        # Feature net: summarize each window, then compare the two summaries.
        # |difference| = how different the heartbeats are; product = how they move together.
        # (Similar to adding interaction terms in a regression.)
        va = np.array([beat_vector(a) for a in pa]); vb = np.array([beat_vector(b) for b in pb])
        s_nn = self.feature_nn.predict(np.hstack([np.abs(va - vb), va * vb]), verbose=0).ravel()

        # CNN(s): each looks at the two full signals; average the CNNs together.
        # [..., None] adds the "channel" dimension the CNN expects (shape 2500 x 1).
        s_cnn = np.mean([m.predict([pa[..., None], pb[..., None]], verbose=0).ravel() for m in self.cnns], 0)

        # Blend the two models, then average over all window pairs -> one final score.
        return float(np.mean(self.w * s_cnn + (1 - self.w) * s_nn))

if __name__ == "__main__":
    # Demo: runs when you type "python score_pair.py" in the terminal.
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=os.path.expanduser("~/SJLIFE-data"), help="path to the SJLIFE repository")
    root = ap.parse_args().demo
    m = MatchModel()
    clin = lambda n: np.load(f"{root}/ClinicalECGs_full_243/clinical_ecg_{n}.npy")[0, 0]   # lead I, 500 Hz, 10 s
    watch = lambda n: np.load(f"{root}/AppleECGs_full_243/apple_ecg_{n}.npy")              # full 30 s, 512 Hz
    # Same person (hospital ECG vs watch ECG of patient 132): should score high.
    print("same patient (132 vs 132):       %.3f" % m.score(clin(132), 500, watch(132), 512))
    # Different people (patient 132 vs patient 140): should score low.
    print("different patients (132 vs 140): %.3f" % m.score(clin(132), 500, watch(140), 512))
