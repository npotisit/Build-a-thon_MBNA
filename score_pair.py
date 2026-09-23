"""Score how likely two Lead I ECGs come from the same patient.

Usage (demo on the public SJLIFE data):
    python score_pair.py --demo path/to/SJLIFE-repo
Usage in code:
    from score_pair import MatchModel
    model = MatchModel()                       # loads the two .h5 files next to this script
    score = model.score(ecg_a, fs_a, ecg_b, fs_b)
"""
import os, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import keras
from prepare import prepare_ecg, beat_vector

HERE = os.path.dirname(os.path.abspath(__file__))

class MatchModel:
    def __init__(self, folder=HERE):
        self.feature_nn = keras.models.load_model(os.path.join(folder, "feature_nn.h5"), compile=False)
        self.siamese = keras.models.load_model(os.path.join(folder, "siamese_cnn.h5"), compile=False)

    def score(self, ecg_a, fs_a, ecg_b, fs_b):
        """ecg_*: 1D array of Lead I samples (any units, up to 10 s; longer is center cropped).
        fs_*: sampling rate in Hz. Returns a float in [0, 1]; higher = more likely the same patient."""
        a, b = prepare_ecg(ecg_a, fs_a), prepare_ecg(ecg_b, fs_b)
        va, vb = beat_vector(a), beat_vector(b)
        pair = np.hstack([np.abs(va - vb), va * vb])[None, :]
        s1 = float(self.feature_nn.predict(pair, verbose=0)[0, 0])
        s2 = float(self.siamese.predict([a[None, :, None], b[None, :, None]], verbose=0)[0, 0])
        return 0.5 * s1 + 0.5 * s2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", required=True, help="path to the SJLIFE repository")
    root = ap.parse_args().demo
    m = MatchModel()
    clin = lambda n: np.load(f"{root}/ClinicalECGs_full_243/clinical_ecg_{n}.npy")[0, 0]          # lead I, 500 Hz
    watch = lambda n: np.load(f"{root}/AppleECGs_full_243/apple_ecg_{n}.npy")[768:768 + 5120]     # 10 s, 512 Hz
    print("same patient (132 vs 132):     %.3f" % m.score(clin(132), 500, watch(132), 512))
    print("different patients (132 vs 140): %.3f" % m.score(clin(132), 500, watch(140), 512))
