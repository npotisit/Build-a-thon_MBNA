"""Compare windowing strategies for 30 s Apple Watch ECGs on the PRACTICE test.

Run after experiment.py (it uses the models experiment.py saved in experiments/):
    python eval_windows.py --nn finetune --w 0.7

Strategies (clinical ECGs are 10 s, so only the watch side changes):
  middle : middle 10 s only (what the organizers default to)
  first  : first 10 s (after skipping the 1.5 s finger spike)
  three  : first + middle + last 10 s, scored separately and averaged (Luke's suggestion)
"""
import os, sys, glob, argparse, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, keras

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _d in (os.path.join(HERE, "..", "model_v3"), HERE):      # in the repo, prepare/score_pair live in ../model_v3
    if os.path.exists(os.path.join(_d, "score_pair.py")):
        sys.path.insert(0, _d)
from prepare import prepare_ecg, beat_vector
from score_pair import split_windows
from train_with_mimic import auroc_top1, nn_matrix
from experiment import cnn_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sjlife", default=os.path.expanduser("~/SJLIFE-data"))
    ap.add_argument("--models", default=os.path.join(HERE, "experiments"))
    ap.add_argument("--nn", default="finetune", help="joint or finetune")
    ap.add_argument("--w", type=float, default=0.5, help="CNN weight")
    ap.add_argument("--prefix", default="explore", help="explore (practice models) or final")
    a = ap.parse_args()

    nn = keras.models.load_model(os.path.join(a.models, f"{a.prefix}_feature_nn_{a.nn}.h5"), compile=False)
    cnns = [keras.models.load_model(p, compile=False)
            for p in sorted(glob.glob(os.path.join(a.models, f"{a.prefix}_siamese_cnn_*.h5")))]
    print(f"loaded {len(cnns)} CNN(s) + {a.nn} feature net, CNN weight {a.w}")

    perm = np.random.default_rng(42).permutation(243)                 # same splits as experiment.py
    dev = np.sort(perm[48:]); p2 = np.random.default_rng(11).permutation(len(dev))
    idx = np.sort(dev[p2[:39]]) if a.prefix == "explore" else np.sort(perm[:48])
    label = "PRACTICE TEST (39 patients)" if a.prefix == "explore" else "LOCKED TEST (48 patients)"

    meta = pd.read_csv(f"{a.sjlife}/shared_paired_data_243.csv")
    ns = meta.apple_loc_ECG_243.str.extract(r"(\d+)\.npy")[0].astype(int).values[idx]
    C = np.array([prepare_ecg(np.load(f"{a.sjlife}/ClinicalECGs_full_243/clinical_ecg_{n}.npy")[0, 0], 500) for n in ns])
    raw = [np.load(f"{a.sjlife}/AppleECGs_full_243/apple_ecg_{n}.npy") for n in ns]
    Vc = np.array([beat_vector(x) for x in C])

    # For one set of watch windows: build the full clinical x watch score grid, blending
    # the averaged CNNs and the feature net with weight w.
    def ens(Wp):                                                      # clinical x watch score grid for one window set
        Vw = np.array([beat_vector(x) for x in Wp])
        s_cnn = np.mean([cnn_matrix(m, C, Wp) for m in cnns], 0)
        return a.w * s_cnn + (1 - a.w) * nn_matrix(nn, Vc, Vw)

    wins = [[prepare_ecg(x, 512) for x in split_windows(r, 512, "three")] for r in raw]
    mids = np.array([prepare_ecg(split_windows(r, 512, "middle")[0], 512) for r in raw])
    # Score each of the three windows separately; "three windows averaged" is their mean.
    per_window = [ens(np.array([w[k] for w in wins])) for k in range(3)]

    print(f"\n=== {label}, clinical vs Apple Watch ===")
    print(f"  {'strategy':34s} {'AUROC':>7s} {'top1':>7s}")
    for name, S in [("middle 10 s", ens(mids)), ("first 10 s", per_window[0]),
                    ("last 10 s", per_window[2]), ("three windows averaged", np.mean(per_window, 0))]:
        au, t1 = auroc_top1([S])
        print(f"  {name:34s} {au:7.3f} {t1:7.1%}")

if __name__ == "__main__":
    main()
