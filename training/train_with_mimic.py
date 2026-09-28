"""Retrain the ECG matching model with MIMIC-IV-ECG added, and compare to v1.

Run from the repo's training/ folder (it finds prepare.py in ../model_v2):
    python train_with_mimic.py
Options:
    --sjlife   path to the SJLIFE repo        (default ~/SJLIFE-data)
    --mimic    path to mimic_leadI.npz        (default ~/mimic_leadI/mimic_leadI.npz)
    --epochs   CNN training rounds            (default 100, as used for v2; use 5 for a quick test)
    --v1       folder with the v1 .h5 files   (default ../model_v1)
    --out      where to save the new model    (default ../model_retrained)

What it does
  1. Loads SJLIFE (clinical vs Apple Watch, 243 patients) and your MIMIC download.
  2. Uses the SAME 48 SJLIFE test patients as v1, so the scores are directly comparable.
  3. Trains the feature neural net and the siamese CNN on SJLIFE train patients + MIMIC.
  4. Scores the 48 held-out SJLIFE patients, plus 200 held-out MIMIC patients.
  5. Saves feature_nn.h5, siamese_cnn.h5 and results.json into the --out folder.
     Existing models are never overwritten.
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import tensorflow as tf, keras
from keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(HERE, "..", "model_v2"), HERE):
    if os.path.exists(os.path.join(_d, "prepare.py")):
        sys.path.insert(0, _d)
from prepare import prepare_ecg, beat_vector

V1_RESULTS = {"feature_nn": 0.938, "siamese_cnn": 0.929, "ensemble": 0.948}   # training/test_results.json

# ---------------------------------------------------------------- data
def load_sjlife(root):
    meta = pd.read_csv(f"{root}/shared_paired_data_243.csv")
    ns = meta.apple_loc_ECG_243.str.extract(r"(\d+)\.npy")[0].astype(int)
    C, W = [], []
    for n in ns:
        c = np.load(f"{root}/ClinicalECGs_full_243/clinical_ecg_{n}.npy")[0, 0]
        a = np.load(f"{root}/AppleECGs_full_243/apple_ecg_{n}.npy")[int(1.5 * 512):]
        L = 10 * 512
        C.append(prepare_ecg(c, 500))
        W.append([prepare_ecg(a[s:s + L], 512) for s in (0, L, len(a) - L)])
    return np.array(C), np.array(W)                      # (243, 2500), (243, 3, 2500)

def load_mimic(path):
    d = np.load(path)
    X = np.array([prepare_ecg(x.astype(np.float32), 500) for x in d["signals"]])
    return X, d["subject_id"]

# ---------------------------------------------------------------- scoring helpers
def pair_feats(a, b):
    return np.hstack([np.abs(a - b), a * b])

def auroc_top1(S_list):
    """S_list: square score matrices, S[i, j] = score(query i, candidate j); diagonal = true match."""
    n = S_list[0].shape[0]
    y = np.tile(np.eye(n).ravel(), len(S_list))
    s = np.concatenate([S.ravel() for S in S_list])
    return float(roc_auc_score(y, s)), float(np.mean(S_list[0].argmax(1) == np.arange(n)))

def nn_matrix(model, A, B):
    n, m = len(A), len(B)
    P = pair_feats(np.repeat(A, m, 0), np.tile(B, (n, 1)))
    return model.predict(P, batch_size=4096, verbose=0).reshape(n, m)

def cnn_matrix(model, A, B):
    n, m = len(A), len(B)
    a = np.repeat(A, m, 0)[..., None]; b = np.tile(B, (n, 1))[..., None]
    return model.predict([a, b], batch_size=512, verbose=0).reshape(n, m)

# ---------------------------------------------------------------- models (same design as v1)
def build_mlp(dim, hidden=(16,), drop=0.3):
    inp = keras.Input((dim,)); x = layers.Normalization(name="scaler")(inp)
    for h in hidden:
        x = layers.Dense(h, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        x = layers.Dropout(drop)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))

def build_siamese(emb=64):
    inp = keras.Input((2500, 1)); x = inp
    for f, k in [(16, 9), (32, 7), (32, 7), (64, 5), (64, 5), (128, 3)]:
        x = layers.Conv1D(f, k, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x); x = layers.Dropout(0.3)(x)
    enc = keras.Model(inp, layers.Dense(emb)(x), name="encoder")
    a, b = keras.Input((2500, 1), name="ecg_a"), keras.Input((2500, 1), name="ecg_b")
    ea, eb = enc(a), enc(b)
    d = layers.Subtract()([ea, eb]); d = layers.Multiply()([d, d])
    h = layers.Concatenate()([d, layers.Multiply()([ea, eb])])
    h = layers.Dense(32, activation="relu")(h)
    return keras.Model([a, b], layers.Dense(1, activation="sigmoid", name="score")(h))

# ---------------------------------------------------------------- training pairs
def mimic_groups(pid):
    """dict patient -> list of row indices (only patients with 2+ ECGs)."""
    g = pd.Series(np.arange(len(pid))).groupby(pid).apply(list)
    return [v for v in g.values if len(v) >= 2]

def sample_pairs(rng, sj_idx, C, W, groups, neg=3):
    """Index pairs for one round. Returns lists of (source, i, j, label)."""
    out = []
    for i in sj_idx:                                             # SJLIFE: clinical vs watch
        for _ in range(2):
            out.append(("sj", i, (i, rng.integers(3)), 1))
            for j in rng.choice(sj_idx[sj_idx != i], neg, replace=False):
                out.append(("sj", i, (j, rng.integers(3)), 0))
    for gi, g in enumerate(groups):                              # MIMIC: ECG vs another ECG
        a, b = rng.choice(g, 2, replace=False)
        out.append(("mm", a, b, 1))
        for _ in range(neg):
            o = groups[rng.integers(len(groups))]
            if o is not g:
                out.append(("mm", a, o[rng.integers(len(o))], 0))
    return out

def materialize(pairs, sj_c, sj_w, mm_x):
    A, B, y = [], [], []
    for src, i, j, lab in pairs:
        if src == "sj":
            A.append(sj_c[i]); B.append(sj_w[j[0], j[1]])
        else:
            A.append(mm_x[i]); B.append(mm_x[j])
        y.append(lab)
    return np.array(A), np.array(B), np.array(y, dtype="float32")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sjlife", default=os.path.expanduser("~/SJLIFE-data"))
    ap.add_argument("--mimic", default=os.path.expanduser("~/mimic_leadI/mimic_leadI.npz"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--v1", default=next((d for d in (os.path.join(HERE, "..", "model_v1"),
                    os.path.join(HERE, "v1_backup"), HERE) if os.path.exists(os.path.join(d, "feature_nn.h5"))), HERE))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "model_retrained"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keras.utils.set_random_seed(0); rng = np.random.default_rng(0)
    t0 = time.time()

    print("loading SJLIFE ...", flush=True)
    C, W = load_sjlife(a.sjlife)
    print("loading MIMIC ...", flush=True)
    MX, MP = load_mimic(a.mimic)
    print(f"  {len(MX):,} MIMIC ECGs from {len(np.unique(MP)):,} patients", flush=True)

    # same SJLIFE split as v1
    perm = np.random.default_rng(42).permutation(243)
    TEST, DEV = np.sort(perm[:48]), np.sort(perm[48:])

    # hold out 200 MIMIC patients for a second check
    groups = mimic_groups(MP)
    order = np.random.default_rng(7).permutation(len(groups))
    n_hold = min(200, len(groups) // 5)
    hold = [groups[i] for i in order[:n_hold]]
    train_groups = [groups[i] for i in order[n_hold:]]
    print(f"  MIMIC: {len(train_groups):,} train patients, {len(hold)} held out", flush=True)

    print("computing heartbeat features ...", flush=True)
    Vc = np.array([beat_vector(x) for x in C])
    Vw = np.array([[beat_vector(x) for x in w] for w in W])
    VM = np.array([beat_vector(x) for x in MX])

    # ---- feature neural net
    print("training feature neural net ...", flush=True)
    idx = sample_pairs(rng, DEV, C, W, train_groups, neg=3)
    for _ in range(2):                                           # a second round for more MIMIC variety
        idx += [p for p in sample_pairs(rng, DEV, C, W, train_groups, neg=3) if p[0] == "mm"]
    Xa, Xb, y = materialize(idx, Vc, Vw, VM)
    X = pair_feats(Xa, Xb)
    nn = build_mlp(X.shape[1]); nn.get_layer("scaler").adapt(X)
    nn.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy")
    nn.fit(X, y, epochs=40, batch_size=256, verbose=0,
           class_weight={0: 1.0, 1: float((y == 0).sum() / (y == 1).sum())})

    # ---- siamese CNN
    print(f"training siamese CNN ({a.epochs} rounds) ...", flush=True)
    cnn = build_siamese()
    cnn.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy")
    for ep in range(a.epochs):
        t = time.time()
        A, B, yy = materialize(sample_pairs(rng, DEV, C, W, train_groups, neg=3), C, W, MX)
        sw = rng.random(len(yy)) < 0.5
        A[sw], B[sw] = B[sw].copy(), A[sw].copy()
        for Z in (A, B):
            Z *= rng.uniform(0.8, 1.2, (len(Z), 1)); Z += rng.normal(0, 0.05, Z.shape)
        cnn.fit([A[..., None], B[..., None]], yy, batch_size=64, epochs=1, verbose=0,
                class_weight={0: 1.0, 1: 3.0})
        if ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == a.epochs:
            print(f"  round {ep + 1}/{a.epochs} ({time.time() - t:.0f}s per round)", flush=True)

    # ---- evaluate on the 48 SJLIFE test patients (comparable to v1)
    print("scoring held-out patients ...", flush=True)
    S_nn = [nn_matrix(nn, Vc[TEST], Vw[TEST, k]) for k in range(3)]
    S_cnn = [cnn_matrix(cnn, C[TEST], W[TEST, k]) for k in range(3)]
    S_ens = [0.5 * p + 0.5 * q for p, q in zip(S_nn, S_cnn)]
    res = {"sjlife_test": {"feature_nn": auroc_top1(S_nn), "siamese_cnn": auroc_top1(S_cnn),
                           "ensemble": auroc_top1(S_ens)}}

    # ---- evaluate on held-out MIMIC patients (first ECG vs second ECG)
    q = np.array([g[0] for g in hold]); c = np.array([g[1] for g in hold])
    M_nn = nn_matrix(nn, VM[q], VM[c]); M_cnn = cnn_matrix(cnn, MX[q], MX[c])
    res["mimic_holdout"] = {"feature_nn": auroc_top1([M_nn]), "siamese_cnn": auroc_top1([M_cnn]),
                            "ensemble": auroc_top1([0.5 * M_nn + 0.5 * M_cnn])}
    # v1 on the same MIMIC patients, so both tables are apples to apples
    try:
        v1_nn = keras.models.load_model(os.path.join(a.v1, "feature_nn.h5"), compile=False)
        v1_cnn = keras.models.load_model(os.path.join(a.v1, "siamese_cnn.h5"), compile=False)
        P_nn = nn_matrix(v1_nn, VM[q], VM[c]); P_cnn = cnn_matrix(v1_cnn, MX[q], MX[c])
        res["mimic_holdout_v1"] = {"feature_nn": auroc_top1([P_nn]), "siamese_cnn": auroc_top1([P_cnn]),
                                   "ensemble": auroc_top1([0.5 * P_nn + 0.5 * P_cnn])}
    except Exception as e:
        print("  (could not score v1 on MIMIC:", e, ")")
    res["mimic_ecgs_used"] = int(len(MX)); res["cnn_rounds"] = a.epochs

    nn.save(os.path.join(a.out, "feature_nn.h5")); cnn.save(os.path.join(a.out, "siamese_cnn.h5"))
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    print("\n=== SJLIFE test (48 patients, same as v1) ===")
    print(f"{'model':12s} {'v1 AUROC':>9s} {'v2 AUROC':>9s} {'v2 top1':>8s}")
    for k in ("feature_nn", "siamese_cnn", "ensemble"):
        au, t1 = res["sjlife_test"][k]
        print(f"{k:12s} {V1_RESULTS[k]:9.3f} {au:9.3f} {t1:8.1%}")
    print(f"\n=== MIMIC held-out ({len(hold)} patients, clinical vs clinical) ===")
    v1m = res.get("mimic_holdout_v1", {})
    print(f"{'model':12s} {'v1 AUROC':>9s} {'v2 AUROC':>9s} {'v2 top1':>8s}")
    for k, (au, t1) in res["mimic_holdout"].items():
        v1a = f"{v1m[k][0]:9.3f}" if k in v1m else f"{'n/a':>9s}"
        print(f"{k:12s} {v1a} {au:9.3f} {t1:8.1%}")
    print(f"\nsaved to {a.out}  ({(time.time() - t0) / 60:.1f} min total)")
    print("Keep v2 only if the ensemble SJLIFE AUROC beats 0.948.")

if __name__ == "__main__":
    main()
