"""Try improvements to the ECG matching model against a PRACTICE test (validation set).

Put this file next to train_with_mimic.py and prepare.py, then run:

  Explore (default): trains everything once, scores on the practice test only.
      python experiment.py
  Final exam (run ONCE, after choosing settings from the explore table):
      python experiment.py --final --seeds 3 --nn finetune --w 0.7

Splits (patients never overlap):
  St. Jude: 48 locked test (same as v1/v2) | 39 practice test | 156 train
  MIMIC:    200 locked test | 300 practice test | rest train
Explore mode never touches the locked test patients.

What it tests
  1. Seed ensembling:   one CNN vs the average of N CNNs (--seeds, default 3)
  2. Feature net recipe: "joint" (v2: St. Jude + MIMIC mixed) vs
                         "finetune" (MIMIC first, then St. Jude watch pairs)
  3. Ensemble weight:   w * CNN + (1 - w) * feature net, for w = 0.3 ... 1.0
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np
import keras

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_with_mimic import (load_sjlife, load_mimic, beat_vector, pair_feats, auroc_top1,
                              nn_matrix, build_mlp, build_siamese, mimic_groups,
                              sample_pairs, materialize)

W_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def cnn_matrix(model, A, B, max_pairs=8000):
    """Memory-safe version: scores the (len(A) x len(B)) grid a few rows at a time."""
    rows = max(1, max_pairs // len(B)); out = []
    for i in range(0, len(A), rows):
        a = A[i:i + rows]; n = len(a)
        out.append(model.predict([np.repeat(a, len(B), 0)[..., None], np.tile(B, (n, 1))[..., None]],
                                 batch_size=512, verbose=0).reshape(n, len(B)))
    return np.vstack(out)

# ---------------------------------------------------------------- splits
def make_splits(MP):
    # Splits people (never individual ECGs) into groups, so one person's ECGs are never split
    # between training and testing. Fixed seeds make the split identical on every run.
    #   St. Jude: 48 locked test | 39 practice test | 156 train
    #   MIMIC:    200 locked test | 300 practice test | everyone else trains
    perm = np.random.default_rng(42).permutation(243)
    sj_test = np.sort(perm[:48])                                  # same locked 48 as v1 / v2
    dev = np.sort(perm[48:])
    p2 = np.random.default_rng(11).permutation(len(dev))
    sj_val, sj_train = np.sort(dev[p2[:39]]), np.sort(dev[p2[39:]])
    groups = mimic_groups(MP)
    order = np.random.default_rng(7).permutation(len(groups))
    mm_test = [groups[i] for i in order[:200]]
    mm_val = [groups[i] for i in order[200:500]]
    mm_train = [groups[i] for i in order[500:]]
    return sj_test, sj_val, sj_train, mm_test, mm_val, mm_train

# ---------------------------------------------------------------- training
def train_cnn(seed, sj_idx, mm_groups, C, W, MX, rounds, per_round):
    keras.utils.set_random_seed(seed); rng = np.random.default_rng(seed)
    cnn = build_siamese(); cnn.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy")
    for ep in range(rounds):
        t = time.time()
        # For the last 30% of rounds, take smaller steps (learning rate 3e-4 instead of 1e-3),
        # like fine adjustments once you're close to the answer.
        if ep == int(rounds * 0.7):                               # slow down for the last 30%
            cnn.optimizer.learning_rate.assign(3e-4)
        # Each round uses a random 3,000 MIMIC patients (not all 9,500) to keep rounds fast;
        # over 100 rounds the model still sees everyone many times.
        sub = [mm_groups[i] for i in rng.choice(len(mm_groups), min(per_round, len(mm_groups)), replace=False)]
        A, B, y = materialize(sample_pairs(rng, sj_idx, C, W, sub, neg=3), C, W, MX)
        sw = rng.random(len(y)) < 0.5
        A[sw], B[sw] = B[sw].copy(), A[sw].copy()
        for Z in (A, B):
            Z *= rng.uniform(0.8, 1.2, (len(Z), 1)); Z += rng.normal(0, 0.05, Z.shape)
        cnn.fit([A[..., None], B[..., None]], y, batch_size=64, epochs=1, verbose=0,
                class_weight={0: 1.0, 1: 3.0})
        if ep == 0 or (ep + 1) % 20 == 0:
            print(f"    round {ep + 1}/{rounds} ({time.time() - t:.0f}s)", flush=True)
    return cnn

def nn_pairs(rng, sj_idx, mm_groups, C, W, Vc, Vw, VM, use_sj=True, use_mm=True, mm_rounds=3):
    idx = []
    for r in range(mm_rounds if use_mm else 1):
        p = sample_pairs(rng, sj_idx, C, W, mm_groups if use_mm else [], neg=3)
        idx += [q for q in p if (q[0] == "mm" and use_mm) or (q[0] == "sj" and use_sj and r == 0)]
    Xa, Xb, y = materialize(idx, Vc, Vw, VM)
    return pair_feats(Xa, Xb), y

def fit_nn(model, X, y, lr, epochs):
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy")
    model.fit(X, y, epochs=epochs, batch_size=256, verbose=0,
              class_weight={0: 1.0, 1: float((y == 0).sum() / max((y == 1).sum(), 1))})

def train_nn(mode, sj_idx, mm_groups, C, W, Vc, Vw, VM, seed=0):
    keras.utils.set_random_seed(seed); rng = np.random.default_rng(seed)
    if mode == "joint":                                           # v2 recipe
        X, y = nn_pairs(rng, sj_idx, mm_groups, C, W, Vc, Vw, VM)
        nn = build_mlp(X.shape[1]); nn.get_layer("scaler").adapt(X); fit_nn(nn, X, y, 1e-3, 40)
    # "finetune" = two stages: learn from MIMIC hospital pairs first, then continue on St. Jude
    # hospital vs watch pairs with smaller steps (2e-4). Result: better on watch pairs,
    # worse on hospital pairs, so we chose "joint".
    else:                                                         # finetune: MIMIC, then St. Jude
        Xm, ym = nn_pairs(rng, sj_idx, mm_groups, C, W, Vc, Vw, VM, use_sj=False)
        nn = build_mlp(Xm.shape[1]); nn.get_layer("scaler").adapt(Xm); fit_nn(nn, Xm, ym, 1e-3, 30)
        Xs = np.vstack([nn_pairs(np.random.default_rng(seed + k), sj_idx, [], C, W, Vc, Vw, VM,
                                 use_mm=False)[0] for k in range(5)])
        ys = np.concatenate([nn_pairs(np.random.default_rng(seed + k), sj_idx, [], C, W, Vc, Vw, VM,
                                      use_mm=False)[1] for k in range(5)])
        fit_nn(nn, Xs, ys, 2e-4, 30)
    return nn

# ---------------------------------------------------------------- scoring
def score_sets(cnns, nns, sj_idx, mm_groups, C, W, MX, Vc, Vw, VM):
    """Returns score matrices for each CNN and each NN on one St. Jude set and one MIMIC set."""
    q = np.array([g[0] for g in mm_groups]); c = np.array([g[1] for g in mm_groups])
    out = {"cnn": [], "nn": {}}
    for m in cnns:
        out["cnn"].append(([cnn_matrix(m, C[sj_idx], W[sj_idx, k]) for k in range(3)],
                           cnn_matrix(m, MX[q], MX[c])))
    for name, m in nns.items():
        out["nn"][name] = ([nn_matrix(m, Vc[sj_idx], Vw[sj_idx, k]) for k in range(3)],
                           nn_matrix(m, VM[q], VM[c]))
    return out

def avg(mats):
    # Averages several CNNs' score grids into one (the "3 seeds averaged" ensemble).
    return [np.mean([m[k] for m in mats], 0) for k in range(len(mats[0]))]

def row(label, sj, mm):
    a1, t1 = auroc_top1(sj); a2, t2 = auroc_top1([mm])
    print(f"  {label:34s} {a1:7.3f} {t1:7.1%}   {a2:7.3f} {t2:7.1%}   {(a1 + a2) / 2:7.3f}")
    return (a1 + a2) / 2

def header(title):
    print(f"\n{title}\n  {'':34s} {'StJude':>7s} {'top1':>7s}   {'MIMIC':>7s} {'top1':>7s}   {'avg':>7s}")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sjlife", default=os.path.expanduser("~/SJLIFE-data"))
    ap.add_argument("--mimic", default=os.path.expanduser("~/mimic_leadI/mimic_leadI.npz"))
    ap.add_argument("--rounds", type=int, default=100, help="CNN training rounds")
    ap.add_argument("--per_round", type=int, default=3000, help="MIMIC patients sampled per CNN round")
    ap.add_argument("--seeds", type=int, default=3, help="how many CNNs to train and average")
    ap.add_argument("--final", action="store_true", help="train on train+practice, score the LOCKED test once")
    ap.add_argument("--nn", choices=["joint", "finetune"], default="finetune", help="feature net recipe (final mode)")
    ap.add_argument("--w", type=float, default=0.5, help="CNN weight in the ensemble (final mode)")
    ap.add_argument("--out", default=os.path.join(HERE, "experiments"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()

    print("loading data ...", flush=True)
    C, W = load_sjlife(a.sjlife); MX, MP = load_mimic(a.mimic)
    sj_test, sj_val, sj_train, mm_test, mm_val, mm_train = make_splits(MP)
    if a.final:                                                   # practice set joins training
        sj_train = np.sort(np.concatenate([sj_train, sj_val])); mm_train = mm_train + mm_val
        sj_eval, mm_eval, label = sj_test, mm_test, "LOCKED TEST"
    else:
        sj_eval, mm_eval, label = sj_val, mm_val, "PRACTICE TEST"
    print(f"  {len(MX):,} MIMIC ECGs | St. Jude train {len(sj_train)} | MIMIC train {len(mm_train):,} patients",
          flush=True)
    print("computing heartbeat features ...", flush=True)
    Vc = np.array([beat_vector(x) for x in C]); Vw = np.array([[beat_vector(x) for x in w] for w in W])
    VM = np.array([beat_vector(x) for x in MX])

    nn_modes = [a.nn] if a.final else ["joint", "finetune"]
    nns = {}
    for mode in nn_modes:
        print(f"training feature net ({mode}) ...", flush=True)
        nns[mode] = train_nn(mode, sj_train, mm_train, C, W, Vc, Vw, VM)
    cnns = []
    for s in range(a.seeds):
        print(f"training CNN {s + 1}/{a.seeds} ...", flush=True)
        cnns.append(train_cnn(100 + s, sj_train, mm_train, C, W, MX, a.rounds, a.per_round))

    print("scoring ...", flush=True)
    S = score_sets(cnns, nns, sj_eval, mm_eval, C, W, MX, Vc, Vw, VM)
    cnn_sj1, cnn_mm1 = S["cnn"][0]
    cnn_sj = avg([s[0] for s in S["cnn"]]); cnn_mm = np.mean([s[1] for s in S["cnn"]], 0)

    res = {}
    header(f"=== {label}: single models ===")
    res["cnn_1"] = row("CNN, 1 seed", cnn_sj1, cnn_mm1)
    res[f"cnn_{a.seeds}"] = row(f"CNN, average of {a.seeds} seeds", cnn_sj, cnn_mm)
    for mode, (nsj, nmm) in S["nn"].items():
        res[f"nn_{mode}"] = row(f"feature net ({mode})", nsj, nmm)

    header(f"=== {label}: ensembles (w = CNN weight) ===")
    best = (None, -1)
    for mode, (nsj, nmm) in S["nn"].items():
        for w in (W_GRID if not a.final else [a.w]):
            sj = [w * p + (1 - w) * q for p, q in zip(cnn_sj, nsj)]
            v = row(f"{a.seeds} CNNs + {mode} net, w={w:.1f}", sj, w * cnn_mm + (1 - w) * nmm)
            res[f"ens_{mode}_w{w:.1f}"] = v
            if v > best[1]:
                best = ((mode, w), v)

    tag = "final" if a.final else "explore"
    for i, m in enumerate(cnns):
        m.save(os.path.join(a.out, f"{tag}_siamese_cnn_{i + 1}.h5"))
    for mode, m in nns.items():
        m.save(os.path.join(a.out, f"{tag}_feature_nn_{mode}.h5"))
    json.dump({"mode": tag, "scores_avg_auroc": res, "best": {"nn": best[0][0], "w": best[0][1]},
               "seeds": a.seeds, "rounds": a.rounds, "mimic_ecgs": int(len(MX))},
              open(os.path.join(a.out, f"{tag}_results.json"), "w"), indent=1)
    print(f"\nbest on {label.lower()}: {best[0][0]} feature net, CNN weight {best[0][1]:.1f} (avg AUROC {best[1]:.3f})")
    if not a.final:
        print(f"next: python experiment.py --final --seeds {a.seeds} --nn {best[0][0]} --w {best[0][1]:.1f}")
    print(f"saved to {a.out}  ({(time.time() - t0) / 60:.1f} min)")

if __name__ == "__main__":
    main()
