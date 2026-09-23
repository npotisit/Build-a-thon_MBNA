import numpy as np, warnings, sys, time; warnings.filterwarnings("ignore")
from exp_common import *
import tensorflow as tf, keras
from keras import layers
d = np.load("vec_robust.npz"); C, W = d["c"], d["w"]          # prepared signals (243,2500) / (243,3,2500)

def build_encoder(emb=64):
    inp = keras.Input((2500, 1))
    x = inp
    for f, k in [(16, 9), (32, 7), (32, 7), (64, 5), (64, 5), (128, 3)]:
        x = layers.Conv1D(f, k, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(emb)(x)
    return keras.Model(inp, x, name="encoder")

def build_siamese(enc):
    a, b = keras.Input((2500, 1), name="ecg_a"), keras.Input((2500, 1), name="ecg_b")
    ea, eb = enc(a), enc(b)
    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([ea, eb]) if False else layers.Subtract()([ea, eb])
    diff = layers.Multiply()([diff, diff])                      # squared difference: symmetric, no custom code
    prod = layers.Multiply()([ea, eb])
    h = layers.Concatenate()([diff, prod])
    h = layers.Dense(32, activation="relu")(h)
    out = layers.Dense(1, activation="sigmoid", name="score")(h)
    return keras.Model([a, b], out)

def sample(idx, rng, neg_per=3, aug=True):
    A, B, y = [], [], []
    for i in idx:
        for _ in range(2):
            k = rng.integers(3)
            A.append(C[i]); B.append(W[i, k]); y.append(1)
            for j in rng.choice(idx[idx != i], neg_per, replace=False):
                A.append(C[i]); B.append(W[j, rng.integers(3)]); y.append(0)
    A, B, y = np.array(A), np.array(B), np.array(y, dtype="float32")
    sw = rng.random(len(y)) < 0.5                               # random order so it can't learn "left = clinical"
    A[sw], B[sw] = B[sw].copy(), A[sw].copy()
    if aug:
        for X in (A, B):
            X *= rng.uniform(0.8, 1.2, (len(X), 1))
            X += rng.normal(0, 0.05, X.shape)
    return A[..., None], B[..., None], y

def train(idx, epochs=25, seed=0):
    keras.utils.set_random_seed(seed); rng = np.random.default_rng(seed)
    enc = build_encoder(); model = build_siamese(enc)
    model.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy")
    for ep in range(epochs):
        A, B, y = sample(idx, rng)
        model.fit([A, B], y, batch_size=64, epochs=1, verbose=0, class_weight={0: 1., 1: 3.})
    return model

def scorer(model):
    def f(Xa, Xb):
        n, m = len(Xa), len(Xb)
        A = np.repeat(Xa, m, 0)[..., None]; B = np.tile(Xb, (n, 1))[..., None]
        return model.predict([A, B], batch_size=1024, verbose=0).reshape(n, m)
    return f

if __name__ == "__main__":
    folds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    res = []
    for f, (tr, va) in enumerate(kf.split(DEV)):
        if f >= folds: break
        t = time.time(); m = train(DEV[tr])
        r = eval_matrix(scorer(m), DEV[va], C, W); res.append(r)
        print(f"fold {f}: AUROC {r[0]:.3f} top1 {r[1]:.3f}  ({time.time()-t:.0f}s)", flush=True)
    print("CNN CV mean", np.mean(res, 0))
