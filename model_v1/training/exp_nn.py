import numpy as np, warnings; warnings.filterwarnings("ignore")
from exp_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf, keras
from keras import layers
tf.random.set_seed(0); keras.utils.set_random_seed(0)
d = np.load("vec_robust.npz"); Vc, Vw = d["Vc"], d["Vw"]

def matrix_scorer(predict):
    def f(A, B):
        n, m = len(A), len(B)
        P = pair_feats(np.repeat(A, m, 0), np.tile(B, (n, 1)))
        return predict(P).reshape(n, m)
    return f

def fit_logit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return lambda P: m.decision_function(sc.transform(P))

def build_mlp(dim, hidden=(32, 16), drop=0.3):
    inp = keras.Input((dim,))
    x = layers.Normalization(name="scaler")(inp)
    for h in hidden:
        x = layers.Dense(h, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out); return m

def fit_mlp(Xtr, ytr, hidden=(32, 16), drop=0.3, epochs=40):
    m = build_mlp(Xtr.shape[1], hidden, drop); m.get_layer("scaler").adapt(Xtr)
    m.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy")
    w = {0: 1.0, 1: (ytr == 0).sum() / (ytr == 1).sum()}
    m.fit(Xtr, ytr, epochs=epochs, batch_size=128, verbose=0, class_weight=w)
    return (lambda P: m.predict(P, batch_size=4096, verbose=0).ravel()), m

if __name__ == "__main__":
    configs = {"logistic (linear)": lambda X, y: fit_logit(X, y)}
    for h in [(16,), (32, 16), (64, 32)]:
        configs[f"NN {h}"] = (lambda h: lambda X, y: fit_mlp(X, y, h)[0])(h)
    for name, fitter in configs.items():
        res = []
        for tr, va in kf.split(DEV):
            X, y = make_pairs(DEV[tr], Vc, Vw)
            res.append(eval_matrix(matrix_scorer(fitter(X, y)), DEV[va], Vc, Vw))
        print(f"{name:20s} CV AUROC {np.mean(res,0)[0]:.3f} (sd {np.std(res,0)[0]:.3f})  top1 {np.mean(res,0)[1]:.3f}", flush=True)
