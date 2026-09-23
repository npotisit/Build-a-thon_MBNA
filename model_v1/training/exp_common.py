import numpy as np, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from prepare import beat_vector

rng = np.random.default_rng(42)
perm = rng.permutation(243)
TEST = np.sort(perm[:48]); DEV = np.sort(perm[48:])
kf = KFold(5, shuffle=True, random_state=42)

def vecs(c, w):
    V_c = np.array([beat_vector(x) for x in c])
    V_w = np.array([[beat_vector(x) for x in ww] for ww in w])
    return V_c, V_w

def eval_matrix(score_fn, idx, Xc, Xw):
    """AUROC over every clinical-vs-watch pair among patients idx (all 3 watch windows)."""
    ys, ss = [], []
    for k in range(Xw.shape[1]):
        S = score_fn(Xc[idx], Xw[idx, k])          # S[i, j] = score(clinical i, watch j)
        ys.append(np.eye(len(idx)).ravel()); ss.append(S.ravel())
    y, s = np.concatenate(ys), np.concatenate(ss)
    S0 = score_fn(Xc[idx], Xw[idx, 0])
    top1 = np.mean(S0.argmax(1) == np.arange(len(idx)))
    return roc_auc_score(y, s), top1

def cos_shape(A, B):
    a, b = A[:, :175], B[:, :175]
    a = (a - a.mean(1, keepdims=True)); a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    b = (b - b.mean(1, keepdims=True)); b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    return a @ b.T

def pair_feats(a, b):
    return np.hstack([np.abs(a - b), a * b])

def make_pairs(idx, Vc, Vw, neg_per=6, rng=np.random.default_rng(0)):
    X, y = [], []
    for i in idx:
        others = idx[idx != i]
        for k in range(Vw.shape[1]):
            X.append(pair_feats(Vc[i], Vw[i, k])); y.append(1)
            for j in rng.choice(others, neg_per // Vw.shape[1] * 1 + 1, replace=False):
                X.append(pair_feats(Vc[i], Vw[j, rng.integers(Vw.shape[1])])); y.append(0)
    return np.array(X), np.array(y)
