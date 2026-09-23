import numpy as np, warnings, pickle; warnings.filterwarnings("ignore")
from exp_common import *
import exp_nn, exp_cnn
Vc, Vw = exp_nn.Vc, exp_nn.Vw; C, W = exp_cnn.C, exp_cnn.W
out = []
for f, (tr, va) in enumerate(kf.split(DEV)):
    X, y = make_pairs(DEV[tr], Vc, Vw)
    nn = exp_nn.matrix_scorer(exp_nn.fit_mlp(X, y, (16,))[0])
    cnn_m = exp_cnn.train(DEV[tr], epochs=100)
    cnn = exp_cnn.scorer(cnn_m)
    v = DEV[va]
    rec = {"idx": v}
    for name, fn, A, B in [("base", cos_shape, Vc, Vw), ("nn", nn, Vc, Vw), ("cnn", cnn, C, W)]:
        rec[name] = [fn(A[v], B[v, k]) for k in range(3)]
    out.append(rec); pickle.dump(out, open("cv_scores.pkl", "wb"))
    print("fold", f, "done", flush=True)
