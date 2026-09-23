import numpy as np, warnings, json; warnings.filterwarnings("ignore")
from exp_common import *
import exp_nn, exp_cnn
Vc, Vw = exp_nn.Vc, exp_nn.Vw; C, W = exp_cnn.C, exp_cnn.W
X, y = make_pairs(DEV, Vc, Vw)
_, nn_model = exp_nn.fit_mlp(X, y, (16,))
cnn_model = exp_cnn.train(DEV, epochs=100)
nn_model.save("feature_nn.h5"); cnn_model.save("siamese_cnn.h5")
nn = exp_nn.matrix_scorer(lambda P: nn_model.predict(P, batch_size=4096, verbose=0).ravel())
cnn = exp_cnn.scorer(cnn_model)
ens_fn = lambda k: 0.5 * nn(Vc[TEST], Vw[TEST, k]) + 0.5 * cnn(C[TEST], W[TEST, k])
res = {}
for name, fn, A, B in [("baseline", cos_shape, Vc, Vw), ("feature_nn", nn, Vc, Vw), ("siamese_cnn", cnn, C, W)]:
    res[name] = eval_matrix(fn, TEST, A, B)
S = {k: ens_fn(k) for k in range(3)}
res["ensemble"] = eval_matrix(lambda A, B, _c=[0]: None, TEST, Vc, Vw) if False else None
n = len(TEST)
res["ensemble"] = (roc_auc_score(np.tile(np.eye(n).ravel(), 3), np.concatenate([S[k].ravel() for k in range(3)])),
                   float(np.mean(S[0].argmax(1) == np.arange(n))))
np.save("test_ens_S0.npy", S[0])
json.dump({k: [float(v) for v in r] for k, r in res.items()}, open("test_results.json", "w"), indent=1)
print(res)
