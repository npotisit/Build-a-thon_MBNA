import numpy as np, pandas as pd
from prepare import prepare_ecg, FS

def load_sjlife(root="data", norm="robust", band=(0.5, 40.0)):
    meta = pd.read_csv(f"{root}/shared_paired_data_243.csv")
    meta["n"] = meta.apple_loc_ECG_243.str.extract(r"(\d+)\.npy").astype(int)
    clin, watch = [], []
    for n in meta.n:
        c = np.load(f"{root}/ClinicalECGs_full_243/clinical_ecg_{n}.npy")[0, 0]   # lead I, 500 Hz, 10 s
        a = np.load(f"{root}/AppleECGs_full_243/apple_ecg_{n}.npy")               # 512 Hz, 30 s
        clin.append(prepare_ecg(c, 500, norm, band))
        a = a[int(1.5 * 512):]                                                      # finger contact spike
        L = 10 * 512
        starts = [0, L, len(a) - L]                                                 # three 10 s windows
        watch.append([prepare_ecg(a[s:s + L], 512, norm, band) for s in starts])
    return meta, np.array(clin), np.array(watch)       # (243, 2500), (243, 3, 2500)
