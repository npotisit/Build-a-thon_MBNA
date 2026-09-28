"""Download a subset of MIMIC-IV-ECG and keep only Lead I.

Picks patients who have 2+ ECGs (needed to learn same-patient matching),
downloads their recordings one at a time from PhysioNet, keeps Lead I,
and throws the rest away. Nothing big is stored on disk.

Usage:
    pip install wfdb pandas requests
    python download_mimic_leadI.py --patients 2000 --out C:/Users/brian/mimic_leadI

Safe to stop and rerun: it skips anything already downloaded.
Output: mimic_leadI.npz with
    signals    (n_ecgs, 5000) float16, Lead I, 500 Hz, 10 s, in mV
    subject_id (n_ecgs,)      patient id (same id = same person)
    study_id   (n_ecgs,)      ECG id
    ecg_time   (n_ecgs,)      when the ECG was taken
"""
import os, argparse, tempfile, time
import numpy as np, pandas as pd, requests, wfdb
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"

def get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))

def lead_one(path):
    """path like files/p1000/p10000032/s40689238/40689238 -> Lead I array or None."""
    with tempfile.TemporaryDirectory() as d:
        name = os.path.basename(path)
        for ext in (".hea", ".dat"):
            with open(os.path.join(d, name + ext), "wb") as f:
                f.write(get(BASE + path + ext))
        rec = wfdb.rdrecord(os.path.join(d, name), channel_names=["I"])
        if rec.p_signal is None or rec.p_signal.shape[1] == 0:
            return None
        x = rec.p_signal[:, 0]
        if rec.fs != 500 or len(x) != 5000:
            return None
        return np.nan_to_num(x).astype(np.float16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=2000, help="how many patients to sample")
    ap.add_argument("--max_per_patient", type=int, default=5, help="cap ECGs per patient")
    ap.add_argument("--out", default=os.path.expanduser("~/mimic_leadI"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)

    rl_path = os.path.join(a.out, "record_list.csv")
    if not os.path.exists(rl_path) or os.path.getsize(rl_path) < 1000:
        print("downloading record list ...")
        data = get(BASE + "record_list.csv")          # download first, then save
        if not data.lstrip().startswith(b"subject_id"):
            raise SystemExit("PhysioNet did not return the record list. Open " + BASE +
                             " in your browser; you may need to log in and accept the data use agreement.")
        with open(rl_path, "wb") as f:
            f.write(data)
    rl = pd.read_csv(rl_path)

    counts = rl.subject_id.value_counts()
    eligible = counts[counts >= 2].index.values
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(eligible, size=min(a.patients, len(eligible)), replace=False)
    sub = (rl[rl.subject_id.isin(pick)]
           .sort_values(["subject_id", "ecg_time"])
           .groupby("subject_id").head(a.max_per_patient))
    print(f"{len(eligible):,} patients have 2+ ECGs; using {len(pick):,} patients, {len(sub):,} ECGs")

    todo = [r for r in sub.itertuples()
            if not os.path.exists(os.path.join(a.out, "parts", f"{r.study_id}.npy"))]
    print(f"{len(sub) - len(todo):,} already done, {len(todo):,} to download")

    def work(r):
        x = lead_one(r.path)
        np.save(os.path.join(a.out, "parts", f"{r.study_id}.npy"),
                x if x is not None else np.zeros(0, np.float16))
    failed = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception:
                failed += 1
            if i % 200 == 0 or i == len(futs):
                print(f"  {i:,}/{len(futs):,} done, {failed} failed")

    sig, keep = [], []
    for r in sub.itertuples():
        p = os.path.join(a.out, "parts", f"{r.study_id}.npy")
        if os.path.exists(p):
            x = np.load(p)
            if len(x) == 5000:
                sig.append(x); keep.append(r.Index)
    k = sub.loc[keep]
    # drop patients left with fewer than 2 good ECGs
    ok = k.subject_id.map(k.subject_id.value_counts()) >= 2
    sig = np.array(sig)[ok.values]; k = k[ok]
    out = os.path.join(a.out, "mimic_leadI.npz")
    np.savez_compressed(out, signals=sig, subject_id=k.subject_id.values,
                        study_id=k.study_id.values, ecg_time=k.ecg_time.astype(str).values)
    print(f"saved {out}: {len(sig):,} ECGs from {k.subject_id.nunique():,} patients")

if __name__ == "__main__":
    main()
