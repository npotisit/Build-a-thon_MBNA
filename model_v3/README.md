# ECG Patient Matching Model (v3, submission model)

Scores how likely two Lead I ECGs come from the same patient. Built for the 2026 Wake Forest School of Medicine Build-A-Thon (ECG similarity and patient matching).

## What changed from v2

| | v2 | v3 |
|---|---|---|
| MIMIC-IV-ECG training data | 3,000 patients | **10,000 patients (37,237 ECGs)** |
| Siamese CNN | 1 model | **3 models trained from different random starts, averaged** |
| Ensemble weight | 50% CNN / 50% feature net | **80% CNN / 20% feature net** |
| Long recordings (30 s Apple Watch) | middle 10 s | middle 10 s, after skipping the first 1.5 s (tested against first, last, and 3 window average) |

Every choice above was made on a separate practice set (39 St. Jude + 300 MIMIC patients), not on the locked test.

## Results (locked test patients, never used for training or tuning)

| Test | v1 AUROC | v2 AUROC | **v3 AUROC** | v3 top 1 |
|---|---|---|---|---|
| St. Jude, hospital vs Apple Watch (48 patients) | 0.948 | 0.969 | **0.984** | 79.2% |
| MIMIC, hospital vs hospital (200 patients) | 0.889 | 0.974 | **0.981** | 68.5% |

The St. Jude row uses exactly the scoring pipeline in `score_pair.py` (middle 10 s of each 30 s watch ECG). The same 48 patients were used for v1, v2 and v3. The MIMIC test group is a fresh set of 200 held-out patients drawn from the larger v3 download, so that row is close to, but not exactly, the same patients as v2's.

AUROC: 0.5 = random guessing, 1.0 = perfect ranking of true matches above non matches. Top 1 = how often the model's first choice among all test patients is the correct person (random chance is about 2% for 48 patients). Single training run per version; the St. Jude test set is small, so small differences are within noise.

## Files

| File | Purpose |
|---|---|
| `siamese_cnn_1.h5`, `siamese_cnn_2.h5`, `siamese_cnn_3.h5` | Three Siamese 1D CNNs that compare the full 10 s signals; their scores are averaged |
| `feature_nn.h5` | Neural net that compares heartbeat summaries (median beat + 6 measurements) |
| `config.json` | Which files to load, the CNN weight (0.8) and the windowing method (middle) |
| `prepare.py` | Lead I preprocessing, identical in training and scoring |
| `score_pair.py` | Example script: loads everything, prepares two ECGs, returns one score |
| `results.json` | Full test results, including the windowing comparison |

## How to use

Demo on the public St. Jude data:

    python score_pair.py --demo path/to/SJLIFE-data

In your own code (from this folder):

    from score_pair import MatchModel
    model = MatchModel()                           # reads config.json, loads 3 CNNs + feature net
    score = model.score(ecg_a, fs_a, ecg_b, fs_b)  # float in [0, 1]

**Input:** each ECG is a 1D array of raw Lead I samples (any units, any length) plus its sampling rate in Hz.
**Output:** one score from 0 to 1. Higher means stronger evidence of the same patient. No threshold is needed. No custom layers; all .h5 files load with `keras.models.load_model(path, compile=False)`.

## Resampling and windowing (handled inside `score_pair.py` and `prepare.py`)

1. **Windowing:** recordings of about 10 s or less are used as they are. Recordings over 20 s (such as 30 s Apple Watch ECGs) have the first 1.5 s removed, then the middle 10 s is used. Recordings between 11 and 20 s use the middle 10 s.
2. **Resampling:** every window is resampled to 250 Hz (2,500 samples).
3. **Filtering:** band pass 0.5 to 40 Hz.
4. **Scaling:** subtract the median, divide by a robust spread (MAD), clip extremes. Shorter recordings are zero padded to 2,500 samples.

**Final score** = 0.8 x (average of the 3 CNN scores) + 0.2 x (feature net score).

## Requirements

Python 3.13, tensorflow 2.21.0, keras 3.15.1, numpy, scipy, h5py (exact versions in `../requirements.txt`). Runs on CPU.
