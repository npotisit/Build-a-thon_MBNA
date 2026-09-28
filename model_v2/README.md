# ECG Patient Matching Model (v2, submission model)

Scores how likely two Lead I ECGs come from the same patient. Built for the 2026 Wake Forest School of Medicine Build-A-Thon (ECG similarity and patient matching).

Same design as v1 (feature neural net + Siamese CNN ensemble), retrained with 3,000 MIMIC-IV-ECG patients (11,128 ECGs) added to the St. Jude training data.

## Results (held-out patients only)

| Test | Model | v1 AUROC | v2 AUROC | v2 top 1 |
|---|---|---|---|---|
| St. Jude, hospital vs Apple Watch (48 patients) | Feature neural net | 0.938 | 0.921 | 54.2% |
| | Signal neural net (Siamese CNN) | 0.929 | 0.966 | 60.4% |
| | **Ensemble** | 0.948 | **0.969** | 64.6% |
| MIMIC, hospital vs hospital (200 patients) | **Ensemble** | 0.889 | **0.974** | 60.5% |

AUROC: 0.5 = random guessing, 1.0 = perfect ranking of true matches above non matches. Top 1 = how often the model's first choice among all test patients is the correct person.

The St. Jude test patients are the same 48 used for v1 (seed 42), so the comparison is direct. Single training run; small differences are within noise.

## Files

| File | Purpose |
|---|---|
| `feature_nn.h5` | Neural net that compares heartbeat measurements between two ECGs |
| `siamese_cnn.h5` | Siamese 1D CNN that compares the raw 10 second signals |
| `prepare.py` | Signal preparation, used for both training and scoring (unchanged from v1) |
| `score_pair.py` | Loads both models and returns one similarity score for a pair |
| `results.json` | Full test results for v1 and v2 |

Training scripts are in [`../training/`](../training).

## Requirements

Python 3.13, tensorflow 2.21, keras 3.15, numpy, scipy, h5py (exact versions in `../requirements.txt`)

    pip install tensorflow==2.21.0 keras==3.15.1 numpy scipy h5py

## How to use

Demo on the public St. Jude data:

    python score_pair.py --demo path/to/SJLIFE-data

In your own code (run from this folder):

    from score_pair import MatchModel
    model = MatchModel()
    score = model.score(ecg_a, fs_a, ecg_b, fs_b)

**Input:** each ECG is a 1D array of Lead I samples (any units, up to 10 seconds; longer is center cropped) plus its sampling rate in Hz.
**Output:** one score from 0 to 1. Higher means stronger evidence the two ECGs belong to the same patient. No threshold is needed. No custom layers; both .h5 files load with `keras.models.load_model(path, compile=False)`.
