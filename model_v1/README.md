# ECG Patient Matching Model (v1)

Scores how likely two Lead I ECGs come from the same patient. Built for the 2026 Wake Forest School of Medicine Build-A-Thon (ECG similarity and patient matching).

## Results (St. Jude paired data, hospital ECG vs Apple Watch)

| Model | Cross validation AUROC (195 patients) | Locked test AUROC (48 patients) |
|---|---|---|
| Baseline (beat shape similarity) | 0.857 | 0.803 |
| Feature neural net | 0.960 | 0.938 |
| Signal neural net (Siamese CNN) | 0.935 | 0.929 |
| **Ensemble (average of both nets)** | **0.969** | **0.948** |

AUROC: 0.5 = random guessing, 1.0 = perfect ranking of true matches above non matches.

## Files

| File | Purpose |
|---|---|
| `feature_nn.h5` | Neural net that compares heartbeat measurements between two ECGs |
| `siamese_cnn.h5` | Siamese 1D CNN that compares the raw 10 second signals |
| `prepare.py` | Signal preparation, used for both training and scoring |
| `score_pair.py` | Loads both models and returns one similarity score for a pair |
| `training/` | Scripts used to train and evaluate the models |

## Requirements

Python 3.11, tensorflow 2.21, keras 3.15, numpy, scipy

    pip install tensorflow==2.21 keras==3.15 numpy scipy

## How to use

    from score_pair import MatchModel
    model = MatchModel()
    score = model.score(ecg_a, fs_a, ecg_b, fs_b)

**Input:** each ECG is a 1D array of Lead I samples (any units) plus its sampling rate in Hz.
**Output:** one score from 0 to 1. Higher means stronger evidence the two ECGs belong to the same patient. The score is symmetric, so swapping the two inputs gives the same result. No threshold is needed.

Demo on the public St. Jude data:

    python score_pair.py --demo path/to/SJLIFE-Paired-Clinical-and-Apple-Watch-ECG-Repository

## Signal preparation (prepare.py)

1. Fill missing values by interpolation (an all missing signal becomes flat zeros)
2. Resample to 250 Hz
3. Bandpass filter 0.5 to 40 Hz
4. Center crop to 10 seconds (2,500 samples)
5. Scale each recording by its own median and spread, so units and device do not matter
6. Recordings shorter than 10 seconds are zero padded. Accuracy drops on very short clips (under 5 seconds)

## Training

Scripts are in `training/`. Before running, copy `prepare.py` into that folder and clone the St. Jude repository into a folder named `data`. Patients were split once: 48 locked away as a final test, 195 used for 5-fold cross validation. `final.py` retrains both models and rescores the test set.

## Next steps

Pretrain the signal net on MIMIC-IV-ECG, improve short recording handling.
