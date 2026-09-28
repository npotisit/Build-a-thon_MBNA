# ECG Patient Matching (Build-A-Thon 2026, Team MBNA)

Given two Lead I ECGs, the model returns a score from 0 to 1 for how likely they come from the same person. Higher means more likely the same patient. It works across devices (hospital ECG vs Apple Watch) and between hospital ECGs.

**Submission model: `model_v2/`** (`model_v1/` is kept for reference).

## Quick start

```
pip install -r requirements.txt
git clone --depth 1 https://github.com/akbilgic/SJLIFE-Paired-Clinical-and-Apple-Watch-ECG-Repository.git SJLIFE-data
python model_v2/score_pair.py --demo SJLIFE-data
```

Expected output (last digit may vary slightly by machine):

```
same patient (132 vs 132):       0.997
different patients (132 vs 140): 0.027
```

## Using the model in your own code

```python
import sys; sys.path.insert(0, "model_v2")
from score_pair import MatchModel

model = MatchModel("model_v2")               # loads feature_nn.h5 and siamese_cnn.h5
score = model.score(ecg_a, fs_a, ecg_b, fs_b)  # float in [0, 1]
```

`ecg_a`, `ecg_b`: 1D arrays of raw Lead I samples, any units, up to 10 s (longer recordings are center cropped).
`fs_a`, `fs_b`: sampling rate in Hz (for example 500 for hospital ECGs, 512 for Apple Watch).

## Files

| File | What it is |
|---|---|
| `model_v2/feature_nn.h5` | Neural net that compares summarized heartbeats |
| `model_v2/siamese_cnn.h5` | Siamese 1D CNN that compares the full 10 s signals |
| `model_v2/prepare.py` | Lead I preprocessing (required by the models) |
| `model_v2/score_pair.py` | Example inference script: load models, prepare two ECGs, return a score |
| `model_v2/results.json` | Test results for v2 |
| `training/download_mimic_leadI.py` | Downloads a MIMIC-IV-ECG subset and keeps Lead I only |
| `training/train_with_mimic.py` | Trains v2 on SJLIFE + MIMIC and compares it to v1 |

No custom layers. Both .h5 files load with `keras.models.load_model(path, compile=False)`.

## Lead I preprocessing (`prepare.py`)

The same steps are applied in training and scoring:

1. Fill missing values by linear interpolation.
2. Resample to 250 Hz.
3. Band pass filter 0.5 to 40 Hz (removes baseline drift and high frequency noise).
4. Center crop or zero pad to 10 s (2,500 samples).
5. Subtract the median and divide by a robust spread (MAD), so only the shape matters, not the device's voltage scale. Clip extreme values.

For the feature net, `beat_vector` also detects R peaks, builds the median heartbeat (175 samples), and adds 6 summary values (heart rate, heart rate variability, peak heights, beat consistency).

## How it works

Two models score each pair and the final score is their average:

* **Feature net:** compares the two median heartbeats (absolute difference and product of the feature vectors) with a small dense network.
* **Siamese CNN:** runs both 10 s signals through the same convolutional encoder, then compares the two embeddings.

## Training data

* **SJLIFE paired clinical and Apple Watch ECGs** (243 patients). 48 patients are held out for testing (fixed seed 42) and never used in training.
* **MIMIC-IV-ECG**, Lead I only: 3,000 randomly sampled patients with 2+ ECGs (11,128 ECGs). 200 patients held out for testing.

## Results (held-out patients only)

| Test | v1 AUROC | v2 AUROC | v2 top 1 |
|---|---|---|---|
| SJLIFE, hospital vs Apple Watch (48 patients) | 0.948 | **0.969** | 64.6% |
| MIMIC, hospital vs hospital (200 patients) | 0.889 | **0.974** | 60.5% |

Top 1 = how often the model's first choice among all test patients is the correct person (random chance is about 2% for 48 patients).

Caveats: single training run; the SJLIFE test set is small (48 patients), so small differences are within noise.

## Reproducing v2

MIMIC data is not included in this repo (size and data use terms). To rebuild it:

```
cd training
python download_mimic_leadI.py --patients 3000
python train_with_mimic.py
```

Training took about 10 minutes on a laptop CPU. New models are saved to `model_retrained/`.

## Environment

Tested on Windows, Python 3.13, TensorFlow 2.21 / Keras 3.15, CPU only. See `requirements.txt`.
