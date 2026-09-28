# Build-A-Thon 2026: ECG Similarity and Patient Matching

Team repository for the 2026 Wake Forest School of Medicine Build-A-Thon, hosted by the Center for Artificial Intelligence Research (CAIR).

## The challenge

Build a model that takes two Lead I ECG recordings and returns a similarity score showing how likely they belong to the same patient. The final evaluation uses a Wake Forest holdout set of hospital ECGs and Apple Watch ECGs from patients the model has never seen, scored with AUROC.

**Model rules:** Lead I only, 250 Hz, up to 10 seconds (2,500 samples). Submission is one or more `.h5` model files, an example script and a README.

## Team

| Name | Role |
|---|---|
| Brian Mejia-Lopez | |
| | |
| | |
| | |

## Repository layout

| Folder | Contents | Best result (locked test AUROC) |
|---|---|---|
| [`model_v1/`](model_v1) | Feature neural net + Siamese CNN ensemble, trained on St. Jude data only | 0.948 |
| [`model_v2/`](model_v2) | **Submission model.** Same design, trained on St. Jude + 3,000 MIMIC-IV-ECG patients | **0.969** |
| [`training/`](training) | Scripts to download the MIMIC subset and train v2 | |

New versions go in their own folder (`model_v2/`, etc.) so earlier results stay reproducible.

## Quick start (submission model)

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

Using it in your own code:

```python
import sys; sys.path.insert(0, "model_v2")
from score_pair import MatchModel

model = MatchModel("model_v2")               # loads feature_nn.h5 and siamese_cnn.h5
score = model.score(ecg_a, fs_a, ecg_b, fs_b)  # float in [0, 1]
```

`ecg_a`, `ecg_b`: 1D arrays of raw Lead I samples, any units, up to 10 s (longer recordings are center cropped). `fs_a`, `fs_b`: sampling rate in Hz (for example 500 for hospital ECGs, 512 for Apple Watch). Higher score means stronger evidence of the same patient. No custom layers; both .h5 files load with `keras.models.load_model(path, compile=False)`.

## Results (held-out patients only)

| Test | v1 AUROC | v2 AUROC | v1 top 1 | v2 top 1 |
|---|---|---|---|---|
| St. Jude, hospital vs Apple Watch (48 patients) | 0.948 | **0.969** | 62.5% | 64.6% |
| MIMIC, hospital vs hospital (200 patients) | 0.889 | **0.974** | 33.0% | 60.5% |

AUROC: 0.5 = random guessing, 1.0 = perfect ranking of true matches above non matches. Top 1 = how often the model's first choice among all test patients is the correct person (random chance is about 2% for 48 patients).

Caveats: single training run; the St. Jude test set is small (48 patients), so small differences are within noise.

## How it works

Two models score each pair and the final score is their average:

* **Feature neural net:** builds the median heartbeat of each ECG plus 6 summary values (heart rate, variability, peak heights, beat consistency) and compares them with a small dense network.
* **Siamese CNN:** runs both 10 s signals through the same convolutional encoder, then compares the two embeddings.

**Lead I preprocessing (`prepare.py`)**, applied identically in training and scoring: fill gaps, resample to 250 Hz, band pass 0.5 to 40 Hz, center crop or zero pad to 10 s (2,500 samples), then subtract the median and scale by a robust spread so only the shape matters, not the device's voltage.

## Datasets

| Dataset | Use | Link |
|---|---|---|
| MIMIC-IV-ECG v1.0 | Large scale training (about 800,000 ECGs, 160,000 patients); v2 uses Lead I from 3,000 patients with 2+ ECGs (11,128 ECGs) | [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) |
| St. Jude (SJLIFE) paired clinical + Apple Watch ECGs | Cross device training and validation (243 patients; 48 locked for testing, seed 42) | [GitHub](https://github.com/akbilgic/SJLIFE-Paired-Clinical-and-Apple-Watch-ECG-Repository) |

Raw data is not stored in this repo. Download it from the links above.

## Reproducing v2

```
cd training
python download_mimic_leadI.py --patients 3000
python train_with_mimic.py
```

Training took about 10 minutes on a laptop CPU. New models are saved to `model_retrained/`, so the submission files are never overwritten.

## Environment

Tested on Windows, Python 3.13, TensorFlow 2.21 / Keras 3.15, CPU only. See `requirements.txt`.

## Timeline

| Date | Milestone |
|---|---|
| Sep 22, 2026 | Opening event |
| Sep 28 | Data pipeline and baseline done |
| Oct 4 | First submission ready model |
| Oct 10 | Improvements finalized |
| Oct 13, 2026 | Closing event and submission (Bowman Gray Center, Rooms 5206 to 5207, 12:00 to 4:00 p.m.) |

## Contact

Challenge contact: Luke Patterson, Luke.Patterson@AdvocateHealth.org
