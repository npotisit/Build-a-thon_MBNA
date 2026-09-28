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
| [`model_v2/`](model_v2) | Same design, trained on St. Jude + 3,000 MIMIC-IV-ECG patients | 0.969 |
| [`model_v3/`](model_v3) | **Submission model.** 3 averaged CNNs + feature net (80/20), 10,000 MIMIC patients | **0.984** |
| [`training/`](training) | Scripts to download the MIMIC subset, train, and run experiments | |

New versions go in their own folder (`model_v2/`, etc.) so earlier results stay reproducible.

## Quick start (submission model)

```
pip install -r requirements.txt
git clone --depth 1 https://github.com/akbilgic/SJLIFE-Paired-Clinical-and-Apple-Watch-ECG-Repository.git SJLIFE-data
python model_v3/score_pair.py --demo SJLIFE-data
```

Expected output (last digit may vary slightly by machine):

```
same patient (132 vs 132):       1.000
different patients (132 vs 140): 0.050
```

Using it in your own code:

```python
import sys; sys.path.insert(0, "model_v3")
from score_pair import MatchModel

model = MatchModel("model_v3")               # reads config.json: 3 CNNs + feature net, CNN weight 0.8
score = model.score(ecg_a, fs_a, ecg_b, fs_b)  # float in [0, 1]
```

`ecg_a`, `ecg_b`: 1D arrays of raw Lead I samples, any units, any length. `fs_a`, `fs_b`: sampling rate in Hz (for example 500 for hospital ECGs, 512 for Apple Watch). Higher score means stronger evidence of the same patient. No custom layers; all .h5 files load with `keras.models.load_model(path, compile=False)`. Full details in [`model_v3/README.md`](model_v3/README.md).

## Results (held-out patients only)

| Test | v1 AUROC | v2 AUROC | **v3 AUROC** | v3 top 1 |
|---|---|---|---|---|
| St. Jude, hospital vs Apple Watch (48 patients) | 0.948 | 0.969 | **0.984** | 79.2% |
| MIMIC, hospital vs hospital (200 patients) | 0.889 | 0.974 | **0.981** | 68.5% |

AUROC: 0.5 = random guessing, 1.0 = perfect ranking of true matches above non matches. Top 1 = how often the model's first choice among all test patients is the correct person (random chance is about 2% for 48 patients).

All v3 settings (number of CNNs, ensemble weight, feature net recipe, windowing) were chosen on a separate practice set, not the locked test. Caveats: the St. Jude test set is small (48 patients), so small differences are within noise; the v3 MIMIC test is a fresh set of 200 held-out patients from the larger download.

## How it works

Two kinds of models score each pair (v3: final score = 0.8 x average of 3 CNNs + 0.2 x feature net):

* **Feature neural net:** builds the median heartbeat of each ECG (174 samples) plus 6 summary values (heart rate, variability, peak heights, beat consistency) and compares them with a small dense network.
* **Siamese CNN:** runs both 10 s signals through the same convolutional encoder, then compares the two embeddings. v3 averages three CNNs trained from different random starts.

**Windowing (`score_pair.py`):** recordings over 20 s (such as 30 s Apple Watch ECGs) skip the first 1.5 s, then use the middle 10 s. We compared this against the first, last, and an average of all three windows on the practice set; the middle window scored as well or better.

**Lead I preprocessing (`prepare.py`)**, applied identically in training and scoring: fill gaps, resample to 250 Hz, band pass 0.5 to 40 Hz, center crop or zero pad to 10 s (2,500 samples), then subtract the median and scale by a robust spread so only the shape matters, not the device's voltage.

## Datasets

| Dataset | Use | Link |
|---|---|---|
| MIMIC-IV-ECG v1.0 | Large scale training (about 800,000 ECGs, 160,000 patients); v3 uses Lead I from 10,000 patients with 2+ ECGs (37,237 ECGs) | [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) |
| St. Jude (SJLIFE) paired clinical + Apple Watch ECGs | Cross device training and validation (243 patients; 48 locked for testing, seed 42) | [GitHub](https://github.com/akbilgic/SJLIFE-Paired-Clinical-and-Apple-Watch-ECG-Repository) |

Raw data is not stored in this repo. Download it from the links above.

## Reproducing v3

```
cd training
python download_mimic_leadI.py --patients 10000
python experiment.py                                        # practice set: compares settings (about 1 hour)
python eval_windows.py --nn joint --w 0.8                   # practice set: compares windowing
python experiment.py --final --seeds 3 --nn joint --w 0.8   # trains v3, scores the locked test once
```

Models are saved to `training/experiments/`, so the submission files are never overwritten. `train_with_mimic.py` reproduces v2.

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
