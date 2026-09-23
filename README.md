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

New versions go in their own folder (`model_v2/`, etc.) so earlier results stay reproducible.

## Datasets

| Dataset | Use | Link |
|---|---|---|
| MIMIC-IV-ECG v1.0 | Large scale pretraining (about 800,000 ECGs, 160,000 patients) | [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) |
| St. Jude (SJLIFE) paired clinical + Apple Watch ECGs | Cross device training and validation (243 patients) | [GitHub](https://github.com/akbilgic/SJLIFE-Paired-Clinical-and-Apple-Watch-ECG-Repository) |

Raw data is not stored in this repo. Download it from the links above.

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
