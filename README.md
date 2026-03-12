# Multi-Task ADMET Prediction

[![Status: In Progress](https://img.shields.io/badge/status-in%20progress-yellow.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

> Part of a connected research arc: [AMP Benchmark](https://github.com/SauravKulkarni3999/AMP-Activity-Toxicity-Baseline-Benchmark) · [GraphDTA-3D](https://github.com/SauravKulkarni3999/GraphDTA-3D) · [Cyclic AMP Pipeline — Chiral 2026](https://github.com/SauravKulkarni3999/chiral-amp-pipeline)

---

## Overview

A multi-task deep learning framework for simultaneous prediction of ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties from molecular structures.

The central motivation: ADMET properties are not independent. A molecule that crosses the blood-brain barrier likely shares physicochemical features with molecules that inhibit CYP3A4. A shared encoder that learns across tasks should generalise better than four isolated models — and produce a more drug-discovery-relevant representation of molecular space.

**Tasks:**

| Dataset | Task | Type | Size |
|:---|:---|:---:|:---:|
| BBBP | Blood-brain barrier penetration | Classification | 2,039 |
| hERG | Cardiac ion channel inhibition | Classification | 655 |
| CYP3A4 | Cytochrome P450 3A4 inhibition | Classification | 670 |
| FreeSolv | Hydration free energy | Regression | 642 |

All datasets sourced from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).

---

## Current Status

| Phase | Description | Status |
|:---|:---|:---:|
| **Phase 1** | Data pipeline — TDC integration, feature engineering, scaffold splits | ✅ Complete |
| **Phase 2** | Model architecture — shared encoder, task heads, loss functions | ✅ Complete |
| **Phase 3** | Training pipeline — AdamW, early stopping, checkpointing | ✅ Complete |
| **Phase 4** | Per-task evaluation — AUC, RMSE, calibration metrics | 🔄 In Progress |
| **Phase 5** | Ablation — single-task vs multi-task performance comparison | ⏳ Pending |
| **Phase 6** | Uncertainty quantification — temperature scaling calibration | ⏳ Pending |

---

## What's Built

### Data Pipeline (Complete)

Datasets are downloaded automatically via TDC. Molecular features are generated using RDKit:

- **ECFP4 fingerprints** — 1,024-bit Morgan fingerprints (radius 2) capturing local structural environments
- **Molecular descriptors** — 25 physicochemical features: TPSA, LogP, MW, HBD, HBA, rotatable bonds, ring counts, stereocenters
- **Total input dimension:** 1,049 features per molecule
- **Splitting strategy:** Murcko scaffold split (75/15/10) — ensures structurally distinct molecules in each split, the correct approach for evaluating generalisation in drug discovery

### Model Architecture (Complete)

```
Input (1,049 features)
       │
┌──────▼──────────────────────────┐
│  Shared Encoder                 │
│  Dense(512) → ReLU → Dropout    │
│  Dense(256) → ReLU → Dropout    │
│  Dense(128) → ReLU → Dropout    │
└──────┬──────────────────────────┘
       │
  ┌────┴────┬─────────┬──────────┐
  ▼         ▼         ▼          ▼
BBBP      hERG     CYP3A4    FreeSolv
head      head      head       head
(BCE)     (BCE)     (BCE)      (MSE)
```

- **Shared encoder:** Learns a joint molecular representation across all four tasks
- **Task-specific heads:** Independent prediction layers per task, allowing task-specific decision boundaries
- **Loss:** Weighted sum of per-task losses (BCE for classification, MSE for regression)
- **Optimiser:** AdamW with differentiated learning rates — lower for shared encoder (1e-4), higher for task heads (3e-4)
- **Regularisation:** Dropout, weight decay, early stopping (patience=10)

### Training Pipeline (Complete)

Training converges stably. Early stopping triggered at epoch 11 in the reference run, indicating the model learns quickly from the combined signal. Per-task evaluation metrics (AUC per classification task, RMSE for FreeSolv) are being computed and will be added to this README when complete.

---

## Quickstart

```bash
git clone https://github.com/SauravKulkarni3999/multi-task-admet
cd multi-task-admet

# Install dependencies
pip install -r requirements.txt
pip install -r tdc_requirements.txt

# Verify setup
python tests/run_all_tests.py

# Preprocess data (downloads TDC datasets automatically)
python run_preprocessing.py

# Train
python run_training.py --config configs/model_config.yaml
```

> **RDKit note:** Installation can conflict in some local environments. If you encounter issues, running in a fresh conda environment or Google Colab resolves them reliably.

---

## Repository Structure

```
multi-task-admet/
│
├── README.md
├── requirements.txt
├── tdc_requirements.txt
│
├── configs/
│   ├── model_config.yaml          # Architecture and training hyperparameters
│   └── preprocessing_config.yaml  # Feature generation and split parameters
│
├── data/
│   └── processed/                 # Generated by run_preprocessing.py
│       ├── processed_admet_data.pkl
│       ├── feature_info.json
│       └── dataset_statistics.csv
│
├── src/
│   ├── data/
│   │   ├── datasets.py            # TDC dataset loaders per task
│   │   ├── preprocessor.py        # End-to-end preprocessing pipeline
│   │   └── data_loader.py         # PyTorch DataLoader integration
│   ├── features/
│   │   └── molecular_features.py  # ECFP4 + descriptor generation
│   ├── loaders/
│   │   └── tdc.py                 # TDC API wrappers
│   ├── models/
│   │   └── admet_model.py         # MultiTaskADMETPredictor
│   ├── training/
│   │   └── training.py            # ADMETTrainer, EarlyStopping
│   └── utils/
│       └── evaluation.py          # Per-task metrics (in progress)
│
├── tests/
│   ├── test_integration.py        # End-to-end pipeline test
│   ├── test_tdc_connection.py     # TDC dataset loading test
│   ├── test_data_splits.py        # Scaffold split verification
│   └── run_all_tests.py           # Test runner
│
├── run_preprocessing.py
└── run_training.py
```

---

## Design Decisions

**Why scaffold splitting over random splitting:** Random splits allow structurally similar molecules to appear in both train and test sets, inflating apparent generalisation. Murcko scaffold splits ensure the test set contains genuinely unseen chemical scaffolds — the correct evaluation for a model intended to guide drug discovery decisions.

**Why shared encoder over four independent models:** Multi-task learning is motivated by the known correlations between ADMET properties. Lipophilicity affects both BBB penetration and metabolic stability. A shared representation should capture these joint signals. The ablation study (Phase 5) will quantify whether this actually improves over single-task baselines.

**Why differentiated learning rates:** The shared encoder should update conservatively to avoid catastrophic forgetting of signal from low-data tasks (hERG N=655, CYP3A4 N=670). Task heads update faster because they need to specialise quickly from the shared representation.

**Why ECFP4 over graph representations:** ECFP4 fingerprints provide a fast, reproducible baseline that is well-understood in the ADMET literature. The natural extension — replacing the fingerprint encoder with a GNN operating on molecular graphs — is flagged as future work and connects directly to the [GraphDTA-3D](https://github.com/SauravKulkarni3999/GraphDTA-3D) architecture in this portfolio.

---

## Connection to the Broader Research Arc

This project closes the loop on the portfolio's core thesis at the drug-likeness screening stage. The [AMP Benchmark](https://github.com/SauravKulkarni3999/AMP-Activity-Toxicity-Baseline-Benchmark) characterised individual peptides for activity and toxicity. [GraphDTA-3D](https://github.com/SauravKulkarni3999/GraphDTA-3D) predicted binding affinity for small molecule drug-target pairs. This project takes a candidate molecule and simultaneously screens it across the four most decision-critical ADMET filters before it ever enters a wet lab.

The CLI integration hook is designed to accept the output of the [Cyclic AMP Pipeline](https://github.com/SauravKulkarni3999/chiral-amp-pipeline) directly:

```bash
python src/utils/evaluation.py \
  --smiles_file candidates.csv \
  --tasks BBBP,HERG,CYP3A4,FreeSolv
```

| Project | Focus | Status |
|:---|:---|:---|
| [**AMP Benchmark**](https://github.com/SauravKulkarni3999/AMP-Activity-Toxicity-Baseline-Benchmark) | Proves 3D biophysical features outperform 1D sequences for AMP prediction | Complete |
| [**GraphDTA-3D**](https://github.com/SauravKulkarni3999/GraphDTA-3D) | Structure-aware GNN for drug-target affinity using AlphaFold residue graphs | Complete |
| [**Cyclic AMP Pipeline**](https://github.com/SauravKulkarni3999/chiral-amp-pipeline) | DPO-aligned ProtGPT2 for de novo cyclic AMP generation with ESMFold validation | Submitted — Chiral Blueprint Build Challenge 2026 |
| [**Struct-MMP**](https://github.com/SauravKulkarni3999/struct-mmp) | Interpretable ML for binding affinity changes from matched molecular pairs | In Progress |
| **Multi-Task ADMET** (this repo) | Simultaneous ADMET property prediction with shared molecular encoder | In Progress |

---

## References

1. Huang K, et al. Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. *NeurIPS*. 2021. https://tdcommons.ai
2. Wu Z, et al. MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*. 2018;9(2):513–530.
3. Ramsundar B, et al. Massively Multitask Networks for Drug Discovery. *arXiv*. 2015. arXiv:1502.02072
4. Rogers D, Hahn M. Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*. 2010;50(5):742–754.

---

*For questions or collaboration, reach out via [LinkedIn](https://linkedin.com/in/sauravkulkarni) or [email](mailto:sauravajitkulkarni@gmail.com).*
