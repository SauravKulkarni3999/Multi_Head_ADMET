# Multi-Head ADMET Prediction

A deep learning framework for multi-task ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction using molecular fingerprints and descriptors.

## Project Overview

This project implements a multi-head neural network architecture to predict multiple ADMET properties simultaneously from molecular structures.

## Datasets

The project uses the following standard ADMET datasets:

- **BBBP** (Blood-Brain Barrier Penetration): Binary classification
- **hERG** (Human Ether-a-go-go Related Gene): Binary classification  
- **CYP3A4** (Cytochrome P450 3A4): Binary classification
- **ClinTox** (Clinical Toxicity): Binary classification
- **ESOL** (Estimated SOLubility): Regression

## Features

### Primary Features
- **RDKit ECFP4 fingerprints**: 1024-bit Morgan fingerprints with radius 2
- **Molecular descriptors**: TPSA, LogP, MW, HBD, HBA, rotatable bonds (20-30 scalar features)

### Feature Engineering
- ECFP4 fingerprints provide structural information
- Molecular descriptors capture physicochemical properties
- Concatenated feature vector for comprehensive molecular representation

## Project Structure

```
Multi_Head_ADMET/
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Neural network architectures
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
├── results/               # Model outputs and visualizations
└── configs/               # Configuration files
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**: Run data preprocessing scripts
2. **Feature Engineering**: Generate molecular fingerprints and descriptors
3. **Model Training**: Train multi-head neural network
4. **Evaluation**: Assess model performance across all tasks

## License

MIT License
