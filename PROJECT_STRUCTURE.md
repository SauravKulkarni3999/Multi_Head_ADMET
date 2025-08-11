# Multi-Head ADMET Project Structure

## Overview

This project implements a comprehensive pipeline for multi-task ADMET prediction using molecular fingerprints and descriptors. The project is organized to support the complete machine learning workflow from data preprocessing to model training and evaluation.

## Directory Structure

```
Multi_Head_ADMET/
├── README.md                    # Project overview and usage instructions
├── requirements.txt             # Python dependencies
├── PROJECT_STRUCTURE.md         # This file - detailed project structure
├── run_preprocessing.py         # Main preprocessing script
├── test_preprocessing.py        # Test script for the preprocessing pipeline
│
├── configs/                     # Configuration files
│   └── preprocessing_config.yaml # Preprocessing parameters
│
├── data/                        # Data storage (created during runtime)
│   ├── raw/                     # Raw dataset files
│   └── processed/               # Processed data and features
│
├── src/                         # Source code
│   ├── __init__.py
│   │
│   ├── data/                    # Data handling modules
│   │   ├── __init__.py
│   │   ├── datasets.py          # Dataset loading and preprocessing
│   │   └── preprocessor.py      # Complete preprocessing pipeline
│   │
│   ├── features/                # Feature engineering modules
│   │   ├── __init__.py
│   │   └── molecular_features.py # Molecular fingerprint and descriptor generation
│   │
│   ├── models/                  # Neural network architectures (future)
│   │   └── __init__.py
│   │
│   ├── training/                # Training scripts (future)
│   │   └── __init__.py
│   │
│   └── utils/                   # Utility functions (future)
│       └── __init__.py
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── __init__.py
│   └── 01_data_exploration.ipynb # Data exploration and visualization
│
├── results/                     # Model outputs and results (future)
│   └── .gitkeep
│
└── documents/                   # Project documentation
    └── Project Scope - Multi-Task ADMET.pdf
```

## Core Components

### 1. Data Management (`src/data/`)

#### `datasets.py`
- **Purpose**: Handles loading and preprocessing of ADMET datasets
- **Key Classes**:
  - `ADMETDataset`: Base class for all ADMET datasets
  - `BBBPDataset`: Blood-Brain Barrier Penetration dataset
  - `HERGDataset`: Human Ether-a-go-go Related Gene dataset
  - `CYP3A4Dataset`: Cytochrome P450 3A4 dataset
  - `ClinToxDataset`: Clinical Toxicity dataset
  - `ESOLDataset`: Estimated SOLubility dataset
  - `DatasetManager`: Manages multiple datasets

#### `preprocessor.py`
- **Purpose**: Complete preprocessing pipeline
- **Key Classes**:
  - `ADMETPreprocessor`: Orchestrates the entire preprocessing workflow
- **Features**:
  - Dataset loading and validation
  - Feature generation
  - Data splitting (train/val/test)
  - Statistics generation
  - Data persistence

### 2. Feature Engineering (`src/features/`)

#### `molecular_features.py`
- **Purpose**: Generates molecular features from SMILES strings
- **Key Classes**:
  - `MolecularFeatureGenerator`: Generates ECFP4 fingerprints and descriptors
  - `FeatureProcessor`: Combines and processes features
  - `FeatureAnalyzer`: Analyzes feature properties
- **Features**:
  - RDKit ECFP4 fingerprints (1024 bits, radius 2)
  - 25 molecular descriptors (TPSA, LogP, MW, HBD, HBA, etc.)
  - Feature analysis and visualization

### 3. Configuration (`configs/`)

#### `preprocessing_config.yaml`
- **Purpose**: Centralized configuration for preprocessing parameters
- **Sections**:
  - Feature generation parameters
  - Dataset configurations
  - Data splitting parameters
  - Output settings
  - Logging configuration

### 4. Scripts

#### `run_preprocessing.py`
- **Purpose**: Main script to run the complete preprocessing pipeline
- **Features**:
  - Command-line interface
  - Configuration file support
  - Comprehensive logging
  - Progress tracking
  - Error handling

#### `test_preprocessing.py`
- **Purpose**: Test script to verify pipeline functionality
- **Tests**:
  - Module imports
  - Dataset loading
  - Feature generation
  - Complete pipeline
  - Save/load functionality

### 5. Notebooks (`notebooks/`)

#### `01_data_exploration.ipynb`
- **Purpose**: Interactive data exploration and visualization
- **Features**:
  - Dataset overview
  - Feature analysis
  - Statistical summaries
  - Visualizations
  - Pipeline testing

## Datasets

### Supported ADMET Datasets

1. **BBBP** (Blood-Brain Barrier Penetration)
   - Task: Binary classification
   - Target: p_np (permeable/non-permeable)
   - Size: ~2,000 compounds

2. **hERG** (Human Ether-a-go-go Related Gene)
   - Task: Binary classification
   - Target: hERG_inhibition
   - Size: ~1,000 compounds

3. **CYP3A4** (Cytochrome P450 3A4)
   - Task: Binary classification
   - Target: CYP3A4_inhibition
   - Size: ~1,000 compounds

4. **ClinTox** (Clinical Toxicity)
   - Task: Binary classification
   - Target: toxicity
   - Size: ~1,500 compounds

5. **ESOL** (Estimated SOLubility)
   - Task: Regression
   - Target: log_solubility
   - Size: ~1,100 compounds

## Features

### Molecular Fingerprints
- **Type**: RDKit ECFP4 (Morgan fingerprints)
- **Size**: 1024 bits
- **Radius**: 2
- **Purpose**: Capture structural information

### Molecular Descriptors
- **Count**: 25 descriptors
- **Categories**:
  - Physicochemical properties (MolWt, LogP, TPSA)
  - Structural features (HBD, HBA, RotatableBonds)
  - Ring systems (AromaticRings, SaturatedRings)
  - Atom types (HeavyAtomCount, Heteroatoms)
  - Stereochemistry (Stereocenters, SpiroAtoms)

## Usage

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run preprocessing with sample data**:
   ```bash
   python run_preprocessing.py --use-sample-data
   ```

3. **Run preprocessing with real data**:
   ```bash
   python run_preprocessing.py --config configs/preprocessing_config.yaml
   ```

4. **Test the pipeline**:
   ```bash
   python test_preprocessing.py
   ```

### Configuration

Edit `configs/preprocessing_config.yaml` to customize:
- Feature generation parameters
- Dataset selection
- Data splitting ratios
- Output settings

### Data Exploration

Open `notebooks/01_data_exploration.ipynb` for interactive exploration of:
- Dataset statistics
- Feature distributions
- Molecular property analysis
- Pipeline validation

## Output Files

After running the preprocessing pipeline, the following files are generated in `data/processed/`:

- `processed_admet_data.pkl`: Complete processed data
- `feature_info.json`: Feature metadata
- `dataset_statistics.csv`: Dataset summaries
- `feature_analysis.json`: Feature analysis results
- `preprocessing.log`: Processing logs

## Future Extensions

The project structure is designed to support future extensions:

- **Models** (`src/models/`): Neural network architectures
- **Training** (`src/training/`): Training scripts and utilities
- **Utils** (`src/utils/`): Common utility functions
- **Results** (`results/`): Model outputs and visualizations

## Dependencies

### Core Dependencies
- `torch`: Deep learning framework
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `rdkit-pypi`: Molecular cheminformatics
- `matplotlib`, `seaborn`: Visualization

### Development Dependencies
- `tqdm`: Progress bars
- `pyyaml`: Configuration file parsing
- `jupyter`: Interactive notebooks

## Contributing

When adding new components:

1. Follow the existing directory structure
2. Add appropriate `__init__.py` files
3. Include comprehensive docstrings
4. Add tests to `test_preprocessing.py`
5. Update this documentation

## License

MIT License - see LICENSE file for details.
