# Multi-Head ADMET Prediction

A deep learning framework for multi-task ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction using molecular fingerprints and descriptors.

## Project Overview

This project implements a multi-head neural network architecture to predict multiple ADMET properties simultaneously from molecular structures.

## Datasets

The project uses the following standard ADMET datasets from TDC (Therapeutics Data Commons):

- **BBBP** (Blood-Brain Barrier Penetration): Binary classification (2,039 samples)
- **hERG** (Human Ether-a-go-go Related Gene): Binary classification (655 samples)
- **CYP3A4** (Cytochrome P450 3A4): Binary classification (670 samples)
- **FreeSolv** (Hydration Free Energy): Regression (642 samples)

**Note**: All datasets are automatically downloaded from TDC and preprocessed with scaffold splitting for molecular data.

## Features

### Primary Features
- **RDKit ECFP4 fingerprints**: 1024-bit Morgan fingerprints with radius 2
- **Molecular descriptors**: TPSA, LogP, MW, HBD, HBA, rotatable bonds (20-30 scalar features)

### Feature Engineering
- **ECFP4 fingerprints** (1,024 bits): Morgan fingerprints with radius 2 for structural information
- **Molecular descriptors** (25 features): TPSA, LogP, MW, HBD, HBA, rotatable bonds, and other physicochemical properties
- **Total features**: 1,049 concatenated features for comprehensive molecular representation
- **Feature sparsity**: ~96.3% (typical for molecular fingerprints)

## Project Structure

```
Multi_Head_ADMET/
├── data/                   # Dataset storage
│   └── processed/         # Preprocessed data files
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── loaders/           # TDC data loaders
│   ├── models/            # Neural network architectures
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── tests/                  # Test suite
│   ├── test_integration.py
│   ├── test_tdc_connection.py
│   ├── test_data_splits.py
│   └── run_all_tests.py
├── notebooks/             # Jupyter notebooks for exploration
├── results/               # Model outputs and visualizations
├── configs/               # Configuration files
├── run_preprocessing.py   # Data preprocessing script
├── run_training.py        # Model training script
└── requirements.txt       # Python dependencies
```

## Installation

1. **Clone the repository**
2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r tdc_requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python tests/run_all_tests.py
   ```

## Usage

### Quick Start

1. **Run Tests** (Recommended first step):
   ```bash
   python tests/run_all_tests.py
   ```

2. **Data Preprocessing**:
   ```bash
   python run_preprocessing.py
   ```
   This will:
   - Download datasets from TDC
   - Generate molecular features (ECFP4 + descriptors)
   - Create scaffold splits (train/val/test)
   - Normalize regression targets
   - Save processed data to `data/processed/`

3. **Model Training**:
   ```bash
   python run_training.py
   ```
   This will:
   - Load preprocessed data
   - Train multi-task model with early stopping
   - Save best model checkpoint
   - Display training progress

4. **Model Evaluation** (Optional):
   ```bash
   python src/utils/evaluation.py --model-path results/checkpoints/best_model.pth --data-path data/processed/processed_admet_data.pkl --output-dir reports
   ```

### Individual Components

- **Data Preparation**: Run data preprocessing scripts
- **Feature Engineering**: Generate molecular fingerprints and descriptors  
- **Model Training**: Train multi-head neural network
- **Evaluation**: Assess model performance across all tasks

## Model Architecture

### Multi-Task Learning Framework
The model uses a shared encoder with task-specific prediction heads:

- **Shared Encoder**: 3-layer neural network (512 → 256 → 128 units)
- **Task-Specific Heads**: Separate prediction layers for each ADMET property
- **Loss Functions**: 
  - Classification tasks: Binary Cross-Entropy with Logits
  - Regression tasks: Mean Squared Error
- **Optimization**: AdamW with differentiated learning rates
- **Regularization**: Dropout, weight decay, early stopping

### Training Configuration
- **Batch Size**: 128
- **Learning Rate**: 1e-4 (shared), 3e-4 (task-specific)
- **Early Stopping**: Patience of 10 epochs
- **Data Splitting**: Scaffold split (molecular structure-based)
- **Target Normalization**: Z-score normalization for regression tasks

## Current Performance

Based on the latest training run:
- **Training Loss**: Decreased from 0.80 to 0.58 (11 epochs)
- **Validation Loss**: Decreased from 0.89 to 0.73
- **Early Stopping**: Triggered at epoch 11 (no improvement)
- **Model Convergence**: Stable training with positive, decreasing losses

**Note**: The model shows stable convergence with proper loss behavior (no negative or exploding losses).

For detailed project status and achievements, see [PROJECT_STATUS.md](PROJECT_STATUS.md).


## Troubleshooting

### Common Issues

1. **TDC Dataset Loading Errors**:
   - Ensure internet connection for downloading datasets
   - Check TDC installation: `pip install -r tdc_requirements.txt`
   - Run: `python tests/test_tdc_connection.py`

2. **Memory Issues**:
   - Reduce batch size in training configuration
   - Use smaller model architecture
   - Ensure sufficient RAM (recommended: 8GB+)

3. **CUDA/GPU Issues**:
   - Model automatically uses CPU if CUDA unavailable
   - For GPU training, ensure PyTorch CUDA installation

4. **Data Preprocessing Errors**:
   - Check RDKit installation
   - Verify SMILES validity in datasets
   - Run: `python tests/test_data_splits.py`

### Performance Optimization

- **Scaffold Splitting**: Ensures molecular diversity in splits
- **Target Normalization**: Prevents regression loss explosion
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stabilizes training

## Integration Analysis and Required Changes

### 1. **Missing Import in Training Module**

The `GradientHarmonizer` class is defined in `admet_model.py` but used in `training.py` without proper import:

```python:src/training/training.py
# Add this import at the top
from ..models.admet_model import GradientHarmonizer
```

### 2. **Data Loading Integration**

The model expects specific data formats. Create a data loader that integrates with the existing preprocessing:

```python:src/data/data_loader.py
"""
Data Loader for Multi-Task ADMET Training

This module provides PyTorch DataLoader classes that integrate with the 
preprocessing pipeline and model architecture.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ADMETDataset(Dataset):
    """PyTorch Dataset for ADMET multi-task learning"""
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray], 
                 masks: Dict[str, np.ndarray], task_types: Dict[str, str]):
        self.features = torch.FloatTensor(features)
        self.targets = {k: torch.FloatTensor(v) for k, v in targets.items()}
        self.masks = {k: torch.BoolTensor(v) for k, v in masks.items()}
        self.task_types = task_types
        self.task_names = list(targets.keys())
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': {k: v[idx] for k, v in self.targets.items()},
            'masks': {k: v[idx] for k, v in self.masks.items()}
        }


class MultiTaskDataLoader:
    """Data loader for multi-task ADMET training"""
    
    def __init__(self, preprocessor, batch_size: int = 128, shuffle: bool = True):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def create_data_loaders(self, splits: Dict[str, Dict]) -> Dict[str, DataLoader]:
        """Create DataLoaders for train/val/test splits"""
        loaders = {}
        
        for split_name, split_data in splits.items():
            # Prepare multi-task data
            features, targets, task_types, masks = self._prepare_multi_task_data(split_data)
            
            # Create dataset
            dataset = ADMETDataset(features, targets, masks, task_types)
            
            # Create DataLoader
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle if split_name == 'train' else False,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=True
            )
            
            loaders[split_name] = loader
            
        return loaders
    
    def _prepare_multi_task_data(self, split_data: Dict) -> Tuple[np.ndarray, Dict, Dict, Dict]:
        """Prepare data for multi-task learning"""
        # This would integrate with the preprocessor's prepare_multi_task_data method
        # For now, creating a simplified version
        features = split_data['X']
        targets = split_data['y']
        task_types = {'task': 'classification'}  # Simplified
        masks = {'task': np.ones(len(features), dtype=bool)}
        
        return features, targets, task_types, masks
```

### 3. **Model Configuration Integration**

Create a model configuration file that aligns with the preprocessing:

```yaml:configs/model_config.yaml
# ADMET Model Configuration

# Model architecture
model:
  input_dim: 1048  # 1024 (ECFP4) + 24 (descriptors)
  hidden_dims: [512, 256, 128]
  use_residual: true
  use_attention: true
  dropout_schedule: true
  
# Task configuration
tasks:
  classification:
    - bbbp
    - herg
    - cyp3a4
  regression:
    - freesolv

# Training parameters
training:
  batch_size: 128
  max_epochs: 100
  learning_rate_shared: 1e-4
  learning_rate_heads: 3e-4
  weight_decay_shared: 1e-4
  weight_decay_heads: 1e-5
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  
# Uncertainty estimation
uncertainty:
  enabled: true
  aleatoric: true
  epistemic: false  # Future enhancement
  
# Model saving
output:
  model_dir: "results/models"
  checkpoint_dir: "results/checkpoints"
  tensorboard_dir: "results/tensorboard"
```

### 4. **Training Script Integration**

Create a main training script that integrates all components:

```python:run_training.py
#!/usr/bin/env python3
"""
ADMET Multi-Task Training Script

This script trains the multi-task ADMET model using the preprocessed data.
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessor import ADMETPreprocessor
from src.data.data_loader import MultiTaskDataLoader
from src.models.admet_model import MultiTaskADMETPredictor, create_model_config
from src.training.training import ADMETTrainer, EarlyStopping


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='ADMET Multi-Task Training')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/model_config.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/preprocessing_config.yaml',
        help='Path to preprocessing configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    model_config = load_config(args.config)
    data_config = load_config(args.data_config)
    
    # Setup logging
    setup_logging(model_config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ADMET multi-task training...")
    
    try:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        preprocessor = ADMETPreprocessor()
        preprocessor.load_processed_data()
        
        # Create data splits
        logger.info("Creating data splits...")
        splits = preprocessor.create_data_splits(
            test_size=data_config['splitting']['test_size'],
            val_size=data_config['splitting']['validation_size'],
            random_state=data_config['splitting']['random_state'],
            split_strategy=data_config['splitting']['split_strategy']
        )
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loader = MultiTaskDataLoader(
            preprocessor, 
            batch_size=model_config['training']['batch_size']
        )
        loaders = data_loader.create_data_loaders(splits)
        
        # Create model
        logger.info("Creating model...")
        model_config_dict = create_model_config(
            input_dim=preprocessor.feature_info['total_features'],
            tasks_config=model_config['tasks']
        )
        model = MultiTaskADMETPredictor(**model_config_dict)
        
        # Create trainer
        logger.info("Setting up trainer...")
        trainer = ADMETTrainer(model)
        
        # Training loop
        logger.info("Starting training...")
        early_stopping = EarlyStopping(
            patience=model_config['training']['early_stopping_patience']
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(model_config['training']['max_epochs']):
            # Train
            train_loss, train_task_losses = trainer.train_epoch(
                loaders['train'], epoch, model_config['training']['max_epochs']
            )
            
            # Validate
            val_loss, predictions, targets, uncertainties = trainer.validate(loaders['val'])
            
            logger.info(f"Epoch {epoch+1}/{model_config['training']['max_epochs']}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info("Early stopping triggered")
                break
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save checkpoint
                checkpoint_path = Path(model_config['output']['checkpoint_dir']) / "best_model.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': model_config
                }, checkpoint_path)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
```

### 5. **Evaluation and Metrics Integration**

Create evaluation utilities:

```python:src/utils/evaluation.py
"""
Evaluation utilities for ADMET multi-task learning
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import torch


class ADMETEvaluator:
    """Evaluator for ADMET multi-task models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        
    def evaluate_predictions(self, predictions: Dict, targets: Dict, 
                           uncertainties: Dict, task_types: Dict) -> Dict:
        """Evaluate predictions for all tasks"""
        results = {}
        
        for task_name in predictions.keys():
            if task_name not in targets:
                continue
                
            pred = predictions[task_name].cpu().numpy()
            target = targets[task_name].cpu().numpy()
            uncertainty = uncertainties[task_name].cpu().numpy()
            
            if task_types[task_name] == 'classification':
                task_results = self._evaluate_classification(pred, target, uncertainty)
            else:
                task_results = self._evaluate_regression(pred, target, uncertainty)
                
            results[task_name] = task_results
            
        return results
    
    def _evaluate_classification(self, pred: np.ndarray, target: np.ndarray, 
                               uncertainty: np.ndarray) -> Dict:
        """Evaluate classification task"""
        # Convert probabilities to binary predictions
        pred_binary = (pred > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(target, pred)
        ap = average_precision_score(target, pred)
        accuracy = np.mean(pred_binary == target)
        
        # Uncertainty calibration
        calibration_error = self._calculate_calibration_error(pred, target, uncertainty)
        
        return {
            'auc': auc,
            'average_precision': ap,
            'accuracy': accuracy,
            'calibration_error': calibration_error,
            'uncertainty_mean': np.mean(uncertainty),
            'uncertainty_std': np.std(uncertainty)
        }
    
    def _evaluate_regression(self, pred: np.ndarray, target: np.ndarray, 
                           uncertainty: np.ndarray) -> Dict:
        """Evaluate regression task"""
        mse = mean_squared_error(target, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(target, pred)
        mae = np.mean(np.abs(pred - target))
        
        # Uncertainty quality
        uncertainty_quality = self._calculate_uncertainty_quality(pred, target, uncertainty)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'uncertainty_quality': uncertainty_quality,
            'uncertainty_mean': np.mean(uncertainty),
            'uncertainty_std': np.std(uncertainty)
        }
    
    def _calculate_calibration_error(self, pred: np.ndarray, target: np.ndarray, 
                                   uncertainty: np.ndarray) -> float:
        """Calculate calibration error for classification"""
        # Simplified calibration error calculation
        bins = np.linspace(0, 1, 10)
        bin_indices = np.digitize(pred, bins) - 1
        
        calibration_error = 0.0
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = pred[mask].mean()
                bin_acc = target[mask].mean()
                calibration_error += np.abs(bin_conf - bin_acc)
                
        return calibration_error / (len(bins) - 1)
    
    def _calculate_uncertainty_quality(self, pred: np.ndarray, target: np.ndarray, 
                                     uncertainty: np.ndarray) -> float:
        """Calculate uncertainty quality for regression"""
        # Correlation between uncertainty and absolute error
        abs_error = np.abs(pred - target)
        correlation = np.corrcoef(uncertainty, abs_error)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
```

### 6. **Requirements Updates**

Add missing dependencies to `requirements.txt`:

```txt:requirements.txt
# Add these lines to requirements.txt
# For uncertainty quantification
scipy>=1.7.0

# For model checkpointing and logging
tensorboard>=2.10.0
wandb>=0.13.0

# For advanced metrics
scikit-learn>=1.0.0
```

### 7. **Integration Testing Script**

Create a test script to verify integration:

```python:test_integration.py
#!/usr/bin/env python3
"""
Integration test for ADMET multi-task model
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessor import ADMETPreprocessor
from src.models.admet_model import MultiTaskADMETPredictor, create_model_config
from src.training.training import ADMETTrainer


def test_integration():
    """Test the complete integration"""
    print("Testing ADMET multi-task model integration...")
    
    # 1. Test data preprocessing
    print("1. Testing data preprocessing...")
    preprocessor = ADMETPreprocessor()
    processed_data = preprocessor.preprocess_all_datasets(use_sample_data=True)
    print(f"   ✓ Processed {len(processed_data)} datasets")
    
    # 2. Test model creation
    print("2. Testing model creation...")
    feature_info = preprocessor.feature_info
    model_config = create_model_config(
        input_dim=feature_info['total_features'],
        tasks_config={
            'classification': ['bbbp', 'herg', 'cyp3a4'],
            'regression': ['freesolv']
        }
    )
    model = MultiTaskADMETPredictor(**model_config)
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. Test forward pass
    print("3. Testing forward pass...")
    batch_size = 4
    input_features = torch.randn(batch_size, feature_info['total_features'])
    
    # Create sample masks
    masks = {}
    for task in model.all_tasks:
        masks[task] = torch.ones(batch_size, dtype=torch.bool)
    
    with torch.no_grad():
        outputs = model(input_features, masks)
    
    print(f"   ✓ Forward pass successful, {len(outputs)} outputs generated")
    
    # 4. Test loss computation
    print("4. Testing loss computation...")
    targets = {}
    for task in model.all_tasks:
        if task in model.classification_tasks:
            targets[task] = torch.randint(0, 2, (batch_size,)).float()
        else:
            targets[task] = torch.randn(batch_size)
    
    loss, task_losses = model.compute_loss(outputs, targets, masks)
    print(f"   ✓ Loss computation successful, total loss: {loss.item():.4f}")
    
    # 5. Test trainer setup
    print("5. Testing trainer setup...")
    trainer = ADMETTrainer(model)
    print(f"   ✓ Trainer created successfully")
    
    print("\n🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    test_integration()
```

### 8. **Documentation Updates**

Update the README to include model training instructions:

```markdown:README.md
# Multi-Task ADMET Prediction

## Quick Start

1. **Preprocess Data:**
   ```bash
   python run_preprocessing.py --use-sample-data
   ```

2. **Train Model:**
   ```bash
   python run_training.py --config configs/model_config.yaml
   ```

3. **Test Integration:**
   ```bash
   python test_integration.py
   ```

## Model Architecture

The multi-task ADMET model features:
- Shared molecular encoder with residual connections
- Task-specific feature extractors
- Uncertainty quantification
- Gradient harmonization to prevent negative transfer
- Adaptive dropout scheduling
