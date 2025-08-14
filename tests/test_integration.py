#!/usr/bin/env python3
"""
Integration test for ADMET multi-task model
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

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
