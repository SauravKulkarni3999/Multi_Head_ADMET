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
            val_loss, predictions, targets = trainer.validate(loaders['val'])
            
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
