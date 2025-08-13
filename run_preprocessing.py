#!/usr/bin/env python3
"""
ADMET Data Preprocessing Pipeline

This script runs the complete preprocessing pipeline for ADMET datasets:
1. Loads datasets (BBBP, hERG, CYP3A4, FreeSolv)
2. Generates molecular features (ECFP4 fingerprints + descriptors)
3. Creates train/validation/test splits
4. Saves processed data for model training

Usage:
    python run_preprocessing.py [--config configs/preprocessing_config.yaml]
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessor import ADMETPreprocessor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['level'].upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if config['logging']['save_logs']:
        log_file = Path(config['output']['processed_data_dir']) / config['logging']['log_filename']
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)


def main():
    """Main preprocessing pipeline"""
    parser = argparse.ArgumentParser(description='ADMET Data Preprocessing Pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/preprocessing_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        help='Use sample data for testing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'features': {
                'fingerprint_size': 1024,
                'fingerprint_radius': 2,
                'include_descriptors': True
            },
            'output': {
                'processed_data_dir': 'data/processed'
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_filename': 'preprocessing.log'
            },
            'processing': {
                'use_sample_data': args.use_sample_data
            }
        }
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ADMET preprocessing pipeline...")
    logger.info(f"Configuration loaded from: {args.config}")
    
    try:
        # Initialize preprocessor
        preprocessor = ADMETPreprocessor(
            fp_size=config['features']['fingerprint_size'],
            fp_radius=config['features']['fingerprint_radius'],
            output_dir=config['output']['processed_data_dir']
        )
        
        # Override config with command line argument
        use_sample_data = args.use_sample_data or config['processing']['use_sample_data']
        
        # Preprocess all datasets
        logger.info("Preprocessing datasets...")
        processed_data = preprocessor.preprocess_all_datasets(use_sample_data=use_sample_data)
        
        # Get dataset statistics
        logger.info("Generating dataset statistics...")
        stats = preprocessor.get_dataset_statistics()
        print("\nDataset Statistics:")
        print(stats.to_string(index=False))
        
        # Save dataset statistics
        stats_path = Path(config['output']['processed_data_dir']) / config['output']['dataset_stats_filename']
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(stats_path, index=False)
        logger.info(f"Dataset statistics saved to: {stats_path}")
        
        # Analyze features
        logger.info("Analyzing features...")
        analysis = preprocessor.analyze_features()
        
        print(f"\nFeature Analysis:")
        print(f"Fingerprint sparsity: {analysis['fingerprint_sparsity']['sparsity']:.3f}")
        print(f"Total fingerprint bits: {analysis['fingerprint_sparsity']['total_bits']}")
        print(f"Active fingerprint bits: {analysis['fingerprint_sparsity']['active_bits']}")
        
        # Save feature analysis
        analysis_path = Path(config['output']['processed_data_dir']) / config['output']['feature_analysis_filename']
        import json
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Feature analysis saved to: {analysis_path}")
        
        # Create data splits
        logger.info("Creating data splits...")
        splits = preprocessor.create_data_splits(
            test_size=config['splitting']['test_size'],
            val_size=config['splitting']['validation_size'],
            random_state=config['splitting']['random_state'],
            split_strategy=config['splitting']['split_strategy']
        )
        
        # Save processed data
        logger.info("Saving processed data...")
        preprocessor.save_processed_data(config['output']['processed_data_filename'])
        
        # Print summary
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Processed datasets: {list(processed_data.keys())}")
        print(f"Total features: {preprocessor.feature_info['total_features']}")
        print(f"  - Fingerprint features: {preprocessor.feature_info['fingerprint_size']}")
        print(f"  - Descriptor features: {preprocessor.feature_info['descriptor_size']}")
        print(f"Output directory: {config['output']['processed_data_dir']}")
        print("="*50)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
