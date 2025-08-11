"""
Data Preprocessing Pipeline

This module provides a complete preprocessing pipeline that:
1. Loads ADMET datasets
2. Generates molecular features (ECFP4 + descriptors)
3. Prepares data for multi-task learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path
import pickle
import json

from .datasets import DatasetManager, create_sample_data
from ..features.molecular_features import MolecularFeatureGenerator, FeatureProcessor, FeatureAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADMETPreprocessor:
    """Complete preprocessing pipeline for ADMET datasets"""
    
    def __init__(self, 
                 fp_size: int = 1024, 
                 fp_radius: int = 2,
                 output_dir: str = "data/processed"):
        self.fp_size = fp_size
        self.fp_radius = fp_radius
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dataset_manager = DatasetManager()
        self.feature_generator = MolecularFeatureGenerator(fp_size, fp_radius)
        self.feature_processor = FeatureProcessor(self.feature_generator)
        self.feature_analyzer = FeatureAnalyzer()
        
        # Store processed data
        self.processed_datasets = {}
        self.feature_info = None
        
    def preprocess_all_datasets(self, use_sample_data: bool = False) -> Dict[str, Dict]:
        """Preprocess all ADMET datasets"""
        logger.info("Starting preprocessing of all ADMET datasets...")
        
        if use_sample_data:
            # Use sample data for testing
            raw_datasets = create_sample_data()
        else:
            # Load actual datasets
            raw_datasets = self.dataset_manager.load_all_datasets()
            
        processed_results = {}
        
        for dataset_name, raw_data in raw_datasets.items():
            logger.info(f"Processing {dataset_name.upper()} dataset...")
            
            try:
                # Process dataset
                features, processed_data = self.feature_processor.process_dataset(raw_data)
                
                # Store results
                processed_results[dataset_name] = {
                    'features': features,
                    'data': processed_data,
                    'n_samples': len(processed_data),
                    'n_features': features.shape[1]
                }
                
                logger.info(f"{dataset_name.upper()}: {len(processed_data)} samples, {features.shape[1]} features")
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue
                
        self.processed_datasets = processed_results
        
        # Get feature information
        self.feature_info = self.feature_processor.get_feature_info()
        
        return processed_results
        
    def save_processed_data(self, filename: str = "processed_admet_data.pkl"):
        """Save processed data to disk"""
        if not self.processed_datasets:
            raise ValueError("No processed data available. Run preprocess_all_datasets() first.")
            
        output_path = self.output_dir / filename
        
        # Prepare data for saving
        save_data = {
            'processed_datasets': self.processed_datasets,
            'feature_info': self.feature_info,
            'preprocessing_params': {
                'fp_size': self.fp_size,
                'fp_radius': self.fp_radius
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        logger.info(f"Processed data saved to {output_path}")
        
        # Also save feature info as JSON for easy inspection
        json_path = self.output_dir / "feature_info.json"
        with open(json_path, 'w') as f:
            json.dump(self.feature_info, f, indent=2)
            
        logger.info(f"Feature info saved to {json_path}")
        
    def load_processed_data(self, filename: str = "processed_admet_data.pkl") -> Dict:
        """Load processed data from disk"""
        input_path = self.output_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")
            
        with open(input_path, 'rb') as f:
            save_data = pickle.load(f)
            
        self.processed_datasets = save_data['processed_datasets']
        self.feature_info = save_data['feature_info']
        
        logger.info(f"Processed data loaded from {input_path}")
        return save_data
        
    def get_dataset_statistics(self) -> pd.DataFrame:
        """Get statistics for all processed datasets"""
        if not self.processed_datasets:
            raise ValueError("No processed data available.")
            
        stats = []
        for dataset_name, data in self.processed_datasets.items():
            dataset_info = self.dataset_manager.get_dataset_info()[dataset_name]
            
            stat = {
                'dataset': dataset_name.upper(),
                'task_type': dataset_info['task_type'],
                'n_samples': data['n_samples'],
                'n_features': data['n_features'],
                'target_column': dataset_info['target_column']
            }
            
            # Add target statistics
            target_col = dataset_info['target_column']
            target_values = data['data'][target_col]
            
            if dataset_info['task_type'] == 'classification':
                stat['n_positive'] = int(target_values.sum())
                stat['n_negative'] = int(len(target_values) - target_values.sum())
                stat['positive_ratio'] = float(target_values.mean())
            else:  # regression
                stat['target_mean'] = float(target_values.mean())
                stat['target_std'] = float(target_values.std())
                stat['target_min'] = float(target_values.min())
                stat['target_max'] = float(target_values.max())
                
            stats.append(stat)
            
        return pd.DataFrame(stats)
        
    def analyze_features(self) -> Dict:
        """Analyze features across all datasets"""
        if not self.processed_datasets:
            raise ValueError("No processed data available.")
            
        analysis_results = {}
        
        # Analyze fingerprint sparsity
        all_fingerprints = []
        for dataset_name, data in self.processed_datasets.items():
            fp_size = self.feature_info['fingerprint_size']
            fingerprints = data['features'][:, :fp_size]
            all_fingerprints.append(fingerprints)
            
        combined_fingerprints = np.vstack(all_fingerprints)
        sparsity_info = self.feature_analyzer.analyze_fingerprint_sparsity(combined_fingerprints)
        analysis_results['fingerprint_sparsity'] = sparsity_info
        
        # Analyze descriptor distributions
        all_descriptors = []
        for dataset_name, data in self.processed_datasets.items():
            fp_size = self.feature_info['fingerprint_size']
            descriptors = data['features'][:, fp_size:]
            all_descriptors.append(descriptors)
            
        combined_descriptors = np.vstack(all_descriptors)
        descriptor_df = pd.DataFrame(
            combined_descriptors, 
            columns=self.feature_info['descriptor_names']
        )
        desc_stats = self.feature_analyzer.analyze_descriptor_distributions(descriptor_df)
        analysis_results['descriptor_statistics'] = desc_stats
        
        return analysis_results
        
    def prepare_multi_task_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, str]]:
        """Prepare data for multi-task learning"""
        if not self.processed_datasets:
            raise ValueError("No processed data available.")
            
        # Find common compounds across datasets
        common_compounds = self.dataset_manager.get_common_compounds()
        
        if len(common_compounds) == 0:
            logger.warning("No common compounds found across datasets. Using all compounds.")
            # Use all compounds from each dataset
            multi_task_features = []
            multi_task_targets = {}
            task_types = {}
            
            for dataset_name, data in self.processed_datasets.items():
                features = data['features']
                target_col = self.dataset_manager.get_dataset_info()[dataset_name]['target_column']
                targets = data['data'][target_col].values
                
                multi_task_features.append(features)
                multi_task_targets[dataset_name] = targets
                task_types[dataset_name] = self.dataset_manager.get_dataset_info()[dataset_name]['task_type']
                
            # For now, we'll use the first dataset's features as the common feature set
            # In practice, you might want to implement a more sophisticated approach
            common_features = multi_task_features[0]
            
        else:
            logger.info(f"Found {len(common_compounds)} common compounds across datasets")
            # Implement logic to align features and targets for common compounds
            # This is a simplified version - you might want to implement more sophisticated alignment
            common_features = None
            multi_task_targets = {}
            task_types = {}
            
        return common_features, multi_task_targets, task_types
        
    def create_data_splits(self, 
                          test_size: float = 0.2, 
                          val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, Dict]:
        """Create train/validation/test splits for all datasets"""
        if not self.processed_datasets:
            raise ValueError("No processed data available.")
            
        from sklearn.model_selection import train_test_split
        
        splits = {}
        
        for dataset_name, data in self.processed_datasets.items():
            features = data['features']
            target_col = self.dataset_manager.get_dataset_info()[dataset_name]['target_column']
            targets = data['data'][target_col].values
            
            # For very small datasets, use simpler splitting
            if len(features) < 10:
                # Use 60/20/20 split for small datasets
                X_temp, X_test, y_temp, y_test = train_test_split(
                    features, targets, 
                    test_size=0.2, 
                    random_state=random_state,
                    stratify=None  # Don't stratify for very small datasets
                )
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=0.25,  # 25% of remaining 80% = 20% of total
                    random_state=random_state,
                    stratify=None
                )
            else:
                # Create train/test split
                X_temp, X_test, y_temp, y_test = train_test_split(
                    features, targets, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=targets if self.dataset_manager.get_dataset_info()[dataset_name]['task_type'] == 'classification' else None
                )
                
                # Create train/validation split
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size_adjusted,
                    random_state=random_state,
                    stratify=y_temp if self.dataset_manager.get_dataset_info()[dataset_name]['task_type'] == 'classification' else None
                )
            
            splits[dataset_name] = {
                'train': {'X': X_train, 'y': y_train},
                'val': {'X': X_val, 'y': y_val},
                'test': {'X': X_test, 'y': y_test}
            }
            
            logger.info(f"{dataset_name.upper()} splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
        return splits


def main():
    """Main preprocessing pipeline"""
    logger.info("Starting ADMET preprocessing pipeline...")
    
    # Initialize preprocessor
    preprocessor = ADMETPreprocessor()
    
    # Preprocess all datasets (using sample data for testing)
    processed_data = preprocessor.preprocess_all_datasets(use_sample_data=True)
    
    # Get dataset statistics
    stats = preprocessor.get_dataset_statistics()
    print("\nDataset Statistics:")
    print(stats)
    
    # Analyze features
    analysis = preprocessor.analyze_features()
    print(f"\nFingerprint sparsity: {analysis['fingerprint_sparsity']['sparsity']:.3f}")
    
    # Create data splits
    splits = preprocessor.create_data_splits()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    logger.info("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
