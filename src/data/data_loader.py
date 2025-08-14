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
        
        # Get dataset info for task types
        dataset_info = self.preprocessor.dataset_manager.get_dataset_info()
        
        # Create loaders for each split (train, val, test)
        for split_name in ['train', 'val', 'test']:
            # Prepare multi-task data for this split
            features, targets, task_types, masks = self._prepare_multi_task_data_for_split(
                splits, split_name, dataset_info
            )
            
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
    
    def _prepare_multi_task_data_for_split(self, splits: Dict, split_name: str, dataset_info: Dict) -> Tuple[np.ndarray, Dict, Dict, Dict]:
        """Prepare multi-task data for a specific split"""
        # Find the dataset with the most samples for this split
        max_samples = 0
        reference_dataset = None
        
        for dataset_name, split_data in splits.items():
            if split_name in split_data:
                n_samples = len(split_data[split_name]['X'])
                if n_samples > max_samples:
                    max_samples = n_samples
                    reference_dataset = dataset_name
        
        if reference_dataset is None:
            raise ValueError(f"No data found for split: {split_name}")
        
        # Use the reference dataset's features as the base
        reference_features = splits[reference_dataset][split_name]['X']
        combined_features = reference_features
        
        # Initialize targets and masks for all tasks
        all_targets = {}
        all_masks = {}
        task_types = {}
        
        for dataset_name, split_data in splits.items():
            if split_name in split_data:
                features = split_data[split_name]['X']
                targets = split_data[split_name]['y']
                
                # Create mask for this task (True where we have data)
                mask = np.zeros(max_samples, dtype=bool)
                mask[:len(features)] = True
                
                # Pad targets to match the maximum size
                padded_targets = np.zeros(max_samples)
                padded_targets[:len(targets)] = targets
                
                all_targets[dataset_name] = padded_targets
                all_masks[dataset_name] = mask
                task_types[dataset_name] = dataset_info[dataset_name]['task_type']
        
        return combined_features, all_targets, task_types, all_masks
