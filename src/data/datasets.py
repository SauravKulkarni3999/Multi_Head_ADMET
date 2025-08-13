"""
ADMET Dataset Loading and Preprocessing

This module provides functionality to load and preprocess standard ADMET datasets:
- BBBP (Blood-Brain Barrier Penetration)
- hERG (Human Ether-a-go-go Related Gene)
- CYP3A4 (Cytochrome P450 3A4)
- FreeSolv (Hydration Free Energy)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import logging
from src.loaders.tdc import load_bbbp, load_herg, load_cyp3a4, load_freesolv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADMETDataset:
    """Base class for ADMET datasets"""
    
    def __init__(self, name: str, data_path: str = None):
        self.name = name
        self.data_path = data_path or f"data/{name.lower()}"
        self.data = None
        self.smiles_col = "smiles"
        self.target_col = "target"
        
    def load_data(self) -> pd.DataFrame:
        """Load dataset from file"""
        raise NotImplementedError
        
    def preprocess(self) -> pd.DataFrame:
        """Preprocess the dataset"""
        raise NotImplementedError
        
    def get_task_type(self) -> str:
        """Return task type: 'classification' or 'regression'"""
        raise NotImplementedError


class BBBPDataset(ADMETDataset):
    """Blood-Brain Barrier Penetration dataset"""
    
    def __init__(self, data_path: str = None):
        super().__init__("BBBP", data_path)
        self.target_col = "p_np"
        self.smiles_col = "Drug"
        
    def load_data(self) -> pd.DataFrame:
        """Load BBBP dataset"""
        logger.info(f"Loading BBBP dataset from TDC...")
        self.data = load_bbbp()
        #Renaming columns for consistency
        self.data = self.data.rename(columns={"Drug": "smiles", "Y": "p_np"})
        self.smiles_col = "smiles"
        self.target_col = "p_np"
        return self.data
        
    def preprocess(self) -> pd.DataFrame:
        """Preprocess BBBP data"""
        if self.data is None:
            self.load_data()
            
        # Remove invalid SMILES
        self.data = self.data.dropna(subset=[self.smiles_col])
        
        # Ensure target is binary
        self.data[self.target_col] = self.data[self.target_col].astype(int)
        
        logger.info(f"BBBP dataset: {len(self.data)} samples")
        return self.data
        
    def get_task_type(self) -> str:
        return "classification"


class HERGDataset(ADMETDataset):
    """Human Ether-a-go-go Related Gene dataset"""
    
    def __init__(self, data_path: str = None):
        super().__init__("hERG", data_path)
        self.target_col = "hERG_inhibition"
        self.smiles_col = "Drug"
        
    def load_data(self) -> pd.DataFrame:
        """Load hERG dataset"""
        logger.info(f"Loading hERG dataset from TDC...")
        self.data = load_herg()
        #Renaming columns for consistency
        self.data = self.data.rename(columns={"Drug": "smiles", "Y": "hERG_inhibition"})
        self.smiles_col = "smiles"
        self.target_col = "hERG_inhibition"
        return self.data
        
    def preprocess(self) -> pd.DataFrame:
        """Preprocess hERG data"""
        if self.data is None:
            self.load_data()
            
        # Remove invalid SMILES
        self.data = self.data.dropna(subset=[self.smiles_col])
        
        # Ensure target is binary
        self.data[self.target_col] = self.data[self.target_col].astype(int)
        
        logger.info(f"hERG dataset: {len(self.data)} samples")
        return self.data
        
    def get_task_type(self) -> str:
        return "classification"


class CYP3A4Dataset(ADMETDataset):
    """Cytochrome P450 3A4 dataset"""
    
    def __init__(self, data_path: str = None):
        super().__init__("CYP3A4", data_path)
        self.target_col = "CYP3A4_inhibition"
        
    def load_data(self) -> pd.DataFrame:
        """Load CYP3A4 dataset"""
        logger.info(f"Loading CYP3A4 dataset from TDC...")
        self.data = load_cyp3a4()
        #Renaming columns for consistency
        self.data = self.data.rename(columns={"Drug": "smiles", "Y": "CYP3A4_inhibition"})
        self.smiles_col = "smiles"
        self.target_col = "CYP3A4_inhibition"
        return self.data
        
    def preprocess(self) -> pd.DataFrame:
        """Preprocess CYP3A4 data"""
        if self.data is None:
            self.load_data()
            
        # Remove invalid SMILES
        self.data = self.data.dropna(subset=[self.smiles_col])
        
        # Ensure target is binary
        self.data[self.target_col] = self.data[self.target_col].astype(int)
        
        logger.info(f"CYP3A4 dataset: {len(self.data)} samples")
        return self.data
        
    def get_task_type(self) -> str:
        return "classification"


class FreeSolvDataset(ADMETDataset):
    """FreeSolv dataset"""
    
    def __init__(self, data_path: str = None):
        super().__init__("FreeSolv", data_path)
        self.target_col = "hydration_free_energy"
        self.smiles_col = "Drug"
        
    def load_data(self) -> pd.DataFrame:
        """Load FreeSolv dataset"""
        logger.info(f"Loading FreeSolv dataset from TDC...")
        self.data = load_freesolv()
        #Renaming columns for consistency
        self.data = self.data.rename(columns={"Drug": "smiles", "Y": "hydration_free_energy"})
        self.smiles_col = "smiles"
        self.target_col = "hydration_free_energy"
        return self.data
        
    def preprocess(self) -> pd.DataFrame:
        """Preprocess FreeSolv data"""
        if self.data is None:
            self.load_data()
            
        # Remove invalid SMILES
        self.data = self.data.dropna(subset=[self.smiles_col])
        
        # Ensure target is float
        self.data[self.target_col] = self.data[self.target_col].astype(float)
        
        logger.info(f"FreeSolv dataset: {len(self.data)} samples")
        return self.data
        
    def get_task_type(self) -> str:
        return "regression"


class DatasetManager:
    """Manager class for handling multiple ADMET datasets"""
    
    def __init__(self):
        self.datasets = {
            'bbbp': BBBPDataset(),
            'herg': HERGDataset(),
            'cyp3a4': CYP3A4Dataset(),
            'freesolv': FreeSolvDataset(),
        }
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess all datasets"""
        processed_datasets = {}
        
        for name, dataset in self.datasets.items():
            logger.info(f"Processing {name.upper()} dataset...")
            processed_data = dataset.preprocess()
            processed_datasets[name] = processed_data
            
        return processed_datasets
        
    def get_dataset_info(self) -> Dict[str, Dict]:
        """Get information about all datasets"""
        info = {}
        
        for name, dataset in self.datasets.items():
            info[name] = {
                'task_type': dataset.get_task_type(),
                'target_column': dataset.target_col,
                'smiles_column': dataset.smiles_col
            }
            
        return info
        
    def get_common_compounds(self) -> pd.DataFrame:
        """Find compounds that appear in multiple datasets"""
        all_smiles = []
        dataset_names = []
        
        for name, dataset in self.datasets.items():
            if dataset.data is not None:
                smiles = dataset.data[dataset.smiles_col].tolist()
                all_smiles.extend(smiles)
                dataset_names.extend([name] * len(smiles))
                
        # Count occurrences
        smiles_df = pd.DataFrame({
            'smiles': all_smiles,
            'dataset': dataset_names
        })
        
        # Find compounds in multiple datasets
        compound_counts = smiles_df.groupby('smiles').size().reset_index(name='count')
        common_compounds = compound_counts[compound_counts['count'] > 1]
        
        return common_compounds


def create_sample_data():
    """Create sample data for testing purposes"""
    sample_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)OC",  # Aspirin (simplified)
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",  # Propranolol
        "CC1=C(C(=CC=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"  # Imatinib
    ]
    
    # Create sample datasets
    datasets = {
        'bbbp': pd.DataFrame({
            'smiles': sample_smiles,
            'p_np': [1, 0, 1, 1, 0],
            'name': ['Ibuprofen', 'Aspirin', 'Celecoxib', 'Propranolol', 'Imatinib']
        }),
        'herg': pd.DataFrame({
            'smiles': sample_smiles,
            'hERG_inhibition': [0, 1, 0, 1, 1],
            'name': ['Ibuprofen', 'Aspirin', 'Celecoxib', 'Propranolol', 'Imatinib']
        }),
        'cyp3a4': pd.DataFrame({
            'smiles': sample_smiles,
            'CYP3A4_inhibition': [0, 1, 1, 0, 1],
            'name': ['Ibuprofen', 'Aspirin', 'Celecoxib', 'Propranolol', 'Imatinib']
        }),
        'freesolv': pd.DataFrame({
            'smiles': sample_smiles,
            'hydration_free_energy': [0, 1, 0, 0, 1],
            'name': ['Ibuprofen', 'Aspirin', 'Celecoxib', 'Propranolol', 'Imatinib']
        })
    }
    
    return datasets


if __name__ == "__main__":
    # Test the dataset loading
    manager = DatasetManager()
    
    # Create sample data for testing
    sample_datasets = create_sample_data()
    
    # Print dataset information
    info = manager.get_dataset_info()
    for name, details in info.items():
        print(f"{name.upper()}: {details}")
        
    print("\nSample data created for testing purposes.")
