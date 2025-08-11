"""
Molecular Data Splitting Utilities

This module provides functions for splitting molecular datasets based on scaffolds.
"""

import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random

def _generate_scaffold(smiles, include_chirality=False):
    """Generate a Murcko scaffold from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, includeChirality=include_chirality)

def scaffold_split(df, smiles_col='smiles', test_size=0.1, val_size=0.1, seed=42):
    """
    Perform a scaffold split on a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        smiles_col (str): The name of the column containing SMILES strings.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the dataset to include in the validation split.
        seed (int): The random seed for shuffling scaffolds.

    Returns:
        tuple: A tuple containing the train, validation, and test DataFrames.
    """
    # Generate scaffolds
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(df[smiles_col]):
        scaffold = _generate_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(idx)

    # Shuffle scaffolds
    random.seed(seed)
    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    # Split scaffolds
    n_total = len(df)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    train_indices, val_indices, test_indices = [], [], []
    for scaffold_set in scaffold_sets:
        if len(test_indices) < n_test:
            test_indices.extend(scaffold_set)
        elif len(val_indices) < n_val:
            val_indices.extend(scaffold_set)
        else:
            train_indices.extend(scaffold_set)
            
    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[val_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True),
    )