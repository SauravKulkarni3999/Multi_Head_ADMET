"""
Molecular Feature Engineering

This module provides functionality to generate molecular features:
- RDKit ECFP4 fingerprints (1024 bits, radius 2)
- Molecular descriptors (TPSA, LogP, MW, HBD, HBA, rotatable bonds)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from tqdm import tqdm

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.AtomPairs import Pairs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not found. Please install rdkit-pypi package.")
    Chem = None
    RDKIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MolecularFeatureGenerator:
    """Generate molecular features from SMILES strings"""
    
    def __init__(self, fp_size: int = 1024, fp_radius: int = 2):
        self.fp_size = fp_size
        self.fp_radius = fp_radius
        
        # Define molecular descriptors to calculate
        if RDKIT_AVAILABLE:
            self.descriptor_functions = {
                'MolWt': Descriptors.MolWt,
                'LogP': Descriptors.MolLogP,
                'TPSA': Descriptors.TPSA,
                'HBD': Descriptors.NumHDonors,
                'HBA': Descriptors.NumHAcceptors,
                'RotatableBonds': Descriptors.NumRotatableBonds,
                'AromaticRings': Descriptors.NumAromaticRings,
                'SaturatedRings': Descriptors.NumSaturatedRings,
                'RingCount': Descriptors.RingCount,
                'FractionCsp3': Descriptors.FractionCSP3,
                'HeavyAtomCount': Descriptors.HeavyAtomCount,
                'AtomCount':lambda mol: mol.GetNumAtoms(),
                'BondCount': lambda mol: mol.GetNumBonds(),
                'Stereocenters': rdMolDescriptors.CalcNumAtomStereoCenters,
                'SpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms,
                'BridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms,
                'Heteroatoms': rdMolDescriptors.CalcNumHeteroatoms,
                'AmideBonds': rdMolDescriptors.CalcNumAmideBonds,
                'AromaticAtoms': lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
                'SaturatedAtoms': lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3),
                'AliphaticAtoms': lambda mol: sum(1 for atom in mol.GetAtoms() if not atom.GetIsAromatic()),
                'AliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
                'AromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles,
                'SaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles,
                'AliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles
            }
        else:
            # Mock descriptor functions for testing
            self.descriptor_functions = {
                'MolWt': lambda x: 0.0,
                'LogP': lambda x: 0.0,
                'TPSA': lambda x: 0.0,
                'HBD': lambda x: 0.0,
                'HBA': lambda x: 0.0,
                'RotatableBonds': lambda x: 0.0,
                'AromaticRings': lambda x: 0.0,
                'SaturatedRings': lambda x: 0.0,
                'RingCount': lambda x: 0.0,
                'FractionCsp3': lambda x: 0.0,
                'HeavyAtomCount': lambda x: 0.0,
                'AtomCount': lambda x: 0.0,
                'BondCount': lambda x: 0.0,
                'Stereocenters': lambda x: 0.0,
                'SpiroAtoms': lambda x: 0.0,
                'BridgeheadAtoms': lambda x: 0.0,
                'Heteroatoms': lambda x: 0.0,
                'AmideBonds': lambda x: 0.0,
                'AromaticAtoms': lambda x: 0.0,
                'SaturatedAtoms': lambda x: 0.0,
                'AliphaticAtoms': lambda x: 0.0,
                'AliphaticRings': lambda x: 0.0,
                'AromaticHeterocycles': lambda x: 0.0,
                'SaturatedHeterocycles': lambda x: 0.0,
                'AliphaticHeterocycles': lambda x: 0.0
            }
        
    def smiles_to_mol(self, smiles: str) -> Optional[object]:
        """Convert SMILES string to RDKit Mol object"""
        if Chem is None:
            raise ImportError("RDKit is required for molecular feature generation")
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None
        return mol
        
    def generate_ecfp4_fingerprint(self, mol: object) -> np.ndarray:
        """Generate ECFP4 fingerprint (Morgan fingerprint with radius 2)"""
        if mol is None:
            return np.zeros(self.fp_size)
            
        # Generate Morgan fingerprint (ECFP4 equivalent)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 
            radius=self.fp_radius, 
            nBits=self.fp_size
        )
        
        # Convert to numpy array
        fp_array = np.zeros(self.fp_size)
        for i in range(self.fp_size):
            if fp.GetBit(i):
                fp_array[i] = 1
                
        return fp_array
        
    def calculate_molecular_descriptors(self, mol: object) -> Dict[str, float]:
        """Calculate molecular descriptors"""
        if mol is None:
            # Return default values for invalid molecules
            return {name: 0.0 for name in self.descriptor_functions.keys()}
            
        descriptors = {}
        for name, func in self.descriptor_functions.items():
            try:
                value = func(mol)
                # Handle NaN values
                if pd.isna(value) or np.isnan(value):
                    value = 0.0
                descriptors[name] = float(value)
            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")
                descriptors[name] = 0.0
                
        return descriptors
        
    def generate_features(self, smiles: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate both fingerprint and descriptors for a SMILES string"""
        mol = self.smiles_to_mol(smiles)
        
        # Generate ECFP4 fingerprint
        fingerprint = self.generate_ecfp4_fingerprint(mol)
        
        # Calculate molecular descriptors
        descriptors = self.calculate_molecular_descriptors(mol)
        
        return fingerprint, descriptors
        
    def generate_features_batch(self, smiles_list: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        """Generate features for a batch of SMILES strings"""
        if not RDKIT_AVAILABLE:
            # Return mock data for testing when RDKit is not available
            logger.warning("RDKit not available. Generating mock features for testing.")
            n_samples = len(smiles_list)
            fingerprints = np.random.randint(0, 2, (n_samples, self.fp_size))
            descriptors_list = []
            for _ in range(n_samples):
                desc = {name: np.random.random() for name in self.descriptor_functions.keys()}
                descriptors_list.append(desc)
            descriptors_df = pd.DataFrame(descriptors_list)
            valid_indices = list(range(n_samples))
            return fingerprints, descriptors_df, valid_indices
        
        fingerprints = []
        descriptors_list = []
        valid_indices = []
        
        logger.info(f"Generating features for {len(smiles_list)} molecules...")
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Processing molecules")):
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                # Generate fingerprint
                fp = self.generate_ecfp4_fingerprint(mol)
                fingerprints.append(fp)
                
                # Calculate descriptors
                desc = self.calculate_molecular_descriptors(mol)
                descriptors_list.append(desc)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping invalid SMILES at index {i}: {smiles}")
                
        # Convert to numpy arrays
        fingerprint_matrix = np.array(fingerprints)
        descriptors_df = pd.DataFrame(descriptors_list)
        
        logger.info(f"Generated features for {len(valid_indices)} valid molecules")
        
        return fingerprint_matrix, descriptors_df, valid_indices
        
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Get names of fingerprint bits and descriptor columns"""
        fp_names = [f"fp_{i}" for i in range(self.fp_size)]
        descriptor_names = list(self.descriptor_functions.keys())
        
        return fp_names, descriptor_names
        
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """Get dimensions of fingerprint and descriptor features"""
        return self.fp_size, len(self.descriptor_functions)


class FeatureProcessor:
    """Process and combine molecular features"""
    
    def __init__(self, feature_generator: MolecularFeatureGenerator):
        self.feature_generator = feature_generator
        
    def process_dataset(self, df: pd.DataFrame, smiles_col: str = 'smiles') -> Tuple[np.ndarray, pd.DataFrame]:
        """Process a dataset and return combined features"""
        smiles_list = df[smiles_col].tolist()
        
        # Generate features
        fingerprints, descriptors, valid_indices = self.feature_generator.generate_features_batch(smiles_list)
        
        # Filter dataset to only include valid molecules
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        
        # Combine features
        combined_features = self.combine_features(fingerprints, descriptors)
        
        return combined_features, valid_df
        
    def combine_features(self, fingerprints: np.ndarray, descriptors: pd.DataFrame) -> np.ndarray:
        """Combine fingerprint and descriptor features"""
        # Convert descriptors to numpy array
        descriptor_array = descriptors.values
        
        # Concatenate along feature axis
        combined = np.concatenate([fingerprints, descriptor_array], axis=1)
        
        return combined
        
    def get_feature_info(self) -> Dict[str, Union[int, List[str]]]:
        """Get information about the feature dimensions and names"""
        fp_size, desc_size = self.feature_generator.get_feature_dimensions()
        fp_names, desc_names = self.feature_generator.get_feature_names()
        
        return {
            'fingerprint_size': fp_size,
            'descriptor_size': desc_size,
            'total_features': fp_size + desc_size,
            'fingerprint_names': fp_names,
            'descriptor_names': desc_names,
            'feature_names': fp_names + desc_names
        }


class FeatureAnalyzer:
    """Analyze molecular features"""
    
    def __init__(self):
        pass
        
    def analyze_fingerprint_sparsity(self, fingerprints: np.ndarray) -> Dict[str, float]:
        """Analyze sparsity of fingerprint matrix"""
        total_bits = fingerprints.size
        active_bits = np.sum(fingerprints)
        sparsity = 1 - (active_bits / total_bits)
        
        return {
            'total_bits': total_bits,
            'active_bits': active_bits,
            'sparsity': sparsity,
            'density': 1 - sparsity
        }
        
    def analyze_descriptor_distributions(self, descriptors: pd.DataFrame) -> pd.DataFrame:
        """Analyze distributions of molecular descriptors"""
        stats = descriptors.describe()
        return stats
        
    def find_common_fingerprint_bits(self, fingerprints: np.ndarray, threshold: float = 0.1) -> List[int]:
        """Find fingerprint bits that are active in more than threshold fraction of molecules"""
        bit_frequencies = np.mean(fingerprints, axis=0)
        common_bits = np.where(bit_frequencies > threshold)[0]
        return common_bits.tolist()
        
    def find_rare_fingerprint_bits(self, fingerprints: np.ndarray, threshold: float = 0.01) -> List[int]:
        """Find fingerprint bits that are active in less than threshold fraction of molecules"""
        bit_frequencies = np.mean(fingerprints, axis=0)
        rare_bits = np.where(bit_frequencies < threshold)[0]
        return rare_bits.tolist()


def create_sample_features():
    """Create sample features for testing"""
    sample_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=C(C(C(=C(N1CCC(C(CC)CC)C)CC)C)C(=O)C2=CC=C(C=C2)NC(=O)C3C=CC(=CC=3)CC4C(=C(C(=C(C4C)C)C(=O)C5C=CC(=CC=5)NC(=O)C6C=CC(=CC=6)CC)C)C",  # Tacrolimus
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
        "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",  # Propranolol
        "CC1=C(C(=CC=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"  # Imatinib
    ]
    
    # Create feature generator
    feature_gen = MolecularFeatureGenerator()
    
    # Generate features
    fingerprints, descriptors, valid_indices = feature_gen.generate_features_batch(sample_smiles)
    
    # Create processor
    processor = FeatureProcessor(feature_gen)
    
    # Combine features
    combined_features = processor.combine_features(fingerprints, descriptors)
    
    # Get feature info
    feature_info = processor.get_feature_info()
    
    return {
        'fingerprints': fingerprints,
        'descriptors': descriptors,
        'combined_features': combined_features,
        'feature_info': feature_info,
        'valid_indices': valid_indices
    }


if __name__ == "__main__":
    # Test feature generation
    print("Testing molecular feature generation...")
    
    # Create sample features
    sample_results = create_sample_features()
    
    print(f"Feature dimensions: {sample_results['feature_info']['total_features']}")
    print(f"Fingerprint size: {sample_results['feature_info']['fingerprint_size']}")
    print(f"Descriptor size: {sample_results['feature_info']['descriptor_size']}")
    
    # Analyze features
    analyzer = FeatureAnalyzer()
    
    # Analyze fingerprint sparsity
    sparsity_info = analyzer.analyze_fingerprint_sparsity(sample_results['fingerprints'])
    print(f"\nFingerprint sparsity: {sparsity_info['sparsity']:.3f}")
    
    # Analyze descriptor distributions
    desc_stats = analyzer.analyze_descriptor_distributions(sample_results['descriptors'])
    print(f"\nDescriptor statistics:")
    print(desc_stats)
    
    print("\nFeature generation test completed successfully!")
