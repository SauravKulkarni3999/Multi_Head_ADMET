#!/usr/bin/env python3
"""
Test script for ADMET preprocessing pipeline

This script tests the complete preprocessing pipeline to ensure everything works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.data.datasets import DatasetManager, create_sample_data
        from src.features.molecular_features import MolecularFeatureGenerator, FeatureProcessor
        from src.data.preprocessor import ADMETPreprocessor
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("\nTesting dataset loading...")
    
    try:
        from src.data.datasets import DatasetManager, create_sample_data
        
        # Test dataset manager
        manager = DatasetManager()
        info = manager.get_dataset_info()
        print(f"✓ Dataset manager created with {len(info)} datasets")
        
        # Test sample data creation
        sample_data = create_sample_data()
        print(f"✓ Sample data created with {len(sample_data)} datasets")
        
        for name, data in sample_data.items():
            print(f"  - {name.upper()}: {len(data)} samples")
            
        return True
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False

def test_feature_generation():
    """Test molecular feature generation"""
    print("\nTesting feature generation...")
    
    try:
        from src.features.molecular_features import MolecularFeatureGenerator, FeatureProcessor
        
        # Create feature generator
        feature_gen = MolecularFeatureGenerator(fp_size=1024, fp_radius=2)
        print("✓ Feature generator created")
        
        # Test with sample SMILES
        sample_smiles = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]  # Ibuprofen
        
        # Generate features
        fingerprints, descriptors, valid_indices = feature_gen.generate_features_batch(sample_smiles)
        print(f"✓ Features generated: {fingerprints.shape}, {descriptors.shape}")
        
        # Test feature processor
        processor = FeatureProcessor(feature_gen)
        feature_info = processor.get_feature_info()
        print(f"✓ Feature info: {feature_info['total_features']} total features")
        
        return True
    except Exception as e:
        print(f"✗ Feature generation error: {e}")
        return False

def test_preprocessing_pipeline():
    """Test complete preprocessing pipeline"""
    print("\nTesting preprocessing pipeline...")
    
    try:
        from src.data.preprocessor import ADMETPreprocessor
        
        # Create preprocessor
        preprocessor = ADMETPreprocessor()
        print("✓ Preprocessor created")
        
        # Run preprocessing with sample data
        processed_data = preprocessor.preprocess_all_datasets(use_sample_data=True)
        print(f"✓ Preprocessing completed: {len(processed_data)} datasets processed")
        
        # Get statistics
        stats = preprocessor.get_dataset_statistics()
        print(f"✓ Statistics generated: {len(stats)} datasets")
        
        # Analyze features
        analysis = preprocessor.analyze_features()
        print(f"✓ Feature analysis completed")
        
        # Create data splits
        splits = preprocessor.create_data_splits()
        print(f"✓ Data splits created: {len(splits)} datasets")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing pipeline error: {e}")
        return False

def test_save_load():
    """Test save and load functionality"""
    print("\nTesting save/load functionality...")
    
    try:
        from src.data.preprocessor import ADMETPreprocessor
        
        # Create preprocessor and process data
        preprocessor = ADMETPreprocessor()
        processed_data = preprocessor.preprocess_all_datasets(use_sample_data=True)
        
        # Save data
        preprocessor.save_processed_data("test_data.pkl")
        print("✓ Data saved successfully")
        
        # Create new preprocessor and load data
        new_preprocessor = ADMETPreprocessor()
        loaded_data = new_preprocessor.load_processed_data("test_data.pkl")
        print("✓ Data loaded successfully")
        
        # Clean up test file
        import os
        test_file = Path("data/processed/test_data.pkl")
        if test_file.exists():
            os.remove(test_file)
            print("✓ Test file cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ Save/load error: {e}")
        return False

def main():
    """Run all tests"""
    print("ADMET Preprocessing Pipeline Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_dataset_loading,
        test_feature_generation,
        test_preprocessing_pipeline,
        test_save_load
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The preprocessing pipeline is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
