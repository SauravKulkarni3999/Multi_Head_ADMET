#!/usr/bin/env python3
"""
Test data splitting functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.preprocessor import ADMETPreprocessor

def test_data_splits():
    """Test data splitting"""
    print("🧪 Testing Data Splits...")
    
    try:
        # Load preprocessor
        preprocessor = ADMETPreprocessor()
        preprocessor.load_processed_data()
        
        # Test data splitting with scaffold strategy
        splits = preprocessor.create_data_splits(
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            split_strategy="scaffold"  # ✅ Use scaffold split
        )
        
        # Verify splits
        for dataset_name, split_data in splits.items():
            print(f"\n📊 {dataset_name.upper()} splits:")
            for split_name, data in split_data.items():
                print(f"   {split_name}: {len(data['X'])} samples")
                assert len(data['X']) == len(data['y']), f"Mismatch in {dataset_name} {split_name}"
        
        print("\n✅ All data splits created successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_splits()
