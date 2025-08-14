#!/usr/bin/env python3
"""
Test TDC connection and dataset loading
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.loaders.tdc import load_bbbp, load_herg, load_cyp3a4, load_freesolv

def test_tdc_datasets():
    """Test loading real datasets from TDC"""
    print("🧪 Testing TDC Dataset Loading...")
    
    try:
        # Test BBBP dataset
        print("\n1️⃣ Loading BBBP dataset...")
        bbbp_data = load_bbbp()
        print(f"   ✅ BBBP loaded: {len(bbbp_data)} samples")
        print(f"   📊 Columns: {list(bbbp_data.columns)}")
        print(f"   📈 Sample data:\n{bbbp_data.head(2)}")
        
        # Test hERG dataset
        print("\n2️⃣ Loading hERG dataset...")
        herg_data = load_herg()
        print(f"   ✅ hERG loaded: {len(herg_data)} samples")
        print(f"   📊 Columns: {list(herg_data.columns)}")
        print(f"   📈 Sample data:\n{herg_data.head(2)}")
        
        # Test CYP3A4 dataset
        print("\n3️⃣ Loading CYP3A4 dataset...")
        cyp3a4_data = load_cyp3a4()
        print(f"   ✅ CYP3A4 loaded: {len(cyp3a4_data)} samples")
        print(f"   📊 Columns: {list(cyp3a4_data.columns)}")
        print(f"   📈 Sample data:\n{cyp3a4_data.head(2)}")
        
        # Test FreeSolv dataset
        print("\n4️⃣ Loading FreeSolv dataset...")
        freesolv_data = load_freesolv()
        print(f"   ✅ FreeSolv loaded: {len(freesolv_data)} samples")
        print(f"   📊 Columns: {list(freesolv_data.columns)}")
        print(f"   📈 Sample data:\n{freesolv_data.head(2)}")
        
        print("\n🎉 All TDC datasets loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading TDC datasets: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tdc_datasets()
