#!/usr/bin/env python3
"""
Test runner for ADMET Multi-Task Model
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from test_integration import test_integration
from test_tdc_connection import test_tdc_datasets
from test_data_splits import test_data_splits

def run_all_tests():
    """Run all tests"""
    print("🧪 Running ADMET Multi-Task Model Tests")
    print("=" * 50)
    
    tests = [
        ("TDC Connection", test_tdc_datasets),
        ("Data Splits", test_data_splits),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        print("-" * 30)
        
        start_time = time.time()
        try:
            success = test_func()
            end_time = time.time()
            duration = end_time - start_time
            
            if success:
                print(f"✅ {test_name} Test PASSED ({duration:.2f}s)")
                results.append((test_name, True, duration))
            else:
                print(f"❌ {test_name} Test FAILED ({duration:.2f}s)")
                results.append((test_name, False, duration))
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ {test_name} Test ERROR ({duration:.2f}s): {e}")
            results.append((test_name, False, duration))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.2f}s)")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
