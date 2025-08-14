# ADMET Multi-Task Model Tests

This directory contains comprehensive tests for the ADMET Multi-Task Model pipeline.

## Test Files

### Core Tests
- **`test_integration.py`** - Tests complete pipeline integration
- **`test_tdc_connection.py`** - Tests TDC dataset loading
- **`test_data_splits.py`** - Tests data splitting functionality
- **`run_all_tests.py`** - Test runner that executes all tests

## Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Individual Tests
```bash
# Test TDC connection
python tests/test_tdc_connection.py

# Test data splitting
python tests/test_data_splits.py

# Test complete integration
python tests/test_integration.py
```

## Test Coverage

✅ **Data Loading** - TDC dataset connection and loading  
✅ **Data Preprocessing** - Feature generation and data splitting  
✅ **Model Architecture** - Model creation and forward pass  
✅ **Training Pipeline** - Loss computation and trainer setup  
✅ **Integration** - End-to-end pipeline verification  

## Test Results

Tests should pass before proceeding with training. If any test fails, check the error messages and fix the underlying issues.

## Adding New Tests

When adding new functionality, create corresponding tests in this directory following the existing naming convention: `test_<feature_name>.py`
