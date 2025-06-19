#!/usr/bin/env python3
"""
Test script for Delay Discounting MVPA Pipeline

This script performs basic tests to ensure the pipeline components work correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import scipy
        import statsmodels
        import nibabel as nib
        import nilearn
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_behavioral_analysis():
    """Test behavioral analysis components"""
    print("\nTesting behavioral analysis...")
    
    try:
        # Import the behavioral analysis class
        from delay_discounting_mvpa_pipeline import BehavioralAnalysis, Config
        
        config = Config()
        behavioral_analysis = BehavioralAnalysis(config)
        
        # Test hyperbolic discount function
        delays = np.array([1, 7, 30, 90])
        k = 0.01
        values = behavioral_analysis.hyperbolic_discount_function(delays, k)
        
        # Test subjective value calculation
        amounts = np.array([25, 30, 40, 50])
        sv = behavioral_analysis.subjective_value(amounts, delays, k)
        
        # Create synthetic choice data for testing
        choices = np.array([0, 1, 1, 0])  # Binary choices
        large_amounts = np.array([25, 30, 40, 50])
        test_delays = np.array([1, 7, 30, 90])
        
        # Test discount rate fitting
        fit_result = behavioral_analysis.fit_discount_rate(choices, large_amounts, test_delays)
        
        if fit_result['success']:
            print("✓ Behavioral analysis components working")
            return True
        else:
            print(f"✗ Behavioral analysis failed: {fit_result['error']}")
            return False
            
    except Exception as e:
        print(f"✗ Behavioral analysis test failed: {e}")
        return False

def test_data_structure():
    """Test expected data structure"""
    print("\nTesting data structure expectations...")
    
    # Check if dataset descriptor files exist
    descriptor_dir = Path("Dataset_Descriptor_Files")
    
    if descriptor_dir.exists():
        print("✓ Dataset descriptor directory found")
        
        # Check for key files
        key_files = [
            "task-discountFix_events.json",
            "suggested_exclusions.csv"
        ]
        
        for file in key_files:
            if (descriptor_dir / file).exists():
                print(f"✓ Found {file}")
            else:
                print(f"⚠ Missing {file}")
        
        return True
    else:
        print("⚠ Dataset descriptor directory not found")
        print("  This is expected if running on a different system")
        return True

def create_synthetic_data():
    """Create synthetic data for testing"""
    print("\nCreating synthetic test data...")
    
    # Create temporary directories
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create synthetic behavioral data
    behavioral_dir = test_dir / "behavioral"
    behavioral_dir.mkdir(exist_ok=True)
    
    # Generate synthetic choice data for one subject
    n_trials = 50
    worker_id = "test_s001"
    
    # Create realistic delay discounting trial structure
    np.random.seed(42)  # For reproducible results
    
    large_amounts = np.random.choice([25, 30, 35, 40, 45, 50], size=n_trials)
    later_delays = np.random.choice([1, 7, 14, 30, 90, 180], size=n_trials)
    small_amounts = np.full(n_trials, 20)  # Always $20 for immediate option
    
    # Simulate choices based on hyperbolic discounting with k=0.02
    k_true = 0.02
    sv_large = large_amounts / (1 + k_true * later_delays)
    sv_small = small_amounts  # Immediate, so no discounting
    
    # Choice probability based on difference in subjective values
    choice_prob = 1 / (1 + np.exp(-(sv_large - sv_small)))
    choices = np.random.binomial(1, choice_prob)
    choice_labels = ['smaller_sooner' if c == 0 else 'larger_later' for c in choices]
    
    # Create trial onsets (every 4 seconds)
    onsets = np.arange(n_trials) * 4.0
    durations = np.full(n_trials, 2.0)  # 2 second trial duration
    response_times = np.random.uniform(0.5, 2.0, n_trials)
    
    # Create behavioral dataframe
    behavioral_df = pd.DataFrame({
        'onset': onsets,
        'duration': durations,
        'choice': choice_labels,
        'large_amount': large_amounts,
        'later_delay': later_delays,
        'small_amount': small_amounts,
        'response_time': response_times,
        'worker_id': worker_id
    })
    
    # Save behavioral data
    behavioral_file = behavioral_dir / f"{worker_id}_discountFix_events.tsv"
    behavioral_df.to_csv(behavioral_file, sep='\t', index=False)
    
    print(f"✓ Created synthetic behavioral data: {behavioral_file}")
    print(f"  - {n_trials} trials")
    print(f"  - True k value: {k_true}")
    print(f"  - Choice rate (larger-later): {np.mean(choices):.3f}")
    
    return test_dir

def test_with_synthetic_data(test_dir):
    """Test pipeline components with synthetic data"""
    print("\nTesting with synthetic data...")
    
    try:
        from delay_discounting_mvpa_pipeline import BehavioralAnalysis, Config
        
        # Modify config to use test data
        config = Config()
        config.BEHAVIOR_DIR = str(test_dir / "behavioral")
        
        behavioral_analysis = BehavioralAnalysis(config)
        
        # Test processing synthetic subject
        result = behavioral_analysis.process_subject_behavior("test_s001")
        
        if result['success']:
            print("✓ Successfully processed synthetic subject")
            print(f"  - Estimated k: {result['k']:.4f}")
            print(f"  - Model fit (pseudo-R²): {result['pseudo_r2']:.3f}")
            print(f"  - Choice rate: {result['choice_rate']:.3f}")
            print(f"  - Number of trials: {result['n_trials']}")
            return True
        else:
            print(f"✗ Failed to process synthetic subject: {result['error']}")
            return False
            
    except Exception as e:
        print(f"✗ Synthetic data test failed: {e}")
        return False

def cleanup_test_data(test_dir):
    """Clean up test data"""
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"✓ Cleaned up test data directory")

def main():
    """Run all tests"""
    print("Delay Discounting MVPA Pipeline - Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Package imports
    total_tests += 1
    if test_imports():
        tests_passed += 1
    
    # Test 2: Behavioral analysis components
    total_tests += 1
    if test_behavioral_analysis():
        tests_passed += 1
    
    # Test 3: Data structure
    total_tests += 1
    if test_data_structure():
        tests_passed += 1
    
    # Test 4: Synthetic data processing
    test_dir = create_synthetic_data()
    total_tests += 1
    if test_with_synthetic_data(test_dir):
        tests_passed += 1
    
    # Cleanup
    cleanup_test_data(test_dir)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Pipeline appears to be working correctly.")
        print("\nNext steps:")
        print("1. Create ROI masks: python create_roi_masks.py")
        print("2. Run main analysis: python delay_discounting_mvpa_pipeline.py")
        print("3. Or submit to cluster: sbatch submit_analysis_job.sh")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all packages are installed: pip install -r requirements.txt")
        print("2. Check that the data paths in Config are correct")
        print("3. Verify file permissions and access to data directories")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 