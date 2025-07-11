#!/usr/bin/env python3
"""
Demonstration Script for Data Utilities Module

This script demonstrates how to use the centralized data utilities for the 
Delay Discounting MVPA pipeline. It shows all the key functions and their usage.

Run this script to:
1. Check data availability and integrity
2. Load and validate data
3. Extract ROI time series
4. Perform quality control checks

Author: Cognitive Neuroscience Lab, Stanford University
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Import all data utilities
from data_utils import (
    # Core loading functions
    load_behavioral_data, load_fmri_data, load_confounds, 
    
    # ROI and mask functions
    extract_roi_timeseries, load_mask, check_mask_exists,
    
    # Subject management
    SubjectManager, get_complete_subjects,
    
    # Data validation
    DataValidator, check_data_integrity,
    
    # Utility functions
    save_processed_data, load_processed_data,
    
    # Exceptions
    DataError
)

from oak_storage_config import OAKConfig

def demo_subject_discovery():
    """Demonstrate subject discovery and management"""
    print("\n" + "="*60)
    print("DEMO 1: SUBJECT DISCOVERY AND MANAGEMENT")
    print("="*60)
    
    # Initialize configuration and subject manager
    config = OAKConfig()
    manager = SubjectManager(config)
    
    # Get available subjects
    print("\n1. Finding subjects with complete data...")
    complete_subjects = manager.get_available_subjects(require_both=True)
    print(f"Found {len(complete_subjects)} subjects with complete data:")
    for subj in complete_subjects[:5]:  # Show first 5
        print(f"  - {subj}")
    if len(complete_subjects) > 5:
        print(f"  ... and {len(complete_subjects) - 5} more")
    
    # Get subjects with any data
    print("\n2. Finding subjects with any available data...")
    any_data_subjects = manager.get_available_subjects(require_both=False)
    print(f"Found {len(any_data_subjects)} subjects with any data")
    
    # Get subject summary
    print("\n3. Getting detailed subject summary...")
    summary = manager.get_subject_summary(complete_subjects[:3])  # Check first 3
    print("Subject Summary:")
    print(summary)
    
    return complete_subjects

def demo_behavioral_data_loading(subjects):
    """Demonstrate behavioral data loading and validation"""
    print("\n" + "="*60)
    print("DEMO 2: BEHAVIORAL DATA LOADING AND VALIDATION")
    print("="*60)
    
    config = OAKConfig()
    
    if not subjects:
        print("No subjects available for demonstration")
        return
    
    test_subject = subjects[0]
    print(f"\nTesting with subject: {test_subject}")
    
    # Load behavioral data with validation and SV computation
    print("\n1. Loading behavioral data with full processing...")
    try:
        behavioral_data = load_behavioral_data(
            test_subject, 
            config, 
            validate=True, 
            compute_sv=True
        )
        
        print(f"✓ Loaded {len(behavioral_data)} trials")
        print(f"Columns: {list(behavioral_data.columns)}")
        print(f"Sample data:")
        print(behavioral_data[['onset', 'choice', 'delay_days', 'sv_chosen', 'sv_unchosen']].head())
        
        # Show data quality metrics
        choice_rate = behavioral_data['choice'].mean()
        valid_trials = (~behavioral_data['choice'].isna()).sum()
        print(f"\nData Quality:")
        print(f"  Valid trials: {valid_trials}/{len(behavioral_data)}")
        print(f"  Choice rate (LL): {choice_rate:.3f}")
        
    except DataError as e:
        print(f"✗ Data error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Demonstrate validation separately
    print("\n2. Demonstrating data validation...")
    validator = DataValidator(config)
    validation_result = validator.validate_behavioral_data(behavioral_data, test_subject)
    
    print(f"Validation Results:")
    print(f"  Valid: {validation_result['valid']}")
    print(f"  Metrics: {validation_result['metrics']}")
    if validation_result['warnings']:
        print(f"  Warnings: {validation_result['warnings']}")
    if validation_result['errors']:
        print(f"  Errors: {validation_result['errors']}")

def demo_fmri_data_loading(subjects):
    """Demonstrate fMRI data loading and validation"""
    print("\n" + "="*60)
    print("DEMO 3: fMRI DATA LOADING AND VALIDATION")
    print("="*60)
    
    config = OAKConfig()
    
    if not subjects:
        print("No subjects available for demonstration")
        return None
    
    test_subject = subjects[0]
    print(f"\nTesting with subject: {test_subject}")
    
    # Load fMRI data
    print("\n1. Loading fMRI data...")
    try:
        fmri_img = load_fmri_data(
            test_subject, 
            config, 
            smoothed=False, 
            validate=True
        )
        
        print(f"✓ Loaded fMRI data: {fmri_img.shape}")
        print(f"Voxel size: {fmri_img.header.get_zooms()[:3]}")
        print(f"TR: {fmri_img.header.get_zooms()[3]:.2f}s")
        
        # Show basic statistics
        data = fmri_img.get_fdata()
        print(f"Data range: {data.min():.1f} to {data.max():.1f}")
        print(f"Mean intensity: {data.mean():.1f}")
        
    except DataError as e:
        print(f"✗ Data error: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None
    
    # Load confounds
    print("\n2. Loading confounds...")
    try:
        confounds = load_confounds(test_subject, config)
        if confounds is not None:
            print(f"✓ Loaded confounds: {confounds.shape}")
            print(f"Confound columns: {list(confounds.columns)}")
        else:
            print("⚠ No confounds available")
    except Exception as e:
        print(f"✗ Error loading confounds: {e}")
    
    return fmri_img

def demo_mask_operations():
    """Demonstrate mask loading and checking"""
    print("\n" + "="*60)
    print("DEMO 4: MASK OPERATIONS")
    print("="*60)
    
    config = OAKConfig()
    
    print("\n1. Checking mask availability...")
    for roi_name, mask_path in config.ROI_MASKS.items():
        exists = check_mask_exists(mask_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {roi_name}: {mask_path}")
        
        if exists:
            try:
                mask_img = load_mask(mask_path, validate=True)
                mask_data = mask_img.get_fdata()
                n_voxels = (mask_data > 0).sum()
                print(f"    → {n_voxels} voxels")
            except Exception as e:
                print(f"    → Error loading: {e}")
    
    print("\n2. Attempting to create missing masks...")
    missing_masks = [roi for roi, path in config.ROI_MASKS.items() 
                    if not check_mask_exists(path)]
    
    if missing_masks:
        print(f"Missing masks: {missing_masks}")
        print("Run create_roi_masks.py to create them")
    else:
        print("All masks available!")

def demo_roi_timeseries_extraction(subjects):
    """Demonstrate ROI time series extraction"""
    print("\n" + "="*60)
    print("DEMO 5: ROI TIME SERIES EXTRACTION")
    print("="*60)
    
    config = OAKConfig()
    
    if not subjects:
        print("No subjects available for demonstration")
        return
    
    test_subject = subjects[0]
    print(f"\nTesting with subject: {test_subject}")
    
    # Check which ROIs are available
    available_rois = []
    for roi_name, mask_path in config.ROI_MASKS.items():
        if check_mask_exists(mask_path):
            available_rois.append(roi_name)
    
    if not available_rois:
        print("No ROI masks available. Run create_roi_masks.py first.")
        return
    
    print(f"\nAvailable ROIs: {available_rois}")
    
    # Extract time series for first available ROI
    roi_name = available_rois[0]
    print(f"\n1. Extracting time series from {roi_name}...")
    
    try:
        timeseries = extract_roi_timeseries(
            test_subject, 
            roi_name,
            config,
            smoothed=False,
            standardize=True,
            detrend=True
        )
        
        print(f"✓ Extracted time series: {timeseries.shape}")
        print(f"Time points: {timeseries.shape[0]}")
        print(f"Voxels: {timeseries.shape[1]}")
        print(f"Mean activation: {timeseries.mean():.3f} ± {timeseries.std():.3f}")
        
        # Show temporal characteristics
        print(f"\nTemporal characteristics:")
        print(f"  First 5 time points (mean): {timeseries[:5].mean(axis=1)}")
        
    except DataError as e:
        print(f"✗ Data error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def demo_data_integrity_check():
    """Demonstrate comprehensive data integrity checking"""
    print("\n" + "="*60)
    print("DEMO 6: COMPREHENSIVE DATA INTEGRITY CHECK")
    print("="*60)
    
    print("\n1. Running data integrity check...")
    try:
        integrity_report = check_data_integrity()
        
        print(f"\nIntegrity Report Summary:")
        print(f"  Total subjects: {len(integrity_report)}")
        print(f"  Complete data: {integrity_report['complete'].sum()}")
        print(f"  fMRI available: {integrity_report['has_fmri'].sum()}")
        print(f"  Behavior available: {integrity_report['has_behavior'].sum()}")
        print(f"  Valid behavior: {integrity_report['behavior_valid'].sum()}")
        print(f"  Valid fMRI: {integrity_report['fmri_valid'].sum()}")
        
        # Show details for first few subjects
        print(f"\nFirst 3 subjects detailed:")
        display_cols = ['subject_id', 'complete', 'n_trials', 'accuracy']
        print(integrity_report[display_cols].head(3))
        
        return integrity_report
        
    except Exception as e:
        print(f"✗ Error in integrity check: {e}")
        return None

def demo_data_saving_loading():
    """Demonstrate data saving and loading with metadata"""
    print("\n" + "="*60)
    print("DEMO 7: DATA SAVING AND LOADING")
    print("="*60)
    
    # Create sample processed data
    sample_data = {
        'analysis_type': 'demo',
        'results': np.random.randn(100, 10),
        'parameters': {'k': 0.1, 'accuracy': 0.75},
        'timestamps': pd.date_range('2024-01-01', periods=100, freq='D')
    }
    
    output_file = './demo_processed_data.pkl'
    
    print(f"\n1. Saving processed data to {output_file}...")
    try:
        save_processed_data(sample_data, output_file, subject_id='demo_subject')
        print("✓ Data saved successfully")
        
        print(f"\n2. Loading processed data from {output_file}...")
        loaded_data, metadata = load_processed_data(output_file)
        
        print(f"✓ Data loaded successfully")
        print(f"Data keys: {list(loaded_data.keys())}")
        print(f"Metadata: {metadata}")
        
        # Clean up
        Path(output_file).unlink()
        print("✓ Demo file cleaned up")
        
    except Exception as e:
        print(f"✗ Error in save/load demo: {e}")

def main():
    """Run all data utilities demonstrations"""
    print("DELAY DISCOUNTING DATA UTILITIES DEMONSTRATION")
    print("=" * 70)
    print("This script demonstrates all features of the data_utils module")
    print("Run with --help to see available options")
    
    import argparse
    parser = argparse.ArgumentParser(description='Demonstrate data utilities')
    parser.add_argument('--skip-loading', action='store_true',
                       help='Skip data loading demos (faster)')
    parser.add_argument('--quick', action='store_true',
                       help='Run only quick demos')
    args = parser.parse_args()
    
    try:
        # Demo 1: Subject discovery (always run)
        subjects = demo_subject_discovery()
        
        if not args.quick:
            # Demo 2-3: Data loading (can be slow)
            if not args.skip_loading and subjects:
                demo_behavioral_data_loading(subjects)
                fmri_img = demo_fmri_data_loading(subjects)
            
            # Demo 4: Mask operations
            demo_mask_operations()
            
            # Demo 5: ROI extraction (requires masks and data)
            if not args.skip_loading and subjects:
                demo_roi_timeseries_extraction(subjects)
        
        # Demo 6: Data integrity (always run)
        demo_data_integrity_check()
        
        # Demo 7: Save/load (always run, quick)
        demo_data_saving_loading()
        
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE!")
        print("✓ All data utilities functions demonstrated")
        print("✓ Check the code in demo_data_utils.py for usage examples")
        print("✓ Use these patterns in your own analysis scripts")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 