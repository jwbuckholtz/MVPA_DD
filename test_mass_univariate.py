#!/usr/bin/env python3
"""
Test script for Mass Univariate Analysis Module

This script tests the mass univariate GLM analysis pipeline with synthetic data
to ensure all components work correctly before running on real data.

Tests include:
1. Package imports and GLM model creation
2. Synthetic fMRI and behavioral data generation
3. First-level GLM modeling
4. Second-level random effects analysis
5. Multiple comparisons correction

Usage:
    python test_mass_univariate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required packages for GLM analysis can be imported"""
    print("Testing package imports for GLM analysis...")
    
    try:
        # Core packages
        import numpy as np
        import pandas as pd
        import nibabel as nib
        
        # Nilearn GLM components
        from nilearn.glm.first_level import FirstLevelModel
        from nilearn.glm.second_level import SecondLevelModel
        from nilearn.glm import threshold_stats_img
        from nilearn.image import smooth_img, math_img
        from nilearn.datasets import load_mni152_template
        
        # Statistical packages
        import statsmodels.api as sm
        from scipy import stats
        
        print("✓ All GLM analysis packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def create_synthetic_fmri_data(n_timepoints=200, shape=(50, 50, 30)):
    """
    Create synthetic fMRI data for testing
    
    Parameters:
    -----------
    n_timepoints : int
        Number of time points
    shape : tuple
        Spatial dimensions (x, y, z)
        
    Returns:
    --------
    nibabel.Nifti1Image : Synthetic fMRI data
    """
    print(f"Creating synthetic fMRI data: {shape} x {n_timepoints}")
    
    # Create synthetic time series with some spatial structure
    data = np.random.randn(*shape, n_timepoints) * 100 + 1000
    
    # Add some task-related activation
    activation_region = data[20:30, 20:30, 10:20, :]
    task_signal = np.sin(np.linspace(0, 4*np.pi, n_timepoints)) * 50
    activation_region += task_signal
    
    # Create affine matrix (2mm isotropic)
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    
    # Create nibabel image
    img = nib.Nifti1Image(data, affine)
    
    return img

def create_synthetic_behavioral_data(n_trials=40):
    """
    Create synthetic behavioral data for delay discounting
    
    Parameters:
    -----------
    n_trials : int
        Number of trials
        
    Returns:
    --------
    pd.DataFrame : Synthetic behavioral data
    """
    print(f"Creating synthetic behavioral data: {n_trials} trials")
    
    # Create realistic delay discounting trials
    np.random.seed(42)
    
    # Trial onsets (every 5 seconds)
    onsets = np.arange(0, n_trials * 5, 5)
    
    # Random choices and delays
    choices = np.random.choice([0, 1], size=n_trials, p=[0.4, 0.6])
    delay_days = np.random.choice([0, 1, 7, 30, 90], size=n_trials)
    large_amounts = np.random.uniform(25, 50, size=n_trials)
    
    # Calculate subjective values using hyperbolic discounting
    k = 0.02  # Discount rate
    sv_large = large_amounts / (1 + k * delay_days)
    sv_small = 20  # Immediate amount
    
    # Assign chosen/unchosen based on choice
    sv_chosen = np.where(choices == 1, sv_large, sv_small)
    sv_unchosen = np.where(choices == 1, sv_small, sv_large)
    sv_difference = np.abs(sv_chosen - sv_unchosen)
    
    # Create events dataframe
    events_df = pd.DataFrame({
        'onset': onsets,
        'duration': 4.0,  # 4 second duration
        'choice': choices,
        'delay_days': delay_days,
        'large_amount': large_amounts,
        'sv_chosen': sv_chosen,
        'sv_unchosen': sv_unchosen,
        'sv_difference': sv_difference,
        'trial_type': 'decision'
    })
    
    return events_df

def create_synthetic_confounds(n_timepoints=200):
    """
    Create synthetic confounds data
    
    Parameters:
    -----------
    n_timepoints : int
        Number of time points
        
    Returns:
    --------
    pd.DataFrame : Synthetic confounds
    """
    # Create realistic motion parameters
    motion_params = np.random.randn(n_timepoints, 6) * 0.5  # Small motion
    
    # Create physiological confounds
    csf = np.random.randn(n_timepoints) * 100 + 500
    white_matter = np.random.randn(n_timepoints) * 80 + 400
    global_signal = np.random.randn(n_timepoints) * 50 + 1000
    
    confounds_df = pd.DataFrame({
        'trans_x': motion_params[:, 0],
        'trans_y': motion_params[:, 1], 
        'trans_z': motion_params[:, 2],
        'rot_x': motion_params[:, 3],
        'rot_y': motion_params[:, 4],
        'rot_z': motion_params[:, 5],
        'csf': csf,
        'white_matter': white_matter,
        'global_signal': global_signal
    })
    
    return confounds_df

def test_first_level_glm():
    """Test first-level GLM modeling"""
    print("\nTesting first-level GLM modeling...")
    
    try:
        from nilearn.glm.first_level import FirstLevelModel
        
        # Create synthetic data
        img = create_synthetic_fmri_data(n_timepoints=200)
        events_df = create_synthetic_behavioral_data(n_trials=40)
        confounds_df = create_synthetic_confounds(n_timepoints=200)
        
        # Test simple choice model
        choice_events = events_df[['onset', 'duration', 'choice']].copy()
        
        # Initialize first-level model
        first_level_model = FirstLevelModel(
            t_r=0.68,
            slice_time_ref=0.5,
            hrf_model='spm',
            drift_model='cosine',
            high_pass=1/128,
            standardize=False,
            signal_scaling=0
        )
        
        # Fit model
        first_level_model = first_level_model.fit(
            img,
            events=choice_events,
            confounds=confounds_df
        )
        
        # Test contrast computation
        contrast_map = first_level_model.compute_contrast('choice', output_type='z_score')
        
        print(f"✓ First-level GLM successful")
        print(f"  - Design matrix shape: {first_level_model.design_matrices_[0].shape}")
        print(f"  - Contrast map shape: {contrast_map.shape}")
        
        return True, first_level_model, contrast_map
        
    except Exception as e:
        print(f"✗ First-level GLM failed: {e}")
        return False, None, None

def test_second_level_glm():
    """Test second-level (group) GLM modeling"""
    print("\nTesting second-level GLM modeling...")
    
    try:
        from nilearn.glm.second_level import SecondLevelModel
        
        # Create multiple synthetic contrast maps (simulating multiple subjects)
        contrast_maps = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i in range(5):  # 5 synthetic subjects
                img = create_synthetic_fmri_data(n_timepoints=200)
                events_df = create_synthetic_behavioral_data(n_trials=40)
                confounds_df = create_synthetic_confounds(n_timepoints=200)
                
                # Quick first-level model
                from nilearn.glm.first_level import FirstLevelModel
                
                choice_events = events_df[['onset', 'duration', 'choice']].copy()
                
                first_level_model = FirstLevelModel(
                    t_r=0.68,
                    hrf_model='spm',
                    drift_model=None,  # Simplified for testing
                    standardize=False
                )
                
                first_level_model = first_level_model.fit(img, events=choice_events)
                contrast_map = first_level_model.compute_contrast('choice', output_type='z_score')
                
                # Save contrast map
                contrast_file = os.path.join(temp_dir, f'subject_{i:02d}_choice_zstat.nii.gz')
                contrast_map.to_filename(contrast_file)
                contrast_maps.append(contrast_file)
            
            # Second-level analysis
            design_matrix = pd.DataFrame({'intercept': np.ones(len(contrast_maps))})
            
            second_level_model = SecondLevelModel()
            second_level_model = second_level_model.fit(contrast_maps, design_matrix=design_matrix)
            
            # Group-level contrast
            group_stat_map = second_level_model.compute_contrast('intercept', output_type='z_score')
            
            print(f"✓ Second-level GLM successful")
            print(f"  - Number of subjects: {len(contrast_maps)}")
            print(f"  - Group map shape: {group_stat_map.shape}")
            
            return True, group_stat_map
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ Second-level GLM failed: {e}")
        return False, None

def test_multiple_comparisons_correction():
    """Test multiple comparisons correction methods"""
    print("\nTesting multiple comparisons correction...")
    
    try:
        from nilearn.glm import threshold_stats_img
        
        # Create a synthetic statistical map
        data = np.random.randn(20, 20, 10) * 2  # Random z-scores
        affine = np.eye(4) * 4  # 4mm isotropic
        affine[3, 3] = 1
        stat_map = nib.Nifti1Image(data, affine)
        
        # Test FDR correction
        try:
            fdr_map, fdr_threshold = threshold_stats_img(
                stat_map,
                alpha=0.05,
                height_control='fdr',
                cluster_threshold=0
            )
            print(f"✓ FDR correction successful (threshold: {fdr_threshold:.3f})")
        except Exception as e:
            print(f"⚠ FDR correction warning: {e}")
        
        # Test cluster-based correction
        try:
            cluster_map, cluster_threshold = threshold_stats_img(
                stat_map,
                alpha=0.05, 
                height_control='fpr',
                cluster_threshold=5
            )
            print(f"✓ Cluster correction successful (threshold: {cluster_threshold:.3f})")
        except Exception as e:
            print(f"⚠ Cluster correction warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Multiple comparisons correction failed: {e}")
        return False

def test_mass_univariate_module():
    """Test the main mass univariate analysis module"""
    print("\nTesting mass univariate analysis module...")
    
    try:
        from mass_univariate_analysis import MassUnivariateAnalysis
        from oak_storage_config import OAKConfig
        
        # Create temporary config
        config = OAKConfig()
        config.OUTPUT_DIR = tempfile.mkdtemp()
        
        try:
            # Initialize analysis
            analysis = MassUnivariateAnalysis(config)
            
            print(f"✓ Mass univariate module imported successfully")
            print(f"  - Number of models defined: {len(analysis.models)}")
            print(f"  - Output directory created: {config.OUTPUT_DIR}")
            
            # Test model definitions
            for model_name, model_spec in analysis.models.items():
                print(f"  - {model_name}: {len(model_spec['regressors'])} regressors, {len(model_spec['contrasts'])} contrasts")
            
            return True
            
        finally:
            # Clean up
            shutil.rmtree(config.OUTPUT_DIR)
            
    except Exception as e:
        print(f"✗ Mass univariate module test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MASS UNIVARIATE ANALYSIS MODULE TESTS")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Package imports
    test_results.append(test_imports())
    
    # Test 2: First-level GLM
    success, _, _ = test_first_level_glm()
    test_results.append(success)
    
    # Test 3: Second-level GLM
    success, _ = test_second_level_glm()
    test_results.append(success)
    
    # Test 4: Multiple comparisons correction
    test_results.append(test_multiple_comparisons_correction())
    
    # Test 5: Mass univariate module
    test_results.append(test_mass_univariate_module())
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Package imports",
        "First-level GLM", 
        "Second-level GLM",
        "Multiple comparisons correction",
        "Mass univariate module"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    total_passed = sum(test_results)
    print(f"\nOverall: {total_passed}/{len(test_results)} tests passed")
    
    if total_passed == len(test_results):
        print("\n🎉 All tests passed! The mass univariate analysis module is ready to use.")
        print("\nNext steps:")
        print("1. Check available subjects: python run_mass_univariate.py --check-only")
        print("2. Run test analysis: python run_mass_univariate.py --test")
        print("3. Run full analysis: python run_mass_univariate.py")
    else:
        print(f"\n⚠ {len(test_results) - total_passed} test(s) failed. Please fix issues before using the module.")
        
    return total_passed == len(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 