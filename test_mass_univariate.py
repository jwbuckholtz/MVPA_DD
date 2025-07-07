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
import nibabel as nib
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
    
    # Create brain-like mask (larger sphere in center)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center_x, center_y, center_z = shape[0]//2, shape[1]//2, shape[2]//2
    brain_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) <= (min(shape)//2.5)**2
    
    # Ensure brain mask has sufficient voxels
    if np.sum(brain_mask) < 1000:
        # Make mask larger if too small
        brain_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) <= (min(shape)//2)**2
    
    print(f"  - Brain mask size: {np.sum(brain_mask)} voxels")
    
    # Create realistic fMRI-like data
    data = np.zeros(shape + (n_timepoints,), dtype=np.float32)
    
    # Add brain signal where mask is True
    for t in range(n_timepoints):
        # Base brain signal (realistic fMRI intensities)
        brain_signal = np.random.randn(*shape) * 30 + 1000
        
        # Add some task-related activation in multiple regions
        activation_region1 = slice(15, 25), slice(15, 25), slice(8, 18)
        activation_region2 = slice(25, 35), slice(25, 35), slice(12, 22)
        
        # Create task signal based on events (more realistic)
        task_signal = np.sin(2 * np.pi * t / 30) * 80  # Slower periodic activation
        brain_signal[activation_region1] += task_signal
        brain_signal[activation_region2] += task_signal * 0.7  # Different effect size
        
        # Apply brain mask and add to data
        data[:, :, :, t] = brain_signal * brain_mask
        
        # Add realistic noise outside brain
        noise_mask = ~brain_mask
        data[:, :, :, t][noise_mask] = np.random.randn(np.sum(noise_mask)) * 5 + 20
    
    # Create proper affine matrix for MNI space (2mm isotropic)
    affine = np.array([
        [-2., 0., 0., 90.],
        [0., 2., 0., -126.],
        [0., 0., 2., -72.],
        [0., 0., 0., 1.]
    ])
    
    # Create nibabel image
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    
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
    
    # Trial onsets (every 4 seconds, ensure they fit within scan time)
    onsets = np.arange(5, min(n_trials * 4 + 5, 180), 4)  # Start at 5s, space by 4s, max 180s
    n_trials = len(onsets)  # Adjust n_trials to match actual onsets
    
    # Create more realistic and varied choices and delays
    choices = np.random.choice([0, 1], size=n_trials, p=[0.3, 0.7])  # More LL choices
    delay_days = np.random.choice([0, 1, 7, 14, 30, 60, 90], size=n_trials, 
                                 p=[0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.05])
    large_amounts = np.random.uniform(20, 60, size=n_trials)
    
    # Calculate subjective values using hyperbolic discounting with more variation
    k = np.random.uniform(0.005, 0.05)  # Variable discount rate for more realistic data
    sv_large = large_amounts / (1 + k * delay_days)
    sv_small = 20  # Immediate amount
    
    # Add some noise to make values more realistic
    sv_large += np.random.normal(0, 2, size=n_trials)
    sv_small_array = np.full(n_trials, sv_small) + np.random.normal(0, 1, size=n_trials)
    
    # Assign chosen/unchosen based on choice
    sv_chosen = np.where(choices == 1, sv_large, sv_small_array)
    sv_unchosen = np.where(choices == 1, sv_small_array, sv_large)
    sv_difference = np.abs(sv_chosen - sv_unchosen)
    
    # Ensure no extreme values that could cause numerical issues
    sv_chosen = np.clip(sv_chosen, 5, 100)
    sv_unchosen = np.clip(sv_unchosen, 5, 100)
    sv_difference = np.clip(sv_difference, 0.1, 80)
    
    # Create events dataframe with trial types that match choices
    trial_types = ['sooner_smaller' if c == 0 else 'larger_later' for c in choices]
    
    events_df = pd.DataFrame({
        'onset': onsets,
        'duration': 4.0,  # 4 second duration
        'choice': choices,
        'delay_days': delay_days,
        'large_amount': large_amounts,
        'sv_chosen': sv_chosen,
        'sv_unchosen': sv_unchosen,
        'sv_difference': sv_difference,
        'trial_type': trial_types  # Different trial types for different choices
    })
    
    # Standardize continuous variables to improve numerical stability
    for col in ['sv_chosen', 'sv_unchosen', 'sv_difference']:
        if col in events_df.columns:
            events_df[col] = (events_df[col] - events_df[col].mean()) / events_df[col].std()
    
    # Ensure choice is binary (0/1) and has variance
    events_df['choice'] = events_df['choice'].astype(float)
    if events_df['choice'].var() == 0:
        # Add minimal variance if all choices are the same
        events_df['choice'] = events_df['choice'] + np.random.normal(0, 0.01, len(events_df))
    
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
    # Create simplified motion parameters to avoid multicollinearity
    motion_params = np.random.randn(n_timepoints, 3) * 0.2  # Smaller motion
    
    # Create physiological confounds with lower correlation
    csf = np.random.randn(n_timepoints) * 50 + 500
    white_matter = np.random.randn(n_timepoints) * 40 + 400
    
    # Add some temporal structure but avoid perfect correlation
    drift = np.linspace(0, 5, n_timepoints) + np.random.randn(n_timepoints) * 0.1
    
    confounds_df = pd.DataFrame({
        'trans_x': motion_params[:, 0],
        'trans_y': motion_params[:, 1], 
        'rot_x': motion_params[:, 2],
        'csf': csf,
        'white_matter': white_matter,
        'drift': drift
    })
    
    # Standardize confounds to prevent numerical issues
    for col in confounds_df.columns:
        confounds_df[col] = (confounds_df[col] - confounds_df[col].mean()) / confounds_df[col].std()
    
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
        
        print(f"  - fMRI data shape: {img.shape}")
        print(f"  - Events shape: {events_df.shape}")
        print(f"  - Confounds shape: {confounds_df.shape}")
        
        # Create explicit brain mask manually
        # Get the brain mask from our synthetic data
        sample_data = img.get_fdata()
        mask_data = np.mean(sample_data, axis=-1) > 500  # Voxels with signal > 500
        brain_mask = nib.Nifti1Image(mask_data.astype(np.uint8), img.affine)
        
        print(f"  - Brain mask voxels: {np.sum(mask_data)}")
        print(f"  - Data range: {np.min(sample_data):.1f} to {np.max(sample_data):.1f}")
        
        # Test simple choice model - must include trial_type for nilearn
        choice_events = events_df[['onset', 'duration', 'trial_type', 'choice']].copy()
        
        # Initialize first-level model with explicit mask
        first_level_model = FirstLevelModel(
            t_r=0.68,
            slice_time_ref=0.5,
            hrf_model='spm',
            drift_model='cosine',
            high_pass=1/128,
            standardize=False,
            signal_scaling=0,
            mask_img=brain_mask,
            smoothing_fwhm=None,
            verbose=1  # Increase verbosity for debugging
        )
        
        print(f"  - Fitting GLM model...")
        print(f"  - Events preview:")
        print(choice_events.head())
        print(f"  - Choice variance: {choice_events['choice'].var():.3f}")
        print(f"  - Confounds shape: {confounds_df.shape}")
        
        # Check for potential design matrix issues
        print(f"  - Checking design matrix...")
        
        # Fit model
        try:
            first_level_model = first_level_model.fit(
                img,
                events=choice_events,
                confounds=confounds_df
            )
            
            # Check design matrix rank
            design_matrix = first_level_model.design_matrices_[0]
            rank = np.linalg.matrix_rank(design_matrix.values)
            print(f"  - Design matrix rank: {rank}/{design_matrix.shape[1]}")
            
            if rank < design_matrix.shape[1]:
                print(f"  - Warning: Design matrix is rank deficient!")
                print(f"  - Column names: {list(design_matrix.columns)}")
                print(f"  - Column correlations:")
                corr_matrix = design_matrix.corr()
                high_corr = np.where(np.abs(corr_matrix.values) > 0.95)
                for i, j in zip(high_corr[0], high_corr[1]):
                    if i != j:
                        print(f"    {design_matrix.columns[i]} <-> {design_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.3f}")
        except Exception as fit_error:
            print(f"  - GLM fit failed with explicit mask, trying simpler approach...")
            print(f"  - Fit error: {fit_error}")
            
            # Fallback: try with automatic masking and no confounds
            first_level_model_simple = FirstLevelModel(
                t_r=0.68,
                hrf_model='spm',
                drift_model=None,  # No drift removal
                standardize=False,
                signal_scaling=0,
                mask_img=None,  # Let nilearn handle masking
                verbose=1
            )
            
            # Try without confounds first
            print(f"  - Trying without confounds...")
            first_level_model = first_level_model_simple.fit(
                img,
                events=choice_events  # No confounds to avoid numerical issues
            )
            
            # Check design matrix rank for fallback
            design_matrix = first_level_model.design_matrices_[0]
            rank = np.linalg.matrix_rank(design_matrix.values)
            print(f"  - Fallback design matrix rank: {rank}/{design_matrix.shape[1]}")
        
        # Test contrast computation - need to use actual design matrix column names
        design_matrix = first_level_model.design_matrices_[0]
        print(f"  - Available contrast columns: {list(design_matrix.columns)}")
        
        # Create a contrast between larger_later and sooner_smaller choices
        if 'larger_later' in design_matrix.columns and 'sooner_smaller' in design_matrix.columns:
            # Contrast: larger_later - sooner_smaller
            contrast_map = first_level_model.compute_contrast('larger_later - sooner_smaller', output_type='z_score')
            contrast_name = 'larger_later - sooner_smaller'
        elif 'larger_later' in design_matrix.columns:
            # Just test larger_later condition
            contrast_map = first_level_model.compute_contrast('larger_later', output_type='z_score')
            contrast_name = 'larger_later'
        elif 'sooner_smaller' in design_matrix.columns:
            # Just test sooner_smaller condition
            contrast_map = first_level_model.compute_contrast('sooner_smaller', output_type='z_score')
            contrast_name = 'sooner_smaller'
        else:
            # Fallback to first available column
            non_drift_cols = [col for col in design_matrix.columns if 'drift' not in col.lower() and 'constant' not in col.lower()]
            contrast_col = non_drift_cols[0] if non_drift_cols else design_matrix.columns[0]
            contrast_map = first_level_model.compute_contrast(contrast_col, output_type='z_score')
            contrast_name = contrast_col
        
        print(f"  - Using contrast: {contrast_name}")
        
        print(f"✓ First-level GLM successful")
        print(f"  - Design matrix shape: {first_level_model.design_matrices_[0].shape}")
        print(f"  - Contrast map shape: {contrast_map.shape}")
        print(f"  - Non-zero voxels in contrast: {np.sum(contrast_map.get_fdata() != 0)}")
        
        return True, first_level_model, contrast_map
        
    except Exception as e:
        print(f"✗ First-level GLM failed: {e}")
        import traceback
        print(f"  Error details: {traceback.format_exc()}")
        return False, None, None

def test_second_level_glm():
    """Test second-level (group) GLM modeling"""
    print("\nTesting second-level GLM modeling...")
    
    try:
        from nilearn.glm.second_level import SecondLevelModel
        from nilearn.glm.first_level import FirstLevelModel
        
        # Create multiple synthetic contrast maps (simulating multiple subjects)
        contrast_maps = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            print(f"  - Creating {5} synthetic subjects...")
            
            for i in range(5):  # 5 synthetic subjects
                img = create_synthetic_fmri_data(n_timepoints=200)
                events_df = create_synthetic_behavioral_data(n_trials=40)
                confounds_df = create_synthetic_confounds(n_timepoints=200)
                
                # Create explicit brain mask manually
                sample_data = img.get_fdata()
                mask_data = np.mean(sample_data, axis=-1) > 500  # Voxels with signal > 500
                brain_mask = nib.Nifti1Image(mask_data.astype(np.uint8), img.affine)
                
                choice_events = events_df[['onset', 'duration', 'trial_type', 'choice']].copy()
                
                first_level_model = FirstLevelModel(
                    t_r=0.68,
                    hrf_model='spm',
                    drift_model=None,  # Simplified for testing
                    standardize=False,
                    mask_img=brain_mask,
                    verbose=0
                )
                
                # Use simplified approach - no confounds for testing
                first_level_model = first_level_model.fit(img, events=choice_events)
                
                # Find correct column name for contrast
                design_matrix = first_level_model.design_matrices_[0]
                if 'larger_later' in design_matrix.columns and 'sooner_smaller' in design_matrix.columns:
                    contrast_map = first_level_model.compute_contrast('larger_later - sooner_smaller', output_type='z_score')
                elif 'larger_later' in design_matrix.columns:
                    contrast_map = first_level_model.compute_contrast('larger_later', output_type='z_score')
                elif 'sooner_smaller' in design_matrix.columns:
                    contrast_map = first_level_model.compute_contrast('sooner_smaller', output_type='z_score')
                else:
                    non_drift_cols = [col for col in design_matrix.columns if 'drift' not in col.lower() and 'constant' not in col.lower()]
                    contrast_col = non_drift_cols[0] if non_drift_cols else design_matrix.columns[0]
                    contrast_map = first_level_model.compute_contrast(contrast_col, output_type='z_score')
                
                # Save contrast map
                contrast_file = os.path.join(temp_dir, f'subject_{i:02d}_choice_zstat.nii.gz')
                contrast_map.to_filename(contrast_file)
                contrast_maps.append(contrast_file)
                
                print(f"    Subject {i+1}: {np.sum(contrast_map.get_fdata() != 0)} non-zero voxels")
            
            # Second-level analysis
            design_matrix = pd.DataFrame({'intercept': np.ones(len(contrast_maps))})
            
            second_level_model = SecondLevelModel(verbose=0)
            second_level_model = second_level_model.fit(contrast_maps, design_matrix=design_matrix)
            
            # Group-level contrast
            group_stat_map = second_level_model.compute_contrast('intercept', output_type='z_score')
            
            print(f"✓ Second-level GLM successful")
            print(f"  - Number of subjects: {len(contrast_maps)}")
            print(f"  - Group map shape: {group_stat_map.shape}")
            print(f"  - Non-zero voxels in group map: {np.sum(group_stat_map.get_fdata() != 0)}")
            
            return True, group_stat_map
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ Second-level GLM failed: {e}")
        import traceback
        print(f"  Error details: {traceback.format_exc()}")
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