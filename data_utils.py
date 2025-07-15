#!/usr/bin/env python3
"""
Data Loading and Management Utilities for MVPA Pipeline
======================================================

This module provides centralized data loading and validation functionality.

Key distinction:
- **fMRIPrep preprocessing**: Already completed (skull stripping, normalization, motion correction)
- **Our data preparation**: Loading + standardization + masking + confound removal

Functions for:
- Loading fMRIPrep-preprocessed fMRI data (raw, smoothed)
- Loading behavioral data with validation
- Loading confounds and applying denoising
- ROI mask validation and time series extraction
- Subject discovery and data integrity checking
- Quality control and data validation

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union

# Neuroimaging
import nibabel as nib
from nilearn import image, masking
from nilearn.input_data import NiftiMasker

# Configuration
from oak_storage_config import OAKConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataError(Exception):
    """Custom exception for data-related errors"""
    pass

class DataValidator:
    """Class for validating data integrity and quality"""
    
    def __init__(self, config=None):
        self.config = config if config else OAKConfig()
    
    def validate_behavioral_data(self, df: pd.DataFrame, subject_id: str) -> Dict:
        """
        Validate behavioral data quality
        
        Parameters:
        -----------
        df : pd.DataFrame
            Behavioral data
        subject_id : str
            Subject identifier
            
        Returns:
        --------
        dict : Validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        # Check required columns
        required_cols = ['onset', 'response', 'delay_days', 'large_amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['valid'] = False
        
        if not results['valid']:
            return results
        
        # Calculate quality metrics
        results['metrics']['n_trials'] = len(df)
        results['metrics']['accuracy'] = np.mean(~df['response'].isna())
        results['metrics']['mean_rt'] = df.get('reaction_time', pd.Series([np.nan])).mean()
        
        # Check behavioral accuracy
        if results['metrics']['accuracy'] < self.config.MIN_ACCURACY:
            results['warnings'].append(f"Low accuracy: {results['metrics']['accuracy']:.3f}")
        
        # Check reaction times
        if 'reaction_time' in df.columns:
            high_rt_trials = (df['reaction_time'] > self.config.MAX_RT).sum()
            if high_rt_trials > 0:
                results['warnings'].append(f"{high_rt_trials} trials with RT > {self.config.MAX_RT}s")
        
        # Check for missing responses
        missing_responses = df['response'].isna().sum()
        if missing_responses > len(df) * 0.2:  # > 20% missing
            results['warnings'].append(f"High missing responses: {missing_responses}/{len(df)}")
        
        return results
    
    def validate_fmri_data(self, img_path: str, subject_id: str) -> Dict:
        """
        Validate fMRI data quality
        
        Parameters:
        -----------
        img_path : str
            Path to fMRI image
        subject_id : str
            Subject identifier
            
        Returns:
        --------
        dict : Validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            img = nib.load(img_path)
            data = img.get_fdata()
            
            # Basic metrics
            results['metrics']['shape'] = img.shape
            results['metrics']['voxel_size'] = img.header.get_zooms()[:3]
            results['metrics']['tr'] = img.header.get_zooms()[3] if len(img.shape) > 3 else None
            results['metrics']['mean_intensity'] = np.mean(data[data > 0])
            results['metrics']['std_intensity'] = np.std(data[data > 0])
            
            # Check for obvious issues
            if np.any(np.isnan(data)):
                results['warnings'].append("NaN values detected in fMRI data")
            
            if np.any(np.isinf(data)):
                results['errors'].append("Infinite values detected in fMRI data")
                results['valid'] = False
            
            # Check intensity range (typical for fMRI)
            if results['metrics']['mean_intensity'] < 100 or results['metrics']['mean_intensity'] > 10000:
                results['warnings'].append(f"Unusual intensity range: mean={results['metrics']['mean_intensity']:.1f}")
            
        except Exception as e:
            results['errors'].append(f"Failed to load fMRI data: {str(e)}")
            results['valid'] = False
        
        return results

class SubjectManager:
    """Class for discovering and managing subjects"""
    
    def __init__(self, config=None):
        self.config = config if config else OAKConfig()
        self.validator = DataValidator(config)
    
    def get_available_subjects(self, require_both: bool = True) -> List[str]:
        """
        Get list of subjects with available data
        
        Parameters:
        -----------
        require_both : bool
            If True, require both fMRI and behavioral data
            
        Returns:
        --------
        list : Available subject IDs
        """
        subjects = []
        fmriprep_dir = Path(self.config.FMRIPREP_DIR)
        
        if not fmriprep_dir.exists():
            logger.warning(f"fMRIPrep directory not found: {fmriprep_dir}")
            return subjects
        
        # Look for subject directories
        for subject_dir in fmriprep_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
                subject_id = subject_dir.name
                
                has_fmri = self._check_fmri_exists(subject_id)
                has_behavior = self._check_behavioral_exists(subject_id)
                
                if require_both:
                    if has_fmri and has_behavior:
                        subjects.append(subject_id)
                        logger.debug(f"Found complete data for {subject_id}")
                    else:
                        logger.debug(f"Incomplete data for {subject_id}: fMRI={has_fmri}, behavior={has_behavior}")
                else:
                    if has_fmri or has_behavior:
                        subjects.append(subject_id)
        
        logger.info(f"Found {len(subjects)} subjects with {'complete' if require_both else 'available'} data")
        return sorted(subjects)
    
    def _check_fmri_exists(self, subject_id: str) -> bool:
        """Check if fMRI data exists for subject"""
        func_dir = Path(self.config.FMRIPREP_DIR) / subject_id / "ses-2" / "func"
        if not func_dir.exists():
            return False
        
        pattern = f"{subject_id}_ses-2_task-discountFix_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        fmri_files = list(func_dir.glob(pattern))
        return len(fmri_files) > 0
    
    def _check_behavioral_exists(self, subject_id: str) -> bool:
        """Check if behavioral data exists for subject"""
        behavior_file = Path(self.config.BEHAVIOR_DIR) / f"{subject_id}_discountFix_events.tsv"
        return behavior_file.exists()
    
    def get_subject_summary(self, subject_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary of data availability for subjects
        
        Parameters:
        -----------
        subject_ids : list, optional
            List of subject IDs to check. If None, check all available.
            
        Returns:
        --------
        pd.DataFrame : Summary of data availability
        """
        if subject_ids is None:
            subject_ids = self.get_available_subjects(require_both=False)
        
        summary_data = []
        
        for subject_id in subject_ids:
            row = {'subject_id': subject_id}
            
            # Check fMRI data
            row['has_fmri'] = self._check_fmri_exists(subject_id)
            
            # Check behavioral data
            row['has_behavior'] = self._check_behavioral_exists(subject_id)
            
            # Load and validate if available
            if row['has_behavior']:
                try:
                    behavioral_data = load_behavioral_data(subject_id, self.config)
                    validation = self.validator.validate_behavioral_data(behavioral_data, subject_id)
                    row['behavior_valid'] = validation['valid']
                    row['n_trials'] = validation['metrics'].get('n_trials', 0)
                    row['accuracy'] = validation['metrics'].get('accuracy', np.nan)
                except Exception as e:
                    row['behavior_valid'] = False
                    row['n_trials'] = 0
                    row['accuracy'] = np.nan
            else:
                row['behavior_valid'] = False
                row['n_trials'] = 0
                row['accuracy'] = np.nan
            
            if row['has_fmri']:
                try:
                    fmri_path = get_fmri_path(subject_id, self.config)
                    validation = self.validator.validate_fmri_data(fmri_path, subject_id)
                    row['fmri_valid'] = validation['valid']
                    row['fmri_shape'] = str(validation['metrics'].get('shape', 'Unknown'))
                except Exception as e:
                    row['fmri_valid'] = False
                    row['fmri_shape'] = 'Unknown'
            else:
                row['fmri_valid'] = False
                row['fmri_shape'] = 'Unknown'
            
            row['complete'] = row['has_fmri'] and row['has_behavior'] and row['behavior_valid'] and row['fmri_valid']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

# Core data loading functions
def load_behavioral_data(subject_id: str, config: Optional[OAKConfig] = None, 
                        validate: bool = True, compute_sv: bool = True) -> pd.DataFrame:
    """
    Load and process behavioral data for a subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    config : OAKConfig, optional
        Configuration object
    validate : bool
        Whether to validate data quality
    compute_sv : bool
        Whether to compute subjective values
        
    Returns:
    --------
    pd.DataFrame : Processed behavioral data
    """
    if config is None:
        config = OAKConfig()
    
    behavior_file = Path(config.BEHAVIOR_DIR) / f"{subject_id}_discountFix_events.tsv"
    
    if not behavior_file.exists():
        raise DataError(f"Behavioral file not found for {subject_id}: {behavior_file}")
    
    logger.debug(f"Loading behavioral data for {subject_id}")
    
    try:
        # Load behavioral data
        events_df = pd.read_csv(behavior_file, sep='\t')
        
        # Basic processing
        events_df['choice'] = events_df['response'].map({1: 0, 2: 1})  # 0=SS, 1=LL
        events_df['trial_type'] = events_df['choice'].map({0: 'sooner_smaller', 1: 'larger_later'})
        
        # Set duration if not present
        if 'duration' not in events_df.columns:
            events_df['duration'] = 4.0  # 4 second trials
        
        # Compute subjective values if requested
        if compute_sv:
            events_df = _compute_subjective_values(events_df, subject_id, config)
        
        # Validate if requested
        if validate:
            validator = DataValidator(config)
            validation = validator.validate_behavioral_data(events_df, subject_id)
            if not validation['valid']:
                logger.warning(f"Behavioral data validation failed for {subject_id}: {validation['errors']}")
            if validation['warnings']:
                logger.warning(f"Behavioral data warnings for {subject_id}: {validation['warnings']}")
        
        # Remove invalid trials
        valid_mask = (~events_df['choice'].isna())
        events_df = events_df[valid_mask].reset_index(drop=True)
        
        logger.info(f"Loaded {len(events_df)} valid trials for {subject_id}")
        return events_df
        
    except Exception as e:
        raise DataError(f"Failed to load behavioral data for {subject_id}: {str(e)}")

def _compute_subjective_values(events_df: pd.DataFrame, subject_id: str, config: OAKConfig) -> pd.DataFrame:
    """
    Compute subjective values using hyperbolic discounting
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        Events dataframe
    subject_id : str
        Subject identifier
    config : OAKConfig
        Configuration object
        
    Returns:
    --------
    pd.DataFrame : Events with computed subjective values
    """
    # Import behavioral analysis here to avoid circular imports
    from delay_discounting_mvpa_pipeline import BehavioralAnalysis
    
    behavioral_analysis = BehavioralAnalysis(config)
    
    # Extract choice data for fitting
    choices = events_df['choice'].values
    large_amounts = events_df['large_amount'].values
    delays = events_df['delay_days'].values
    
    # Remove NaN values for fitting
    valid_idx = ~(np.isnan(choices) | np.isnan(large_amounts) | np.isnan(delays))
    
    if np.sum(valid_idx) < 5:  # Need minimum trials for fitting
        logger.warning(f"Insufficient valid trials for discount rate fitting: {subject_id}")
        # Use default values
        events_df['sv_chosen'] = events_df['large_amount'] / 2
        events_df['sv_unchosen'] = events_df['large_amount'] / 2
        events_df['sv_difference'] = np.abs(events_df['sv_chosen'] - events_df['sv_unchosen'])
        events_df['svchosen_unchosen'] = events_df['sv_chosen'] - events_df['sv_unchosen']
        return events_df
    
    # Fit discount rate
    fit_result = behavioral_analysis.fit_discount_rate(
        choices[valid_idx], large_amounts[valid_idx], delays[valid_idx]
    )
    
    if fit_result['success']:
        k = fit_result['k']
        logger.debug(f"Fitted discount rate k={k:.4f} for {subject_id}")
        
        # Calculate subjective values
        sv_large = behavioral_analysis.subjective_value(large_amounts, delays, k)
        sv_small = 20  # Immediate reward amount
        
        # Assign chosen/unchosen based on choice
        events_df['sv_chosen'] = np.where(choices == 1, sv_large, sv_small)
        events_df['sv_unchosen'] = np.where(choices == 1, sv_small, sv_large)
        events_df['sv_difference'] = np.abs(events_df['sv_chosen'] - events_df['sv_unchosen'])
        events_df['svchosen_unchosen'] = events_df['sv_chosen'] - events_df['sv_unchosen']
        
    else:
        logger.warning(f"Could not fit discount rate for {subject_id}, using default values")
        # Use fallback values
        events_df['sv_chosen'] = events_df['large_amount'] / 2
        events_df['sv_unchosen'] = events_df['large_amount'] / 2
        events_df['sv_difference'] = np.abs(events_df['sv_chosen'] - events_df['sv_unchosen'])
        events_df['svchosen_unchosen'] = events_df['sv_chosen'] - events_df['sv_unchosen']
    
    return events_df

def get_fmri_path(subject_id: str, config: Optional[OAKConfig] = None, 
                  run: Optional[int] = None) -> str:
    """
    Get path to fMRI data file
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    config : OAKConfig, optional
        Configuration object
    run : int, optional
        Run number (if None, uses first available)
        
    Returns:
    --------
    str : Path to fMRI file
    """
    if config is None:
        config = OAKConfig()
    
    fmri_dir = Path(config.FMRIPREP_DIR) / subject_id / "ses-2" / "func"
    
    if not fmri_dir.exists():
        raise DataError(f"fMRI directory not found for {subject_id}: {fmri_dir}")
    
    if run is not None:
        pattern = f"{subject_id}_ses-2_task-discountFix_run-{run:02d}_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    else:
        pattern = f"{subject_id}_ses-2_task-discountFix_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    
    fmri_files = list(fmri_dir.glob(pattern))
    
    if not fmri_files:
        raise DataError(f"No fMRI files found for {subject_id} with pattern: {pattern}")
    
    # Return first file (or specific run if requested)
    fmri_file = str(fmri_files[0])
    logger.debug(f"Found fMRI file for {subject_id}: {Path(fmri_file).name}")
    
    return fmri_file

def load_fmri_data(subject_id: str, config: Optional[OAKConfig] = None,
                   smoothed: bool = False, validate: bool = True) -> nib.Nifti1Image:
    """
    Load fMRI data for a subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    config : OAKConfig, optional
        Configuration object
    smoothed : bool
        Whether to load smoothed data (if available)
    validate : bool
        Whether to validate data quality
        
    Returns:
    --------
    nibabel.Nifti1Image : fMRI data
    """
    if config is None:
        config = OAKConfig()
    
    if smoothed:
        # Look for smoothed data first
        smoothed_dir = Path(config.OUTPUT_DIR) / "mass_univariate" / "smoothed_data"
        smoothed_file = smoothed_dir / f"{subject_id}_ses-2_task-discountFix_smoothed-4.0mm_bold.nii.gz"
        
        if smoothed_file.exists():
            logger.debug(f"Loading smoothed fMRI data for {subject_id}")
            img = nib.load(str(smoothed_file))
        else:
            logger.debug(f"Smoothed data not found for {subject_id}, loading raw data")
            fmri_file = get_fmri_path(subject_id, config)
            img = nib.load(fmri_file)
    else:
        fmri_file = get_fmri_path(subject_id, config)
        img = nib.load(fmri_file)
    
    # Validate if requested
    if validate:
        validator = DataValidator(config)
        validation = validator.validate_fmri_data(img.get_filename(), subject_id)
        if not validation['valid']:
            logger.warning(f"fMRI data validation failed for {subject_id}: {validation['errors']}")
        if validation['warnings']:
            logger.warning(f"fMRI data warnings for {subject_id}: {validation['warnings']}")
    
    return img

def load_confounds(subject_id: str, config: Optional[OAKConfig] = None,
                  selected_confounds: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Load fMRIPrep confounds for a subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    config : OAKConfig, optional
        Configuration object
    selected_confounds : list, optional
        Specific confounds to load
        
    Returns:
    --------
    pd.DataFrame or None : Confounds dataframe
    """
    if config is None:
        config = OAKConfig()
    
    # Find confounds file
    fmri_dir = Path(config.FMRIPREP_DIR) / subject_id / "ses-2" / "func"
    pattern = f"{subject_id}_ses-2_task-discountFix_*_desc-confounds_timeseries.tsv"
    
    confounds_files = list(fmri_dir.glob(pattern))
    if not confounds_files:
        logger.warning(f"No confounds file found for {subject_id}")
        return None
    
    confounds_file = confounds_files[0]
    logger.debug(f"Loading confounds for {subject_id}")
    
    try:
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        
        # Select specific confounds if requested
        if selected_confounds is None:
            # Default confound selection
            confound_patterns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                               'csf', 'white_matter', 'global_signal']
            
            # Add motion derivatives if available
            for pattern in ['trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                           'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1']:
                if pattern in confounds_df.columns:
                    confound_patterns.append(pattern)
            
            # Select available columns
            confound_cols = []
            for pattern in confound_patterns:
                matching_cols = [col for col in confounds_df.columns if pattern in col]
                if matching_cols:
                    confound_cols.append(matching_cols[0])  # Take first match
        else:
            confound_cols = [col for col in selected_confounds if col in confounds_df.columns]
        
        if confound_cols:
            result_df = confounds_df[confound_cols].fillna(0)
            logger.debug(f"Loaded {len(confound_cols)} confounds for {subject_id}")
            return result_df
        else:
            logger.warning(f"No suitable confounds found for {subject_id}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load confounds for {subject_id}: {str(e)}")
        return None

def check_mask_exists(mask_path: str) -> bool:
    """
    Check if mask file exists
    
    Parameters:
    -----------
    mask_path : str
        Path to mask file
        
    Returns:
    --------
    bool : True if mask exists
    
    Note:
    -----
    This function has been simplified to only check existence.
    Mask creation is no longer supported - use pre-existing masks on OAK.
    """
    return Path(mask_path).exists()

def load_mask(mask_path: str, validate: bool = True) -> nib.Nifti1Image:
    """
    Load ROI mask
    
    Parameters:
    -----------
    mask_path : str
        Path to mask file
    validate : bool
        Whether to validate mask
        
    Returns:
    --------
    nibabel.Nifti1Image : Mask image
    """
    if not check_mask_exists(mask_path):
        raise DataError(f"Mask file not found: {mask_path}")
    
    try:
        mask_img = nib.load(mask_path)
        
        if validate:
            mask_data = mask_img.get_fdata()
            n_voxels = np.sum(mask_data > 0)
            
            if n_voxels == 0:
                raise DataError(f"Empty mask: {mask_path}")
            
            logger.debug(f"Loaded mask with {n_voxels} voxels: {Path(mask_path).name}")
        
        return mask_img
        
    except Exception as e:
        raise DataError(f"Failed to load mask {mask_path}: {str(e)}")

def extract_roi_timeseries(subject_id: str, roi_name: str, 
                          config: Optional[OAKConfig] = None,
                          smoothed: bool = False, 
                          standardize: bool = True,
                          detrend: bool = True) -> np.ndarray:
    """
    Extract ROI time series for a subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    roi_name : str
        ROI name (must be in config.ROI_MASKS)
    config : OAKConfig, optional
        Configuration object
    smoothed : bool
        Whether to use smoothed data
    standardize : bool
        Whether to standardize time series
    detrend : bool
        Whether to detrend time series
        
    Returns:
    --------
    np.ndarray : Time series data (n_timepoints x n_voxels)
    """
    if config is None:
        config = OAKConfig()
    
    if roi_name not in config.ROI_MASKS:
        raise DataError(f"ROI '{roi_name}' not found in configuration")
    
    # Load fMRI data
    img = load_fmri_data(subject_id, config, smoothed=smoothed)
    
    # Load mask
    mask_path = config.ROI_MASKS[roi_name]
    mask_img = load_mask(mask_path)
    
    # Load confounds
    confounds = load_confounds(subject_id, config)
    
    # Create masker
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=standardize,
        detrend=detrend,
        high_pass=0.01,
        t_r=config.TR,
        memory='nilearn_cache',
        memory_level=1
    )
    
    # Extract time series
    if confounds is not None:
        timeseries = masker.fit_transform(img, confounds=confounds)
    else:
        timeseries = masker.fit_transform(img)
    
    logger.info(f"Extracted {roi_name} time series for {subject_id}: {timeseries.shape}")
    return timeseries

def save_processed_data(data: Dict, output_path: str, subject_id: str = None):
    """
    Save processed data with metadata
    
    Parameters:
    -----------
    data : dict
        Data to save
    output_path : str
        Output file path
    subject_id : str, optional
        Subject identifier for logging
    """
    try:
        # Add metadata
        metadata = {
            'timestamp': pd.Timestamp.now(),
            'subject_id': subject_id,
            'data_keys': list(data.keys())
        }
        
        save_data = {
            'data': data,
            'metadata': metadata
        }
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved processed data to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save data to {output_path}: {str(e)}")
        raise DataError(f"Failed to save data: {str(e)}")

def load_processed_data(input_path: str) -> Tuple[Dict, Dict]:
    """
    Load processed data with metadata
    
    Parameters:
    -----------
    input_path : str
        Input file path
        
    Returns:
    --------
    tuple : (data, metadata)
    """
    try:
        with open(input_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        if isinstance(saved_data, dict) and 'data' in saved_data:
            return saved_data['data'], saved_data.get('metadata', {})
        else:
            # Legacy format
            return saved_data, {}
            
    except Exception as e:
        logger.error(f"Failed to load data from {input_path}: {str(e)}")
        raise DataError(f"Failed to load data: {str(e)}")

# Convenience functions for common operations
def get_complete_subjects(config: Optional[OAKConfig] = None) -> List[str]:
    """Get list of subjects with complete, valid data"""
    manager = SubjectManager(config)
    return manager.get_available_subjects(require_both=True)

def check_data_integrity(subject_ids: Optional[List[str]] = None, 
                        config: Optional[OAKConfig] = None) -> pd.DataFrame:
    """Run comprehensive data integrity check"""
    manager = SubjectManager(config)
    return manager.get_subject_summary(subject_ids)

if __name__ == "__main__":
    # Example usage and testing
    config = OAKConfig()
    manager = SubjectManager(config)
    
    print("Data Utilities Test")
    print("=" * 30)
    
    # Get available subjects
    subjects = manager.get_available_subjects()
    print(f"Found {len(subjects)} subjects with complete data")
    
    if subjects:
        # Test with first subject
        test_subject = subjects[0]
        print(f"\nTesting with subject: {test_subject}")
        
        try:
            # Test behavioral data loading
            behavioral_data = load_behavioral_data(test_subject, config)
            print(f"✓ Loaded {len(behavioral_data)} behavioral trials")
            
            # Test fMRI data loading
            fmri_data = load_fmri_data(test_subject, config)
            print(f"✓ Loaded fMRI data: {fmri_data.shape}")
            
            # Test confounds loading
            confounds = load_confounds(test_subject, config)
            if confounds is not None:
                print(f"✓ Loaded {confounds.shape[1]} confounds")
            else:
                print("⚠ No confounds available")
            
            print("\n✓ All data utilities working correctly!")
            
        except Exception as e:
            print(f"✗ Error testing data utilities: {e}")
    else:
        print("No subjects available for testing") 