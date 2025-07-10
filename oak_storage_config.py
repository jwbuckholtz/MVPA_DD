#!/usr/bin/env python3
"""
Modified configuration to store outputs on OAK storage

This configuration file modifies the original pipeline to store all outputs
on OAK storage for better data management and accessibility.
"""

import os
from pathlib import Path

class OAKConfig:
    """Modified configuration that stores outputs on OAK"""
    
    # Data paths (input - same as original)
    DATA_ROOT = "/oak/stanford/groups/russpold/data/uh2/aim1"
    FMRIPREP_DIR = f"{DATA_ROOT}/derivatives/fmriprep"
    BEHAVIOR_DIR = f"{DATA_ROOT}/behavioral_data/event_files"
    
    # Output directories - NOW ON OAK
    OAK_OUTPUT_ROOT = "/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis"
    OUTPUT_DIR = f"{OAK_OUTPUT_ROOT}/delay_discounting_results"
    BEHAVIOR_OUTPUT = f"{OUTPUT_DIR}/behavioral_analysis"
    MVPA_OUTPUT = f"{OUTPUT_DIR}/mvpa_analysis"
    GEOMETRY_OUTPUT = f"{OUTPUT_DIR}/geometry_analysis"
    
    # Alternative: Store in your personal OAK space
    # USER_OAK_DIR = f"/oak/stanford/groups/russpold/users/{os.getenv('USER', 'your_username')}"
    # OUTPUT_DIR = f"{USER_OAK_DIR}/delay_discounting_mvpa_results"
    
    # Analysis parameters (same as original)
    TR = .68  # Repetition time in seconds
    HEMI_LAG = 0  # Hemodynamic lag in TRs
    
    # ROI masks - PRE-EXISTING masks stored on OAK
    MASKS_DIR = f"{DATA_ROOT}/derivatives/masks"  # Pre-existing masks on OAK
    ROI_MASKS = {
        'striatum': f'{MASKS_DIR}/striatum_mask.nii.gz',
        'dlpfc': f'{MASKS_DIR}/dlpfc_mask.nii.gz',
        'vmpfc': f'{MASKS_DIR}/vmpfc_mask.nii.gz',
        # Add additional pre-existing masks as available
        'left_striatum': f'{MASKS_DIR}/left_striatum_mask.nii.gz',
        'right_striatum': f'{MASKS_DIR}/right_striatum_mask.nii.gz',
        'left_dlpfc': f'{MASKS_DIR}/left_dlpfc_mask.nii.gz',
        'right_dlpfc': f'{MASKS_DIR}/right_dlpfc_mask.nii.gz',
        'acc': f'{MASKS_DIR}/acc_mask.nii.gz',  # Anterior Cingulate Cortex
        'ofc': f'{MASKS_DIR}/ofc_mask.nii.gz',  # Orbitofrontal Cortex
    }
    
    # Core ROI masks (required for analysis)
    CORE_ROI_MASKS = ['striatum', 'dlpfc', 'vmpfc']
    
    # Optional ROI masks (used if available)
    OPTIONAL_ROI_MASKS = ['left_striatum', 'right_striatum', 'left_dlpfc', 'right_dlpfc', 'acc', 'ofc']
    
    # MVPA parameters
    N_JOBS = -1  # Use all available cores
    CV_FOLDS = 5
    N_PERMUTATIONS = 1000
    
    # Quality control
    MIN_ACCURACY = 0.6  # Minimum behavioral accuracy
    MAX_RT = 10.0  # Maximum reaction time in seconds

def setup_oak_directories(config):
    """Create output directories on OAK with proper permissions"""
    
    # Create main output directory
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for directory in [config.BEHAVIOR_OUTPUT, config.MVPA_OUTPUT, config.GEOMETRY_OUTPUT]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set permissions for group access
    try:
        os.chmod(config.OUTPUT_DIR, 0o755)  # rwxr-xr-x
        for directory in [config.BEHAVIOR_OUTPUT, config.MVPA_OUTPUT, config.GEOMETRY_OUTPUT]:
            os.chmod(directory, 0o755)
    except:
        print("Warning: Could not set directory permissions")
    
    print(f"Output directories created on OAK: {config.OUTPUT_DIR}")

# Usage example:
if __name__ == "__main__":
    config = OAKConfig()
    setup_oak_directories(config)
    
    print("OAK storage configuration:")
    print(f"  Input data: {config.DATA_ROOT}")
    print(f"  Output data: {config.OUTPUT_DIR}")
    print(f"  Results will be stored at:")
    print(f"    - Main results: {config.OUTPUT_DIR}/all_results.pkl")
    print(f"    - Behavioral: {config.BEHAVIOR_OUTPUT}/")
    print(f"    - MVPA: {config.MVPA_OUTPUT}/")
    print(f"    - Geometry: {config.GEOMETRY_OUTPUT}/") 