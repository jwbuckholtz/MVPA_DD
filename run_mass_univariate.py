#!/usr/bin/env python3
"""
Run Mass Univariate Analysis for Delay Discounting Data

This script runs the complete mass univariate analysis pipeline including:
1. Data smoothing (4mm FWHM)
2. First-level GLM modeling (5 models)
3. Second-level random effects analysis
4. Multiple comparisons correction

Usage:
    python run_mass_univariate.py [--subjects SUBJECT_LIST] [--test]
    
Examples:
    # Run on all available subjects
    python run_mass_univariate.py
    
    # Run on specific subjects
    python run_mass_univariate.py --subjects s001,s002,s003
    
    # Test run with first 3 subjects
    python run_mass_univariate.py --test
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mass_univariate_analysis import run_full_mass_univariate_pipeline
from oak_storage_config import OAKConfig as Config

def get_available_subjects(config):
    """
    Get list of subjects with available fMRI data
    
    Parameters:
    -----------
    config : Config object
        Configuration with data paths
        
    Returns:
    --------
    list : Available subject IDs
    """
    subjects = []
    fmriprep_dir = Path(config.FMRIPREP_DIR)
    
    if not fmriprep_dir.exists():
        print(f"Warning: fMRIPrep directory not found: {fmriprep_dir}")
        return subjects
    
    # Look for subject directories
    for subject_dir in fmriprep_dir.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
            subject_id = subject_dir.name
            
            # Check if functional data exists
            func_dir = subject_dir / "ses-2" / "func"
            if func_dir.exists():
                # Look for task-discountFix files
                pattern = f"{subject_id}_ses-2_task-discountFix_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                fmri_files = list(func_dir.glob(pattern))
                
                # Check if behavioral data exists
                behavior_file = Path(config.BEHAVIOR_DIR) / f"{subject_id}_discountFix_events.tsv"
                
                if fmri_files and behavior_file.exists():
                    subjects.append(subject_id)
                    print(f"Found subject: {subject_id}")
                else:
                    if not fmri_files:
                        print(f"Warning: No fMRI data for {subject_id}")
                    if not behavior_file.exists():
                        print(f"Warning: No behavioral data for {subject_id}")
    
    print(f"\nTotal subjects with complete data: {len(subjects)}")
    return sorted(subjects)

def check_requirements(config):
    """
    Check if all required directories and dependencies exist
    
    Parameters:
    -----------
    config : Config object
        Configuration object
        
    Returns:
    --------
    bool : True if all requirements met
    """
    print("Checking requirements...")
    
    # Check data directories
    required_dirs = [
        config.FMRIPREP_DIR,
        config.BEHAVIOR_DIR
    ]
    
    for directory in required_dirs:
        if not Path(directory).exists():
            print(f"Error: Required directory not found: {directory}")
            return False
        print(f"✓ Found: {directory}")
    
    # Check Python packages
    try:
        import nibabel
        import nilearn
        from nilearn.glm.first_level import FirstLevelModel
        from nilearn.glm.second_level import SecondLevelModel
        print("✓ All required packages available")
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        return False
    
    return True

def create_slurm_script(subject_list, config):
    """
    Create SLURM script for running on HPC cluster
    
    Parameters:
    -----------
    subject_list : list
        List of subjects to analyze
    config : Config object
        Configuration object
    """
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=mass_univariate_dd
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=russpold
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@stanford.edu
#SBATCH --output={config.OUTPUT_DIR}/mass_univariate/slurm_%j.out
#SBATCH --error={config.OUTPUT_DIR}/mass_univariate/slurm_%j.err

# Load modules
module load python/3.9.0
module load fsl/6.0.4

# Activate virtual environment (modify path as needed)
source ~/venv/bin/activate

# Set environment variables
export OMP_NUM_THREADS=8
export FSLDIR=/opt/fsl-6.0.4
export FSLOUTPUTTYPE=NIFTI_GZ

# Change to project directory
cd {Path(__file__).parent}

# Run analysis
python run_mass_univariate.py --subjects {','.join(subject_list)}
"""
    
    slurm_file = f"{config.OUTPUT_DIR}/mass_univariate/submit_mass_univariate.sh"
    Path(slurm_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)
    
    # Make executable
    os.chmod(slurm_file, 0o755)
    
    print(f"SLURM script created: {slurm_file}")
    print(f"To submit: sbatch {slurm_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Run mass univariate analysis for delay discounting data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on all available subjects
    python run_mass_univariate.py
    
    # Run on specific subjects  
    python run_mass_univariate.py --subjects sub-s001,sub-s002,sub-s003
    
    # Test run with first 3 subjects
    python run_mass_univariate.py --test
    
    # Create SLURM script for cluster submission
    python run_mass_univariate.py --create-slurm
        """
    )
    
    parser.add_argument('--subjects', 
                       help='Comma-separated list of subject IDs (e.g., sub-s001,sub-s002)')
    
    parser.add_argument('--test', action='store_true',
                       help='Run test analysis with first 3 subjects')
    
    parser.add_argument('--create-slurm', action='store_true',
                       help='Create SLURM script for cluster submission')
    
    parser.add_argument('--check-only', action='store_true',
                       help='Only check requirements and available subjects')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    print("DELAY DISCOUNTING MASS UNIVARIATE ANALYSIS")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements(config):
        print("Requirements check failed. Please fix issues before running.")
        sys.exit(1)
    
    # Get available subjects
    available_subjects = get_available_subjects(config)
    
    if not available_subjects:
        print("No subjects with complete data found. Please check data paths.")
        sys.exit(1)
    
    if args.check_only:
        print(f"\nCheck complete. Found {len(available_subjects)} subjects with complete data:")
        for subject in available_subjects:
            print(f"  - {subject}")
        return
    
    # Determine subjects to analyze
    if args.subjects:
        requested_subjects = [s.strip() for s in args.subjects.split(',')]
        # Ensure proper sub- prefix
        requested_subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in requested_subjects]
        
        # Check if requested subjects are available
        subject_list = []
        for subject in requested_subjects:
            if subject in available_subjects:
                subject_list.append(subject)
            else:
                print(f"Warning: Subject {subject} not found in available data")
        
        if not subject_list:
            print("No valid subjects specified. Exiting.")
            sys.exit(1)
            
    elif args.test:
        subject_list = available_subjects[:3]
        print(f"Test mode: Using first {len(subject_list)} subjects")
    else:
        subject_list = available_subjects
        print(f"Using all {len(subject_list)} available subjects")
    
    print(f"\nSubjects to analyze ({len(subject_list)}):")
    for subject in subject_list:
        print(f"  - {subject}")
    
    # Create SLURM script if requested
    if args.create_slurm:
        create_slurm_script(subject_list, config)
        return
    
    # Confirm analysis
    if len(subject_list) > 5:
        confirm = input(f"\nRun analysis on {len(subject_list)} subjects? This may take several hours. [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("Analysis cancelled.")
            return
    
    # Setup output directories
    from oak_storage_config import setup_oak_directories
    setup_oak_directories(config)
    
    print(f"\nStarting mass univariate analysis...")
    print(f"Results will be saved to: {config.OUTPUT_DIR}/mass_univariate/")
    
    # Run analysis
    try:
        results = run_full_mass_univariate_pipeline(subject_list, config)
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"Results saved to: {config.OUTPUT_DIR}/mass_univariate/")
        print(f"Summary report: {config.OUTPUT_DIR}/mass_univariate/second_level/mass_univariate_summary_report.txt")
        
        # Print quick summary
        successful_subjects = 0
        for model_results in results['first_level'].values():
            for subject_result in model_results.values():
                if subject_result.get('success', False):
                    successful_subjects += 1
                    break
        
        print(f"\nQuick Summary:")
        print(f"  - Subjects processed: {successful_subjects}/{len(subject_list)}")
        print(f"  - Models analyzed: {len(results['first_level'])}")
        print(f"  - Group maps generated: {len([r for model in results['second_level'].values() for r in model.values() if r.get('success')])}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 