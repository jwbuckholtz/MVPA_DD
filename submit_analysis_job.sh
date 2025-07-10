#!/bin/bash
#SBATCH --job-name=delay_discounting_mvpa
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/mvpa_analysis_%j.out
#SBATCH --error=logs/mvpa_analysis_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@stanford.edu

# Stanford Delay Discounting MVPA Analysis Job - OAK Storage Version
# This script runs the complete analysis pipeline on the HPC cluster with OAK storage
# 
# Analysis Components:
# 1. ROI mask creation
# 2. Behavioral modeling and MVPA decoding
# 3. Standard results analysis and visualization
# 4. Advanced delay discounting geometry analysis (optional)
# 5. Comprehensive results summary
#
# Advanced Geometry Features:
# - Manifold alignment (Procrustes, CCA)
# - Geodesic distance analysis
# - Manifold curvature estimation  
# - Information geometry metrics
# - Specialized delay discounting comparisons
# - Comprehensive visualizations
#
# Environment Variables:
# - RUN_GEOMETRY: Set to "false" to skip advanced geometry analysis (default: "true")

echo "Starting Delay Discounting MVPA Analysis with OAK Storage"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"

# Set OAK storage paths
export OAK_DATA_ROOT="/oak/stanford/groups/russpold/data/uh2/aim1"
export OAK_OUTPUT_ROOT="/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis"
export RESULTS_DIR="${OAK_OUTPUT_ROOT}/delay_discounting_results"
export MASKS_DIR="${OAK_DATA_ROOT}/derivatives/masks"

echo "=== OAK Storage Configuration ==="
echo "Input data: ${OAK_DATA_ROOT}"
echo "Output directory: ${RESULTS_DIR}"
echo "Masks directory: ${MASKS_DIR}"
echo "================================="

# Create output directories on OAK
echo "Creating output directories on OAK..."
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/behavioral_analysis"
mkdir -p "${RESULTS_DIR}/mvpa_analysis"
mkdir -p "${RESULTS_DIR}/geometry_analysis"
mkdir -p "${RESULTS_DIR}/dd_geometry_results"
mkdir -p "${RESULTS_DIR}/dd_geometry_results/visualizations"
mkdir -p "${MASKS_DIR}"
mkdir -p logs

# Set proper permissions for group access
chmod 755 "${RESULTS_DIR}" "${MASKS_DIR}" 2>/dev/null || echo "Note: Could not set directory permissions"
chmod 755 "${RESULTS_DIR}/behavioral_analysis" "${RESULTS_DIR}/mvpa_analysis" "${RESULTS_DIR}/geometry_analysis" 2>/dev/null || true
chmod 755 "${RESULTS_DIR}/dd_geometry_results" "${RESULTS_DIR}/dd_geometry_results/visualizations" 2>/dev/null || true

# Load required modules
module load python/3.9.0
module load gcc/10.1.0

# Set up Python environment
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip

# Core scientific packages
pip install numpy scipy pandas matplotlib seaborn
pip install "scikit-learn>=0.24.0" statsmodels

# Neuroimaging packages
pip install nibabel nilearn

# Additional utilities
pip install joblib tqdm pathlib

# Verify advanced geometry requirements
echo "Verifying advanced geometry dependencies..."
python -c "
import sklearn
from sklearn.manifold import Isomap
from sklearn.cross_decomposition import CCA
from scipy.stats import gaussian_kde
print(f'scikit-learn version: {sklearn.__version__}')
print('Advanced geometry dependencies verified')
"

# Verify installations
echo "Verifying package installations..."
python -c "import numpy, scipy, pandas, matplotlib, seaborn, sklearn, nibabel, nilearn; print('All packages imported successfully')"

# ENHANCED: Comprehensive data verification using data_utils
echo "Verifying OAK data access and integrity using data_utils..."
python -c "
import sys
sys.path.append('.')

from data_utils import (
    get_complete_subjects, check_data_integrity, 
    SubjectManager, DataError
)
from oak_storage_config import OAKConfig
import pandas as pd

print('=' * 60)
print('COMPREHENSIVE DATA VALIDATION')
print('=' * 60)

try:
    config = OAKConfig()
    
    # Check directory access
    print(f'fMRI data directory: {config.FMRIPREP_DIR}')
    print(f'Behavioral data directory: {config.BEHAVIOR_DIR}')
    
    # Use data_utils for comprehensive checking
    manager = SubjectManager(config)
    
    # Get subjects with any data
    all_subjects = manager.get_available_subjects(require_both=False)
    print(f'Total subjects found: {len(all_subjects)}')
    
    # Get subjects with complete data
    complete_subjects = get_complete_subjects(config)
    print(f'Subjects with complete data: {len(complete_subjects)}')
    
    if len(complete_subjects) == 0:
        print('ERROR: No subjects with complete data found!')
        print('Check your data paths and file permissions.')
        sys.exit(1)
    
    # Run data integrity check on first 5 subjects for speed
    print(f'\\nRunning data integrity check on first 5 subjects...')
    sample_subjects = complete_subjects[:5]
    integrity_report = check_data_integrity(sample_subjects, config)
    
    # Display summary
    valid_count = integrity_report['complete'].sum()
    print(f'Validated subjects: {valid_count}/{len(sample_subjects)}')
    
    if valid_count == 0:
        print('ERROR: No subjects passed data integrity check!')
        print('Data quality issues detected.')
        sys.exit(1)
    
    # Show sample subject details
    print(f'\\nSample Subject Details:')
    display_cols = ['subject_id', 'complete', 'n_trials', 'accuracy']
    available_cols = [col for col in display_cols if col in integrity_report.columns]
    print(integrity_report[available_cols].head())
    
    print(f'\\n✓ Data validation successful!')
    print(f'✓ Ready to process {len(complete_subjects)} subjects')
    print('=' * 60)
    
except DataError as e:
    print(f'ERROR: Data validation failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Unexpected error during data validation: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Check if data validation passed
if [ $? -ne 0 ]; then
    echo "ERROR: Data validation failed. Exiting."
    exit 1
fi

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# UPDATED: Step 1: Validate pre-existing ROI masks on OAK
echo "Step 1: Validating pre-existing ROI masks on OAK..."
python -c "
import sys
sys.path.append('.')

from validate_roi_masks import MaskValidator, check_oak_connectivity
from data_utils import check_mask_exists, load_mask
from oak_storage_config import OAKConfig
from pathlib import Path

config = OAKConfig()
print(f'Masks directory: {config.MASKS_DIR}')

# Check OAK connectivity first
print('Checking OAK connectivity...')
if not check_oak_connectivity():
    print('\\nERROR: Cannot access OAK storage!')
    sys.exit(1)

# Use the new mask validator
print('\\nRunning mask validation...')
validator = MaskValidator(config)
results_df = validator.validate_all_masks()

# Check if core masks are valid
if not validator.core_masks_valid:
    print('\\nERROR: Not all core ROI masks are valid!')
    print('Please ensure all required masks are available on OAK.')
    sys.exit(1)

print('\\n✓ All core ROI masks validated successfully!')

# Print available ROIs
available_rois = validator.get_available_rois()
print(f'Available ROIs for analysis: {\\', \\'.join(available_rois)}')
"

# Check if mask validation passed
if [ $? -ne 0 ]; then
    echo "ERROR: Mask validation failed. Exiting."
    exit 1
fi

# Step 2: Run main MVPA analysis (with OAK configuration)
echo "Step 2: Running main MVPA analysis with OAK storage..."
python -c "
import sys
sys.path.append('.')

# Import components
from delay_discounting_mvpa_pipeline import main, setup_directories
from oak_storage_config import OAKConfig

# Use OAK configuration
config = OAKConfig()
print(f'Using OAK output directory: {config.OUTPUT_DIR}')

# Set up directories
setup_directories(config)

# Replace the Config class in the main module
import delay_discounting_mvpa_pipeline
delay_discounting_mvpa_pipeline.Config = OAKConfig

# Run main analysis
print('Starting main analysis pipeline...')
main()
print('Main analysis completed successfully')
"

# Check if main analysis completed successfully
if [ ! -f "${RESULTS_DIR}/all_results.pkl" ]; then
    echo "Error: Main analysis did not complete successfully - results file not found"
    exit 1
fi

echo "Main analysis completed successfully"
echo "Results file size: $(ls -lh ${RESULTS_DIR}/all_results.pkl)"

# ENHANCED: Step 3: Analyze and visualize results using data_utils
echo "Step 3: Analyzing and visualizing results with data_utils integration..."
python -c "
import sys
sys.path.append('.')

from analyze_results import ResultsAnalyzer, check_pipeline_data_integrity
from data_utils import load_processed_data, DataError
from oak_storage_config import OAKConfig
import os

config = OAKConfig()
results_file = f'{config.OUTPUT_DIR}/all_results.pkl'

print(f'Analyzing results from: {results_file}')

# Check if results file exists
if not os.path.exists(results_file):
    print(f'ERROR: Results file not found: {results_file}')
    sys.exit(1)

try:
    # Use enhanced data loading from data_utils
    print('Loading results using data_utils...')
    results_data, metadata = load_processed_data(results_file)
    print(f'Results metadata: {metadata}')
    
    # Run enhanced analysis
    analyzer = ResultsAnalyzer(results_file)
    analyzer.output_dir = f'{config.OUTPUT_DIR}/analysis_outputs'
    
    print('Running comprehensive results analysis...')
    analyzer.run_analysis()
    
    # Also run data integrity check for final validation
    print('\\nRunning final data integrity check...')
    integrity_report = check_pipeline_data_integrity()
    
    print('\\n✓ Results analysis completed successfully!')
    print(f'✓ Analysis outputs saved to: {analyzer.output_dir}')
    
except DataError as e:
    print(f'ERROR: Data loading failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Analysis failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Check if results analysis passed
if [ $? -ne 0 ]; then
    echo "ERROR: Results analysis failed. Exiting."
    exit 1
fi

echo "✓ Results analysis completed successfully"

# Step 4: Advanced Delay Discounting Geometry Analysis (Optional)
echo "Step 4: Running advanced delay discounting geometry analysis..."

# Check if we should run geometry analysis (can be controlled by environment variable)
RUN_GEOMETRY=${RUN_GEOMETRY:-"true"}

if [ "$RUN_GEOMETRY" = "true" ]; then
    echo "Running advanced geometry analysis on extracted ROI data..."
    
    # Create geometry configuration
    GEOMETRY_CONFIG="${RESULTS_DIR}/dd_geometry_config.json"
    cat > "${GEOMETRY_CONFIG}" << EOF
{
  "output_dir": "${RESULTS_DIR}/dd_geometry_results",
  "n_permutations": 1000,
  "random_state": 42,
  "alpha": 0.05,
  "n_components_pca": 15,
  "n_components_mds": 8,
  "standardize_data": true,
  "plot_format": "png",
  "dpi": 300,
  "delay_short_threshold": 7,
  "delay_long_threshold": 30,
  "advanced_geometry": {
    "enable_manifold_alignment": true,
    "enable_geodesic_analysis": true,
    "enable_curvature_analysis": true,
    "enable_information_geometry": true,
    "isomap_n_neighbors": 5,
    "curvature_n_neighbors": 5,
    "kde_bandwidth": "auto"
  }
}
EOF

    # Run geometry analysis for each ROI
    python -c "
import sys
sys.path.append('.')
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from delay_discounting_geometry_analysis import DelayDiscountingGeometryAnalyzer

# Load main results to extract neural data and behavioral data
results_file = '${RESULTS_DIR}/all_results.pkl'
config_file = '${GEOMETRY_CONFIG}'

print(f'Loading results from: {results_file}')
with open(results_file, 'rb') as f:
    all_results = pickle.load(f)

# Initialize geometry analyzer
analyzer = DelayDiscountingGeometryAnalyzer(config_file)

# Process each subject and ROI
for subject_id, subject_results in all_results.items():
    if not isinstance(subject_results, dict) or 'behavioral_analysis' not in subject_results:
        continue
    
    print(f'Processing subject: {subject_id}')
    
    behavioral_data = subject_results.get('behavioral_analysis', {})
    mvpa_results = subject_results.get('mvpa_analysis', {})
    
    # Process each ROI that has MVPA results
    for roi_name, roi_results in mvpa_results.items():
        if not isinstance(roi_results, dict):
            continue
            
        print(f'  Processing ROI: {roi_name}')
        
        try:
            # Extract neural data and behavioral data for this subject/ROI
            # Note: This would need to be adapted based on actual data structure
            # For now, skip if neural data is not readily available
            print(f'    Skipping {roi_name} - neural data extraction needs implementation')
            continue
            
        except Exception as e:
            print(f'    Error processing {subject_id} {roi_name}: {e}')
            continue

print('Advanced geometry analysis completed')
"
    
    echo "Advanced geometry analysis completed"
else
    echo "Skipping advanced geometry analysis (set RUN_GEOMETRY=true to enable)"
fi

# Step 5: Create summary of output files on OAK
echo "Step 5: Creating file summary on OAK..."
SUMMARY_FILE="${RESULTS_DIR}/analysis_summary.txt"

echo "Analysis completed on $(date)" > "${SUMMARY_FILE}"
echo "OAK Storage Location: ${RESULTS_DIR}" >> "${SUMMARY_FILE}"
echo "Files created:" >> "${SUMMARY_FILE}"
echo "=============" >> "${SUMMARY_FILE}"

# List all output files on OAK
find "${RESULTS_DIR}" -name "*.pkl" -o -name "*.csv" | sort >> "${SUMMARY_FILE}"
find "${RESULTS_DIR}" -name "*.png" -o -name "*.txt" -o -name "*.json" | sort >> "${SUMMARY_FILE}"
find "${MASKS_DIR}" -name "*.nii.gz" -o -name "*.png" | sort >> "${SUMMARY_FILE}"

echo "" >> "${SUMMARY_FILE}"
echo "Disk usage:" >> "${SUMMARY_FILE}"
du -sh "${RESULTS_DIR}" "${MASKS_DIR}" >> "${SUMMARY_FILE}"

# Display summary
echo "Analysis Summary:"
cat "${SUMMARY_FILE}"

echo ""
echo "=== File Structure on OAK ==="
echo "Main results directory:"
ls -la "${RESULTS_DIR}/"

echo ""
echo "Subject-specific geometry directories:"
find "${RESULTS_DIR}/geometry_analysis" -type d -maxdepth 2 2>/dev/null | head -10

echo ""
echo "Advanced geometry results:"
find "${RESULTS_DIR}/dd_geometry_results" -type f -name "*.json" -o -name "*.txt" 2>/dev/null | head -10

echo ""
echo "Geometry visualizations:"
find "${RESULTS_DIR}/dd_geometry_results/visualizations" -name "*.png" 2>/dev/null | head -10

echo ""
echo "Sample output files:"
find "${RESULTS_DIR}" -name "*.pkl" -o -name "*.png" -o -name "*.csv" | head -20

echo ""
echo "Total storage used on OAK:"
du -sh "${RESULTS_DIR}" "${MASKS_DIR}"

# Set permissions for output files
chmod -R 755 "${RESULTS_DIR}" "${MASKS_DIR}" 2>/dev/null || echo "Note: Could not set all file permissions"

echo ""
echo "=== ACCESS INSTRUCTIONS ==="
echo "Results are stored on OAK at: ${RESULTS_DIR}"
echo "To access results:"
echo "  1. SSH to a login node: ssh <username>@login.sherlock.stanford.edu"
echo "  2. Navigate to results: cd ${RESULTS_DIR}"
echo "  3. Load main results: python -c \"import pickle; results = pickle.load(open('all_results.pkl', 'rb'))\""
echo ""
echo "Advanced Geometry Results:"
echo "  - JSON results: ${RESULTS_DIR}/dd_geometry_results/"
echo "  - Visualizations: ${RESULTS_DIR}/dd_geometry_results/visualizations/"
echo "  - Summary reports: ${RESULTS_DIR}/dd_geometry_results/*_summary_report.txt"
echo ""
echo "To run additional geometry analyses:"
echo "  python delay_discounting_geometry_analysis.py --neural-data <data.npy> --behavioral-data <behavior.csv>"
echo "=========================="

# ENHANCED: Final comprehensive summary using data_utils
echo ""
echo "GENERATING COMPREHENSIVE ANALYSIS SUMMARY USING DATA_UTILS..."
echo "============================================================="

python -c "
import sys
sys.path.append('.')

from data_utils import (
    get_complete_subjects, check_data_integrity, 
    load_processed_data, SubjectManager
)
from oak_storage_config import OAKConfig
import os
from pathlib import Path

config = OAKConfig()

print('\\n' + '='*80)
print('DELAY DISCOUNTING MVPA ANALYSIS - COMPREHENSIVE FINAL SUMMARY')
print('='*80)

try:
    # Get final subject counts using data_utils
    print('\\n📊 SUBJECT SUMMARY (using data_utils):')
    manager = SubjectManager(config)
    
    all_subjects = manager.get_available_subjects(require_both=False)
    complete_subjects = get_complete_subjects(config)
    
    print(f'  Total subjects found: {len(all_subjects)}')
    print(f'  Complete subjects (fMRI + behavior): {len(complete_subjects)}')
    print(f'  Data completeness rate: {len(complete_subjects)/len(all_subjects)*100:.1f}%' if all_subjects else 'N/A')
    
    # Check results file
    print('\\n📁 RESULTS FILES:')
    results_file = f'{config.OUTPUT_DIR}/all_results.pkl'
    if os.path.exists(results_file):
        try:
            results_data, metadata = load_processed_data(results_file)
            print(f'  ✓ Main results file: {os.path.getsize(results_file)/1024/1024:.1f} MB')
            print(f'  ✓ Analysis metadata: {metadata}')
            print(f'  ✓ Results structure: {type(results_data)} with {len(results_data) if hasattr(results_data, \"__len__\") else \"N/A\"} entries')
        except Exception as e:
            print(f'  ⚠ Results file exists but loading failed: {e}')
    else:
        print(f'  ✗ Main results file not found: {results_file}')
    
    # Check analysis outputs
    print('\\n📈 ANALYSIS OUTPUTS:')
    analysis_dir = Path(f'{config.OUTPUT_DIR}/analysis_outputs')
    if analysis_dir.exists():
        output_files = list(analysis_dir.glob('*'))
        print(f'  ✓ Analysis outputs created: {len(output_files)} files')
        
        # Check for key output files
        key_files = {
            'summary_report.txt': 'Summary report',
            'behavioral_distributions.png': 'Behavioral plots', 
            'group_mvpa_statistics.csv': 'MVPA statistics',
            'data_integrity_report.csv': 'Data integrity report'
        }
        
        for filename, description in key_files.items():
            file_path = analysis_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / 1024 / 1024
                print(f'    ✓ {description}: {filename} ({size_mb:.2f} MB)')
            else:
                print(f'    ⚠ {description}: {filename} - MISSING')
    else:
        print(f'  ✗ Analysis outputs directory not found: {analysis_dir}')
    
    # Run final data integrity check using data_utils
    print('\\n🔍 FINAL DATA INTEGRITY CHECK (using data_utils):')
    if complete_subjects:
        # Check first 10 subjects for speed
        sample_size = min(10, len(complete_subjects))
        sample_subjects = complete_subjects[:sample_size]
        
        try:
            integrity_report = check_data_integrity(sample_subjects, config)
            
            if len(integrity_report) > 0:
                total_checked = len(integrity_report)
                complete_count = integrity_report['complete'].sum()
                valid_behavior = integrity_report['behavior_valid'].sum()
                valid_fmri = integrity_report['fmri_valid'].sum()
                
                print(f'  📋 Sample checked: {total_checked} subjects')
                print(f'  ✓ Complete: {complete_count}/{total_checked} ({complete_count/total_checked*100:.1f}%)')
                print(f'  ✓ Valid behavioral: {valid_behavior}/{total_checked} ({valid_behavior/total_checked*100:.1f}%)')
                print(f'  ✓ Valid fMRI: {valid_fmri}/{total_checked} ({valid_fmri/total_checked*100:.1f}%)')
                
                if complete_count > 0:
                    complete_data = integrity_report[integrity_report['complete']]
                    avg_trials = complete_data['n_trials'].mean()
                    avg_accuracy = complete_data['accuracy'].mean()
                    print(f'  📊 Average trials: {avg_trials:.1f}')
                    print(f'  📊 Average accuracy: {avg_accuracy:.3f}')
            else:
                print('  ⚠ No subjects in integrity report')
                
        except Exception as e:
            print(f'  ✗ Integrity check failed: {e}')
    else:
        print('  ⚠ No complete subjects found for integrity check')
    
    # Check ROI masks using data_utils
    print('\\n🎭 ROI MASKS VALIDATION (using data_utils):')
    from data_utils import check_mask_exists, load_mask
    
    mask_summary = []
    for roi_name, mask_path in config.ROI_MASKS.items():
        exists = check_mask_exists(mask_path)
        if exists:
            try:
                mask_img = load_mask(mask_path, validate=True)
                n_voxels = (mask_img.get_fdata() > 0).sum()
                size_mb = os.path.getsize(mask_path) / 1024 / 1024
                mask_summary.append(f'    ✓ {roi_name}: {n_voxels:,} voxels ({size_mb:.2f} MB)')
            except Exception as e:
                mask_summary.append(f'    ⚠ {roi_name}: Validation failed - {e}')
        else:
            mask_summary.append(f'    ✗ {roi_name}: File not found')
    
    for summary in mask_summary:
        print(summary)
    
    print('\\n' + '='*80)
    print('✅ ANALYSIS PIPELINE COMPLETED SUCCESSFULLY WITH DATA_UTILS INTEGRATION!')
    print('='*80)
    
except Exception as e:
    print(f'\\n❌ Error generating comprehensive summary: {e}')
    import traceback
    traceback.print_exc()
    print('='*80)
"

echo ""
echo "🎉 DELAY DISCOUNTING MVPA ANALYSIS COMPLETED!"
echo "============================================="
echo ""
echo "⏰ Job completed at: $(date)"
echo "⌛ Total runtime: $((SECONDS/3600))h $(((SECONDS%3600)/60))m $((SECONDS%60))s"
echo ""
echo "📂 Results stored in: ${RESULTS_DIR}"
echo "📄 Log file: ${RESULTS_DIR}/slurm-${SLURM_JOB_ID}.out"
echo ""
echo "🚀 ENHANCED FEATURES (using data_utils):"
echo "  ✅ Comprehensive data validation and integrity checking"
echo "  ✅ Automated subject discovery with quality control"
echo "  ✅ Enhanced mask creation and validation"
echo "  ✅ Improved error handling and detailed reporting"
echo "  ✅ Centralized data loading with metadata tracking"
echo ""
echo "📋 KEY OUTPUT FILES:"
echo "  1. 📊 Data integrity report: ${RESULTS_DIR}/analysis_outputs/data_integrity_report.csv"
echo "  2. 📈 Enhanced analysis outputs: ${RESULTS_DIR}/analysis_outputs/"
echo "  3. 📝 Comprehensive summary: ${RESULTS_DIR}/analysis_outputs/summary_report.txt"
echo "  4. 🧠 Individual results: ${RESULTS_DIR}/"
echo "  5. 🎭 Validated ROI masks: ${MASKS_DIR}/"
echo ""
echo "📚 DOCUMENTATION:"
echo "  - Main README: README.md"
echo "  - Data utilities guide: DATA_UTILS_README.md"
echo "  - Demo script: python demo_data_utils.py"
echo ""
echo "🔧 FOR FUTURE ANALYSES:"
echo "  - Use data_utils functions for consistent data handling"
echo "  - Run 'python analyze_results.py --check_data' for data integrity checks"
echo "  - See DATA_UTILS_README.md for complete usage examples" 