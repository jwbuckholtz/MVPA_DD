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
mkdir -p "${MASKS_DIR}"
mkdir -p logs

# Set proper permissions for group access
chmod 755 "${RESULTS_DIR}" "${MASKS_DIR}" 2>/dev/null || echo "Note: Could not set directory permissions"
chmod 755 "${RESULTS_DIR}/behavioral_analysis" "${RESULTS_DIR}/mvpa_analysis" "${RESULTS_DIR}/geometry_analysis" 2>/dev/null || true

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
pip install scikit-learn statsmodels

# Neuroimaging packages
pip install nibabel nilearn

# Additional utilities
pip install joblib tqdm pathlib

# Verify installations
echo "Verifying package installations..."
python -c "import numpy, scipy, pandas, matplotlib, seaborn, sklearn, nibabel, nilearn; print('All packages imported successfully')"

# Verify data access
echo "Verifying OAK data access..."
echo "fMRI data directory: ${OAK_DATA_ROOT}/derivatives/fmriprep"
if [ -d "${OAK_DATA_ROOT}/derivatives/fmriprep" ]; then
    echo "fMRI data accessible: $(ls ${OAK_DATA_ROOT}/derivatives/fmriprep | wc -l) subjects found"
else
    echo "ERROR: Cannot access fMRI data directory"
    exit 1
fi

echo "Behavioral data directory: ${OAK_DATA_ROOT}/behavioral_data/event_files"
if [ -d "${OAK_DATA_ROOT}/behavioral_data/event_files" ]; then
    echo "Behavioral data accessible: $(ls ${OAK_DATA_ROOT}/behavioral_data/event_files/*.tsv | wc -l) files found"
else
    echo "ERROR: Cannot access behavioral data directory"
    exit 1
fi

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Step 1: Create ROI masks (store on OAK)
echo "Step 1: Creating ROI masks on OAK..."
python -c "
import sys
sys.path.append('.')

# Import and modify create_roi_masks to use OAK paths
from create_roi_masks import create_roi_masks
from oak_storage_config import OAKConfig

config = OAKConfig()
print(f'Creating masks in: {config.MASKS_DIR}')
create_roi_masks(output_dir=config.MASKS_DIR, mni_template=None)
"

# Check if masks were created successfully
if [ ! -f "${MASKS_DIR}/striatum_mask.nii.gz" ] || [ ! -f "${MASKS_DIR}/dlpfc_mask.nii.gz" ] || [ ! -f "${MASKS_DIR}/vmpfc_mask.nii.gz" ]; then
    echo "Error: ROI masks were not created successfully on OAK"
    exit 1
fi

echo "ROI masks created successfully on OAK"
echo "Mask files created:"
ls -lh "${MASKS_DIR}/"*.nii.gz

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

# Step 3: Analyze and visualize results (with OAK paths)
echo "Step 3: Analyzing and visualizing results..."
python -c "
import sys
sys.path.append('.')

from analyze_results import ResultsAnalyzer
from oak_storage_config import OAKConfig

config = OAKConfig()
results_file = f'{config.OUTPUT_DIR}/all_results.pkl'

print(f'Analyzing results from: {results_file}')
analyzer = ResultsAnalyzer(results_file)

# Set output directory to OAK
analyzer.output_dir = f'{config.OUTPUT_DIR}/analysis_outputs'

# Run analysis
analyzer.run_complete_analysis()
print('Results analysis completed successfully')
"

echo "Results analysis completed"

# Step 4: Create summary of output files on OAK
echo "Step 4: Creating file summary on OAK..."
SUMMARY_FILE="${RESULTS_DIR}/analysis_summary.txt"

echo "Analysis completed on $(date)" > "${SUMMARY_FILE}"
echo "OAK Storage Location: ${RESULTS_DIR}" >> "${SUMMARY_FILE}"
echo "Files created:" >> "${SUMMARY_FILE}"
echo "=============" >> "${SUMMARY_FILE}"

# List all output files on OAK
find "${RESULTS_DIR}" -name "*.pkl" -o -name "*.csv" | sort >> "${SUMMARY_FILE}"
find "${RESULTS_DIR}" -name "*.png" -o -name "*.txt" | sort >> "${SUMMARY_FILE}"
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
echo "=========================="

echo "Job completed successfully!"
echo "End time: $(date)" 