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

# Stanford Delay Discounting MVPA Analysis Job
# This script runs the complete analysis pipeline on the HPC cluster

echo "Starting Delay Discounting MVPA Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"

# Load required modules
module load python/3.9.0
module load gcc/10.1.0

# Create logs directory if it doesn't exist
mkdir -p logs

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

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directories
echo "Creating output directories..."
mkdir -p delay_discounting_results/{behavioral_analysis,mvpa_analysis,geometry_analysis}
mkdir -p masks
mkdir -p analysis_outputs

# Step 1: Create ROI masks
echo "Step 1: Creating ROI masks..."
python create_roi_masks.py

# Check if masks were created successfully
if [ ! -f "masks/striatum_mask.nii.gz" ] || [ ! -f "masks/dlpfc_mask.nii.gz" ] || [ ! -f "masks/vmpfc_mask.nii.gz" ]; then
    echo "Error: ROI masks were not created successfully"
    exit 1
fi

echo "ROI masks created successfully"

# Step 2: Run main MVPA analysis
echo "Step 2: Running main MVPA analysis..."
python delay_discounting_mvpa_pipeline.py

# Check if main analysis completed successfully
if [ ! -f "delay_discounting_results/all_results.pkl" ]; then
    echo "Error: Main analysis did not complete successfully"
    exit 1
fi

echo "Main analysis completed successfully"

# Step 3: Analyze and visualize results
echo "Step 3: Analyzing and visualizing results..."
python analyze_results.py

# Check if results analysis completed
if [ ! -f "analysis_outputs/summary_report.txt" ]; then
    echo "Warning: Results analysis may not have completed successfully"
fi

echo "Results analysis completed"

# Step 4: Create summary of output files
echo "Step 4: Creating file summary..."
echo "Analysis completed on $(date)" > analysis_summary.txt
echo "Files created:" >> analysis_summary.txt
echo "=============" >> analysis_summary.txt

# List all output files
find delay_discounting_results -name "*.pkl" -o -name "*.csv" | sort >> analysis_summary.txt
find analysis_outputs -name "*.png" -o -name "*.txt" -o -name "*.csv" | sort >> analysis_summary.txt
find masks -name "*.nii.gz" -o -name "*.png" | sort >> analysis_summary.txt

echo "" >> analysis_summary.txt
echo "Disk usage:" >> analysis_summary.txt
du -sh delay_discounting_results analysis_outputs masks >> analysis_summary.txt

# Display summary
echo "Analysis Summary:"
cat analysis_summary.txt

# Set permissions for output files
chmod -R 755 delay_discounting_results analysis_outputs masks

echo "Job completed successfully!"
echo "End time: $(date)" 