#!/bin/bash
#SBATCH --job-name=mvpa_delay_discounting
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --output=logs/mvpa_%j.out
#SBATCH --error=logs/mvpa_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@stanford.edu

#======================================
# Sherlock SLURM Script for MVPA Analysis
#======================================
#
# This script runs the delay discounting MVPA pipeline on Stanford's Sherlock cluster.
# 
# IMPORTANT: Update the following before running:
# 1. Change --mail-user to your email address
# 2. Adjust memory/time based on your dataset size
# 3. Update any paths if your setup differs
#
# Usage:
#   sbatch submit_mvpa_sherlock.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/mvpa_JOBID.out
#
#======================================

# Print job information
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# Load required modules
echo "Loading modules..."
module purge
module load python/3.9.0
module load py-pip/21.1.2_py39

# Optional: Load pre-installed neuroimaging modules if available
# module load py-numpy/1.20.3_py39
# module load py-scipy/1.7.0_py39
# module load py-nibabel/3.2.1_py39
# module load py-nilearn/0.8.1_py39

echo "Loaded modules:"
module list

# Change to analysis directory
cd $SCRATCH/delay_discounting_analysis || {
    echo "Error: Cannot access analysis directory $SCRATCH/delay_discounting_analysis"
    echo "Make sure you've set up the analysis directory correctly"
    exit 1
}

# Activate virtual environment
echo "Activating virtual environment..."
source mvpa_env/bin/activate || {
    echo "Error: Cannot activate virtual environment"
    echo "Make sure you've created the virtual environment: python3 -m venv mvpa_env"
    exit 1
}

# Verify environment
echo "Python environment:"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Pip location: $(which pip)"

# Set Sherlock-specific configuration
export MVPA_CONFIG_FILE=config_sherlock.yaml

# Verify configuration file exists
if [ ! -f "$MVPA_CONFIG_FILE" ]; then
    echo "Error: Configuration file $MVPA_CONFIG_FILE not found"
    echo "Make sure you've created a Sherlock-specific configuration file"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Display analysis configuration
echo "========================================"
echo "Analysis Configuration"
echo "========================================"
echo "Configuration file: $MVPA_CONFIG_FILE"
echo "Results will be saved to: \$SCRATCH/delay_discounting_analysis/results"
echo "Logs directory: $(pwd)/logs"
echo "========================================"

# Run the analysis
echo "Starting MVPA analysis..."
echo "Analysis start time: $(date)"

# Option 1: Run full analysis on all subjects
python3 delay_discounting_mvpa_pipeline.py

# Option 2: Run on specific subjects (uncomment and modify as needed)
# python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_002 subject_003

# Option 3: Run specific analysis types (uncomment as needed)
# python3 delay_discounting_mvpa_pipeline.py --behavioral-only
# python3 delay_discounting_mvpa_pipeline.py --mvpa-only

# Check exit status
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Analysis completed successfully!"
else
    echo "Analysis failed with exit code: $exit_code"
    echo "Check the error log for details: logs/mvpa_${SLURM_JOB_ID}.err"
fi

echo "Analysis end time: $(date)"

# Generate summary if analysis was successful
if [ $exit_code -eq 0 ] && [ -f "analyze_results.py" ]; then
    echo "Generating result summary..."
    python3 analyze_results.py
fi

# Copy results to permanent storage
if [ $exit_code -eq 0 ] && [ -d "results" ]; then
    echo "Copying results to permanent storage..."
    
    # Create timestamped directory on OAK
    timestamp=$(date +%Y%m%d_%H%M%S)
    oak_results_dir="/oak/stanford/groups/russpold/users/$USER/mvpa_results_$timestamp"
    
    # Create directory and copy results
    mkdir -p "$oak_results_dir"
    cp -r results/* "$oak_results_dir/"
    
    echo "Results copied to: $oak_results_dir"
    
    # Also copy configuration and logs for reproducibility
    cp "$MVPA_CONFIG_FILE" "$oak_results_dir/"
    cp logs/mvpa_${SLURM_JOB_ID}.out "$oak_results_dir/"
    cp logs/mvpa_${SLURM_JOB_ID}.err "$oak_results_dir/"
    
    echo "Configuration and logs also copied for reproducibility"
else
    echo "Skipping result copy due to analysis failure or missing results directory"
fi

# Display resource usage information
echo "========================================"
echo "Job Resource Usage"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Run 'seff $SLURM_JOB_ID' after job completion to see detailed resource usage"
echo "Check efficiency to optimize future jobs"
echo "========================================"

echo "Job completed at: $(date)"
echo "Check logs for detailed output:"
echo "  Standard output: logs/mvpa_${SLURM_JOB_ID}.out"
echo "  Error output: logs/mvpa_${SLURM_JOB_ID}.err"

exit $exit_code 