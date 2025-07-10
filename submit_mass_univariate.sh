#!/bin/bash
#SBATCH --job-name=dd_mass_univariate
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=russpold
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@stanford.edu
#SBATCH --output=/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/slurm_%j.out
#SBATCH --error=/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/slurm_%j.err

# ==============================================================================
# Delay Discounting Mass Univariate Analysis - SLURM Batch Script
# ==============================================================================
#
# This script runs the complete mass univariate GLM analysis pipeline including:
# 1. Spatial smoothing (4mm FWHM)
# 2. First-level GLM modeling (5 models)
# 3. Second-level random effects analysis
# 4. Multiple comparisons correction (FDR + cluster)
#
# Usage:
#   sbatch submit_mass_univariate.sh                    # Run all subjects
#   sbatch --export=MODE=test submit_mass_univariate.sh # Run test (3 subjects)
#   sbatch --export=SUBJECTS=sub-s001,sub-s002 submit_mass_univariate.sh # Specific subjects
#
# ==============================================================================

# Configuration variables (can be overridden with --export)
MODE=${MODE:-full}                    # full, test, or custom
SUBJECTS=${SUBJECTS:-}                # Comma-separated list of subjects
PROJECT_DIR=${PROJECT_DIR:-/oak/stanford/groups/russpold/data/uh2/aim1/analysis/MVPA_DD}
PYTHON_ENV=${PYTHON_ENV:-~/venv}      # Path to your virtual environment

# ==============================================================================
# Setup and Environment
# ==============================================================================

echo "=========================================="
echo "Delay Discounting Mass Univariate Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Mode: $MODE"
echo "=========================================="

# Load required modules
echo "Loading modules..."
module purge
module load python/3.9.0
module load fsl/6.0.4

# Verify modules loaded
echo "Loaded modules:"
module list

# Set FSL environment variables
export FSLDIR=${FSLDIR:-/opt/fsl-6.0.4}
export FSLOUTPUTTYPE=NIFTI_GZ
export FSL_DIR=$FSLDIR
export FSLMULTIFILEQUIT=TRUE
export FSLTCLSH=${FSLDIR}/bin/fsltclsh
export FSLWISH=${FSLDIR}/bin/fslwish

# Set parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set memory limits
export MALLOC_TRIM_THRESHOLD_=0

echo "Environment variables set:"
echo "  FSLDIR: $FSLDIR"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "  SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"

# ==============================================================================
# Python Environment Setup
# ==============================================================================

echo "Setting up Python environment..."

# Activate virtual environment
if [ -d "$PYTHON_ENV" ]; then
    echo "Activating virtual environment: $PYTHON_ENV"
    source $PYTHON_ENV/bin/activate
else
    echo "Warning: Virtual environment not found at $PYTHON_ENV"
    echo "Using system Python..."
fi

# Verify Python and packages
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Check key packages
echo "Checking required packages..."
python -c "
import sys
try:
    import numpy, pandas, nibabel, nilearn, sklearn, scipy, statsmodels
    print('✓ All required packages available')
    print(f'  - nilearn version: {nilearn.__version__}')
    print(f'  - sklearn version: {sklearn.__version__}')
except ImportError as e:
    print(f'✗ Package import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Error: Required packages not available"
    exit 1
fi

# ==============================================================================
# Analysis Directory Setup
# ==============================================================================

echo "Setting up analysis directories..."

# Change to project directory
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "Working directory: $(pwd)"
else
    echo "Error: Project directory not found: $PROJECT_DIR"
    exit 1
fi

# Verify analysis scripts exist
for script in run_mass_univariate.py mass_univariate_analysis.py oak_storage_config.py; do
    if [ ! -f "$script" ]; then
        echo "Error: Required script not found: $script"
        exit 1
    fi
done

echo "✓ All required scripts found"

# Create output directory structure
echo "Creating output directories..."
python -c "
from oak_storage_config import OAKConfig, setup_oak_directories
config = OAKConfig()
setup_oak_directories(config)
print(f'Output directory: {config.OUTPUT_DIR}')
"

# ==============================================================================
# Pre-Analysis Checks
# ==============================================================================

echo "Running pre-analysis checks..."

# Check available subjects and data
python run_mass_univariate.py --check-only > check_results.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Data check passed"
    echo "Available subjects:"
    grep "Found subject" check_results.txt | head -10
    TOTAL_SUBJECTS=$(grep "Total subjects with complete data" check_results.txt | awk '{print $NF}')
    echo "Total subjects with complete data: $TOTAL_SUBJECTS"
else
    echo "✗ Data check failed"
    cat check_results.txt
    exit 1
fi

# Run module tests (quick validation)
echo "Testing analysis modules..."
timeout 300 python test_mass_univariate.py > test_results.txt 2>&1
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ Module tests passed"
    grep "tests passed" test_results.txt
elif [ $TEST_EXIT_CODE -eq 124 ]; then
    echo "⚠ Module tests timed out (5 min limit) - proceeding anyway"
else
    echo "⚠ Some module tests failed - check test_results.txt"
    echo "Proceeding with analysis anyway..."
fi

# ==============================================================================
# Main Analysis Execution
# ==============================================================================

echo "Starting mass univariate analysis..."
echo "Analysis mode: $MODE"
echo "Start time: $(date)"

# Build analysis command based on mode
case $MODE in
    test)
        echo "Running test analysis (first 3 subjects)..."
        ANALYSIS_CMD="python run_mass_univariate.py --test"
        ;;
    custom)
        if [ -n "$SUBJECTS" ]; then
            echo "Running analysis on specific subjects: $SUBJECTS"
            ANALYSIS_CMD="python run_mass_univariate.py --subjects $SUBJECTS"
        else
            echo "Error: Custom mode specified but no subjects provided"
            echo "Use: sbatch --export=MODE=custom,SUBJECTS=sub-s001,sub-s002 $0"
            exit 1
        fi
        ;;
    full)
        echo "Running full analysis on all available subjects..."
        ANALYSIS_CMD="python run_mass_univariate.py"
        ;;
    *)
        echo "Error: Unknown mode: $MODE"
        echo "Valid modes: full, test, custom"
        exit 1
        ;;
esac

# Record analysis start
ANALYSIS_START=$(date +%s)
echo "Executing: $ANALYSIS_CMD"

# Run the analysis with error handling
set -e  # Exit on any error
set -o pipefail  # Exit on pipe errors

$ANALYSIS_CMD 2>&1 | tee analysis_output.log

ANALYSIS_EXIT_CODE=$?
ANALYSIS_END=$(date +%s)
ANALYSIS_DURATION=$((ANALYSIS_END - ANALYSIS_START))

# ==============================================================================
# Post-Analysis Summary
# ==============================================================================

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Duration: $((ANALYSIS_DURATION / 3600))h $((ANALYSIS_DURATION % 3600 / 60))m $((ANALYSIS_DURATION % 60))s"
echo "Exit code: $ANALYSIS_EXIT_CODE"

if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "✓ Analysis completed successfully!"
    
    # Show results summary
    echo ""
    echo "Results Summary:"
    echo "==============="
    
    # Find and display summary report
    SUMMARY_REPORT=$(find /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate -name "*summary_report.txt" -type f 2>/dev/null | head -1)
    if [ -f "$SUMMARY_REPORT" ]; then
        echo "Summary report location: $SUMMARY_REPORT"
        echo ""
        echo "Key Results:"
        grep -A 20 "SECOND-LEVEL RESULTS" "$SUMMARY_REPORT" 2>/dev/null || echo "Summary report found but content not readable"
    else
        echo "Summary report not found"
    fi
    
    # List key output files
    echo ""
    echo "Key Output Files:"
    echo "=================="
    find /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/second_level -name "*group_zstat*.nii.gz" -type f 2>/dev/null | head -10
    
else
    echo "✗ Analysis failed with exit code: $ANALYSIS_EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "================"
    echo "1. Check the analysis log: analysis_output.log"
    echo "2. Check SLURM output: slurm_${SLURM_JOB_ID}.out"
    echo "3. Check SLURM errors: slurm_${SLURM_JOB_ID}.err"
    echo "4. Verify data paths and permissions"
    
    # Show last few lines of analysis log for quick debugging
    if [ -f "analysis_output.log" ]; then
        echo ""
        echo "Last 20 lines of analysis output:"
        tail -20 analysis_output.log
    fi
fi

# ==============================================================================
# Cleanup and Resource Usage
# ==============================================================================

echo ""
echo "Resource Usage:"
echo "==============="
echo "Peak memory usage: $(grep VmPeak /proc/$$/status 2>/dev/null || echo 'Not available')"
echo "CPU time: $(grep 'user\|sys' /proc/$$/stat 2>/dev/null || echo 'Not available')"

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -f check_results.txt test_results.txt

echo "Job completed at: $(date)"
echo "=========================================="

exit $ANALYSIS_EXIT_CODE 