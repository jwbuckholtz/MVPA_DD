# Getting Started with the MVPA Pipeline

## 🎯 Overview

This guide will walk you through everything you need to know to start using the Delay Discounting MVPA Analysis Pipeline. No prior experience with the pipeline is assumed.

**What this pipeline does:**
- Analyzes delay discounting behavior from psychological experiments
- Extracts neural patterns from fMRI brain imaging data
- Performs machine learning analyses to decode choice patterns from brain activity
- Creates visualizations and statistical reports of your results

**Time to complete setup:** ~30-45 minutes  
**Time for first analysis:** ~2-6 hours (depending on data size and computing resources)

---

## 📋 Prerequisites

### What You Need

#### **Data Requirements**
- **fMRI Data**: Preprocessed functional MRI data (preferably from fMRIPrep)
- **Behavioral Data**: Choice data from delay discounting experiments
- **Anatomical Masks**: Brain region masks (ROIs) - included with pipeline

#### **Computing Requirements**
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: At least 8GB RAM (16GB+ recommended for larger datasets)
- **Storage**: 10-50GB free space (depending on your dataset size)
- **Python**: Version 3.8 or newer

#### **Access Requirements**
- Access to Stanford's OAK storage system (if using the provided dataset)
- OR your own preprocessed fMRI and behavioral data

#### **Knowledge Requirements**
- Basic command line usage (we'll guide you through this)
- Basic understanding of your experimental design
- No programming experience required for basic usage

---

## 🚀 Step 1: Environment Setup

### Option A: Stanford Sherlock HPC (Recommended for Large Datasets)

If you have access to Stanford's Sherlock computing cluster:

```bash
# 1. Log into Sherlock
ssh username@login.sherlock.stanford.edu

# 2. Navigate to your scratch directory (recommended for analysis)
cd $SCRATCH
mkdir delay_discounting_analysis
cd delay_discounting_analysis

# 3. Clone or download the pipeline
# (Replace with actual repository location)
git clone /path/to/mvpa_pipeline .
# OR copy from existing location
cp -r /oak/stanford/groups/russpold/users/shared/mvpa_pipeline/* .
```

### Option B: Stanford OAK/Farmshare Users

If you're using OAK storage or Farmshare:

```bash
# 1. Log into your computing environment
ssh username@farmshare.stanford.edu

# 2. Navigate to your project directory
cd $HOME
mkdir delay_discounting_analysis
cd delay_discounting_analysis

# 3. Clone or download the pipeline
# (Replace with actual repository location)
git clone /path/to/mvpa_pipeline .
# OR copy from existing location
cp -r /oak/stanford/groups/russpold/users/shared/mvpa_pipeline/* .
```

### Option C: Local Installation

If you're working on your own computer:

```bash
# 1. Create a project directory
mkdir delay_discounting_analysis
cd delay_discounting_analysis

# 2. Download the pipeline files
# (Download the pipeline files to this directory)

# 3. Verify you have Python 3.8+
python3 --version
# Should show: Python 3.8.x or higher
```

### Setting Up Python Environment

#### For Sherlock Users

```bash
# 1. Load required modules
module load python/3.9.0
module load py-pip/21.1.2_py39

# 2. Create a virtual environment in your scratch directory
cd $SCRATCH/delay_discounting_analysis
python3 -m venv mvpa_env

# 3. Activate the environment
source mvpa_env/bin/activate

# 4. Verify activation (your prompt should show "(mvpa_env)")
which python
# Should show: $SCRATCH/delay_discounting_analysis/mvpa_env/bin/python
```

#### For OAK/Farmshare and Local Users

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv mvpa_env

# 2. Activate the environment
# On Linux/Mac:
source mvpa_env/bin/activate
# On Windows (if using WSL):
source mvpa_env/bin/activate

# 3. Verify activation (your prompt should show "(mvpa_env)")
which python
# Should show: /path/to/mvpa_env/bin/python
```

---

## 🖥️ Sherlock HPC Usage Guide

### Understanding Sherlock Resources

**Sherlock** is Stanford's high-performance computing cluster, ideal for computationally intensive MVPA analyses.

#### **Key Sherlock Concepts:**
- **Login nodes**: For setup, job submission, light tasks only
- **Compute nodes**: Where your analysis actually runs
- **SLURM**: Job scheduler that manages compute resources
- **Partitions**: Different types of compute resources (normal, bigmem, gpu)
- **Storage**: $HOME (limited), $SCRATCH (temporary, fast), $OAK (permanent storage)

#### **Resource Guidelines:**
- **Memory**: 32-64GB for typical MVPA analyses
- **CPU**: 8-16 cores for parallel processing
- **Time**: 4-12 hours for full dataset analysis
- **Storage**: Use $SCRATCH for analysis, $OAK for data/results

### Sherlock-Specific Setup

#### **1. Initial Environment Setup**

```bash
# Log into Sherlock
ssh username@login.sherlock.stanford.edu

# Navigate to scratch (fast storage for analysis)
cd $SCRATCH

# Create analysis directory
mkdir delay_discounting_analysis
cd delay_discounting_analysis

# Copy pipeline from shared location (if available)
cp -r /oak/stanford/groups/russpold/users/shared/mvpa_pipeline/* .

# OR clone from repository
git clone /path/to/repository .
```

#### **2. Load Required Modules**

```bash
# Load Python and essential modules
module load python/3.9.0
module load py-pip/21.1.2_py39
module load py-numpy/1.20.3_py39
module load py-scipy/1.7.0_py39

# Optional: Load neuroimaging modules if available
module load py-nibabel/3.2.1_py39
module load py-nilearn/0.8.1_py39

# See available modules
module avail python
module avail py-
```

#### **3. Configure Data Paths for Sherlock**

Create Sherlock-specific configuration:

```bash
# Create Sherlock-specific config
cp config.yaml config_sherlock.yaml
```

Edit `config_sherlock.yaml`:

```yaml
# Sherlock-specific configuration
data_paths:
  fmri_data_dir: "/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/fmriprep"
  behavioral_data_dir: "/oak/stanford/groups/russpold/data/uh2/aim1/behavioral_data/event_files"
  mask_dir: "./masks"
  
# Sherlock resource settings
cluster_settings:
  use_slurm: true
  max_parallel_jobs: 8
  memory_per_job: "32GB"
  time_limit: "08:00:00"
  partition: "normal"
  
# Analysis parameters optimized for Sherlock
analysis_params:
  cv_folds: 5
  n_permutations: 1000
  tr: 2.0
  hemi_lag: 4.0
  parallel_backend: "slurm"
  
# Output settings
output:
  results_dir: "$SCRATCH/delay_discounting_analysis/results"
  save_plots: true
  verbose: true
```

#### **4. Interactive vs Batch Analysis on Sherlock**

##### **Interactive Mode (for testing)**

```bash
# Request interactive session for testing
sdev -t 2:00:00 -m 16GB -c 4

# Wait for allocation, then run quick test
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 --config config_sherlock.yaml
```

##### **Batch Mode (for production)**

Create SLURM job script `submit_mvpa_sherlock.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=mvpa_analysis
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --output=logs/mvpa_%j.out
#SBATCH --error=logs/mvpa_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@stanford.edu

# Load modules
module load python/3.9.0
module load py-pip/21.1.2_py39

# Activate virtual environment
cd $SCRATCH/delay_discounting_analysis
source mvpa_env/bin/activate

# Set configuration for Sherlock
export MVPA_CONFIG_FILE=config_sherlock.yaml

# Create logs directory
mkdir -p logs

# Run analysis
echo "Starting MVPA analysis at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python3 delay_discounting_mvpa_pipeline.py

echo "Analysis completed at $(date)"
```

Submit the job:

```bash
# Submit batch job
sbatch submit_mvpa_sherlock.sh

# Check job status
squeue -u $USER

# Check job details
scontrol show job JOBID
```

### Sherlock Analysis Workflow

#### **1. Development and Testing**

```bash
# 1. Start interactive session
sdev -t 1:00:00 -m 8GB -c 2

# 2. Test with minimal data
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_002 --config config_sherlock.yaml

# 3. Validate results
python3 run_tests.py --fast

# 4. Exit interactive session
exit
```

#### **2. Production Analysis**

```bash
# 1. Prepare batch script
nano submit_mvpa_sherlock.sh

# 2. Submit production job
sbatch submit_mvpa_sherlock.sh

# 3. Monitor progress
squeue -u $USER
tail -f logs/mvpa_JOBID.out

# 4. Check for completion
ls results/
```

#### **3. Post-Analysis**

```bash
# 1. Copy results to permanent storage
cp -r results/ /oak/stanford/groups/russpold/users/$USER/delay_discounting_results/

# 2. Generate visualization
python3 analyze_results.py --config config_sherlock.yaml

# 3. Clean up scratch space (optional)
rm -rf $SCRATCH/delay_discounting_analysis/temp_files/
```

### Sherlock Resource Management

#### **1. Choosing Partitions**

```bash
# Check available partitions
sinfo

# Normal partition (most analyses)
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --mem=32GB

# Big memory partition (memory-intensive)
#SBATCH --partition=bigmem
#SBATCH --time=12:00:00
#SBATCH --mem=128GB

# GPU partition (if using GPU acceleration)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
```

#### **2. Memory and CPU Guidelines**

| Analysis Type | Memory | CPUs | Time | Partition |
|---------------|--------|------|------|-----------|
| Single subject test | 8GB | 2 | 1h | normal |
| Small dataset (5-10 subjects) | 16GB | 4 | 4h | normal |
| Medium dataset (20-50 subjects) | 32GB | 8 | 8h | normal |
| Large dataset (100+ subjects) | 64GB | 16 | 12h | bigmem |
| Memory-intensive geometry | 128GB | 8 | 16h | bigmem |

#### **3. Monitoring Jobs**

```bash
# Check job queue
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# Job efficiency after completion
seff JOBID

# Cancel job if needed
scancel JOBID

# Check job history
sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
```

### Sherlock Troubleshooting

#### **1. Common SLURM Issues**

**Job fails immediately:**
```bash
# Check error logs
cat logs/mvpa_JOBID.err

# Check job details
scontrol show job JOBID

# Verify resource availability
sinfo -p normal
```

**Out of memory errors:**
```bash
# Check memory usage
seff JOBID

# Increase memory in SLURM script
#SBATCH --mem=64GB

# Or use memory-efficient mode
python3 delay_discounting_mvpa_pipeline.py --memory-efficient
```

**Job timeout:**
```bash
# Check current time limit
scontrol show job JOBID | grep TimeLimit

# Increase time in SLURM script
#SBATCH --time=12:00:00

# Or process fewer subjects per job
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_010
```

#### **2. Module and Environment Issues**

**Module not found:**
```bash
# List available modules
module avail python
module avail py-

# Load correct versions
module load python/3.9.0
module load py-pip/21.1.2_py39
```

**Package import errors:**
```bash
# Check virtual environment
which python
pip list

# Reinstall in virtual environment
source mvpa_env/bin/activate
pip install --upgrade -r requirements.txt
```

#### **3. Data Access Issues**

**Cannot access OAK data:**
```bash
# Check OAK mount
ls /oak/stanford/groups/russpold/

# Verify permissions
ls -la /oak/stanford/groups/russpold/data/

# Contact research computing if issues persist
```

**Slow data access:**
```bash
# Copy frequently accessed data to SCRATCH
cp -r /oak/stanford/groups/russpold/data/subset/ $SCRATCH/

# Update config to use SCRATCH copy
# data_paths:
#   fmri_data_dir: "$SCRATCH/subset/fmri_data"
```

---

## 🧠 Advanced Geometry Analysis (NEW!)

The pipeline now includes comprehensive geometry analysis capabilities with multiple analysis modes.

### Understanding Geometry Analysis Options

#### **Standard Analysis** (Default)
- Choice comparisons (smaller-sooner vs larger-later)
- Delay length analysis
- Subjective value comparisons
- Basic dimensionality reduction and RSA

#### **Trajectory Analysis** (Advanced)
- Manifold alignment across conditions
- Centroid trajectory analysis
- Information geometry metrics (KL, JS divergence, Wasserstein)
- Curvature evolution analysis
- 6-panel advanced visualizations

#### **Comprehensive Analysis** (Complete)
- Combines both standard and trajectory analysis
- Full geometric characterization
- Maximum insight into neural geometry

### Running Geometry Analysis

#### **Option 1: Integrated with Main Pipeline**
```bash
# Geometry runs automatically as part of MVPA pipeline
python delay_discounting_mvpa_pipeline.py
```

#### **Option 2: Standalone Geometry Analysis**
```bash
# Standard geometry analysis only
python delay_discounting_geometry_analysis.py --example

# Advanced trajectory analysis only  
python delay_discounting_geometry_analysis.py --example --trajectory-only

# Comprehensive analysis (recommended for research)
python delay_discounting_geometry_analysis.py --example --comprehensive

# Standard analysis + trajectory as additional output
python delay_discounting_geometry_analysis.py --example --trajectory
```

#### **Option 3: Custom Analysis Development**
```python
from geometry_utils import (
    compute_manifold_alignment,
    compute_information_geometry_metrics,
    analyze_trajectory_dynamics
)

# Custom geometric analysis using utility functions
alignment_results = compute_manifold_alignment(condition1_data, condition2_data)
info_metrics = compute_information_geometry_metrics(condition1_data, condition2_data)
```

### Geometry Analysis Outputs

#### **Standard Analysis Results**
- `geometry_summary.csv` - Quantitative metrics for all comparisons
- `*_geometry.png` - Visualization plots for each comparison
- Statistical significance testing with permutation tests

#### **Trajectory Analysis Results**
- `trajectory_analysis_summary.json` - Comprehensive trajectory metrics
- `*_advanced_geometry.png` - 6-panel trajectory visualizations
- Manifold alignment matrices and curvature evolution data

#### **When to Use Each Mode**

| Research Question | Recommended Mode | Command |
|------------------|------------------|---------|
| **Basic neural geometry** | Standard | `--example` |
| **Manifold dynamics** | Trajectory Only | `--example --trajectory-only` |
| **Complete characterization** | Comprehensive | `--example --comprehensive` |
| **Standard + bonus trajectory** | Standard + Trajectory | `--example --trajectory` |

### Sherlock Best Practices

#### **1. Efficient Resource Usage**

- **Use SCRATCH for analysis**: Faster I/O than OAK
- **Request appropriate resources**: Don't over-request memory/time
- **Use array jobs for multiple subjects**: More efficient than separate jobs
- **Monitor job efficiency**: Use `seff JOBID` after completion

#### **2. Data Management**

- **Copy results to OAK**: SCRATCH is temporary (90-day purge)
- **Organize by date**: Create timestamped result directories
- **Compress large outputs**: Use `tar -czf` for archival
- **Clean up temporary files**: Remove intermediate data

#### **3. Job Optimization**

```bash
# Array job for multiple subjects
#SBATCH --array=1-25
#SBATCH --job-name=mvpa_subject_%a

# In the script:
SUBJECT_LIST=(subject_001 subject_002 subject_003 ...)
SUBJECT=${SUBJECT_LIST[$SLURM_ARRAY_TASK_ID-1]}
python3 delay_discounting_mvpa_pipeline.py --subjects $SUBJECT
```

#### **4. Collaboration and Sharing**

```bash
# Share results with lab members
cp -r results/ /oak/stanford/groups/russpold/shared/mvpa_results/$(date +%Y%m%d)/

# Set proper permissions
chmod -R g+rw /oak/stanford/groups/russpold/shared/mvpa_results/
```

---

## 📦 Step 2: Install Dependencies

### Install Required Packages

```bash
# Make sure you're in the pipeline directory and virtual environment is active
pip install --upgrade pip
pip install -r requirements.txt
```

**What this installs:**
- **Neuroimaging tools**: nibabel, nilearn for handling brain imaging data
- **Machine learning**: scikit-learn for MVPA analyses
- **Data analysis**: numpy, pandas, scipy for data processing
- **Visualization**: matplotlib, seaborn for creating plots
- **Testing**: pytest for running tests (optional but recommended)

### Verify Installation

```bash
# Test that key packages work
python3 -c "import numpy, pandas, sklearn, nibabel, nilearn; print('✓ All packages installed successfully')"
```

If you see any errors, refer to the [Troubleshooting](#troubleshooting) section.

---

## 🔍 Step 3: Validate Your Setup

### Test the Pipeline

```bash
# Run the setup validation
python3 demo_testing.py
```

**Expected output:**
```
MVPA Pipeline Testing System Demo
==================================================
✓ pytest is available
✓ test_pipeline_pytest.py
✓ test_pipeline.py
✓ run_tests.py
✓ pytest.ini
```

### Run Quick Tests (Optional but Recommended)

```bash
# Run fast tests to verify everything works
python3 run_tests.py --fast --no-cov
```

**Expected output:**
```
Running MVPA Pipeline Tests...
==================================================
...
All tests passed! ✓
```

If tests fail, check the [Troubleshooting](#troubleshooting) section.

---

## 📊 Step 4: Prepare Your Data

### Understanding Data Structure

The pipeline expects your data in this structure:

```
your_data/
├── fmri_data/
│   ├── subject_001/
│   │   └── func/
│   │       ├── subject_001_task-discountFix_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
│   │       └── subject_001_task-discountFix_run-1_desc-confounds_timeseries.tsv
│   ├── subject_002/
│   └── ...
└── behavioral_data/
    ├── subject_001_discountFix_events.tsv
    ├── subject_002_discountFix_events.tsv
    └── ...
```

### Check Your Data

#### For Stanford OAK Users:

```bash
# Validate that ROI masks exist and are correct
python3 validate_roi_masks.py
```

**Expected output:**
```
Validating ROI masks...
✓ striatum_mask.nii.gz - Valid
✓ dlpfc_mask.nii.gz - Valid  
✓ vmpfc_mask.nii.gz - Valid
Mask validation complete!
```

#### For Your Own Data:

1. **Place your ROI masks** in the `masks/` directory
2. **Update configuration** (see next step)
3. **Test data loading**:

```bash
# Test data loading with your first subject
python3 -c "
from analysis_base import AnalysisFactory
from config_loader import load_config

config = load_config()
analysis = AnalysisFactory.create('behavioral', config=config)

# Replace 'your_subject_id' with an actual subject ID
try:
    data = analysis.load_behavioral_data('your_subject_id')
    print(f'✓ Successfully loaded {len(data)} trials for your_subject_id')
except Exception as e:
    print(f'✗ Error loading data: {e}')
"
```

---

## ⚙️ Step 5: Configure the Pipeline

### Basic Configuration

The pipeline uses a configuration file that you can customize:

```bash
# Look at the default configuration
cat config.yaml
```

### Common Settings to Modify

Edit `config.yaml` or create your own configuration:

```yaml
# Data paths - UPDATE THESE FOR YOUR DATA
data_paths:
  fmri_data_dir: "/path/to/your/fmri_data"
  behavioral_data_dir: "/path/to/your/behavioral_data"
  mask_dir: "./masks"

# Analysis parameters
analysis_params:
  cv_folds: 5                # Cross-validation folds
  n_permutations: 1000       # Statistical permutations
  tr: 2.0                    # Repetition time in seconds
  hemi_lag: 4.0             # Hemodynamic lag in TRs

# ROI masks - UPDATE PATHS IF NEEDED
roi_masks:
  striatum: "./masks/striatum_mask.nii.gz"
  dlpfc: "./masks/dlpfc_mask.nii.gz"
  vmpfc: "./masks/vmpfc_mask.nii.gz"

# Output settings
output:
  results_dir: "./results"
  save_plots: true
  verbose: true
```

### Advanced Configuration (Optional)

```bash
# Create environment-specific configuration
cp config.yaml config_development.yaml

# Set environment variable to use your config
export MVPA_CONFIG_ENV=development
```

---

## 🎯 Step 6: Run Your First Analysis

### Option A: Interactive Analysis (Recommended for Beginners)

```bash
# Start the interactive pipeline
python3 delay_discounting_mvpa_pipeline.py
```

**What happens:**
1. The pipeline will ask you to confirm settings
2. It will discover subjects in your data
3. It will run behavioral analysis first
4. Then MVPA analysis on brain data
5. Finally generate results and plots

**Example interaction:**
```
Delay Discounting MVPA Pipeline
==============================

Found configuration: config.yaml
Found 25 subjects with complete data

Run behavioral analysis? [Y/n]: Y
Run MVPA analysis? [Y/n]: Y
Run geometry analysis? [Y/n]: n

Starting analysis...
[1/25] Processing subject_001... ✓ (45.2s)
[2/25] Processing subject_002... ✓ (43.8s)
...
```

### Option B: Command Line Analysis

```bash
# Run specific analysis types
python3 delay_discounting_mvpa_pipeline.py --behavioral-only
python3 delay_discounting_mvpa_pipeline.py --mvpa-only
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_002
```

### Option C: Using Analysis Framework (Advanced)

```bash
# Use the new analysis framework
python3 -c "
from analysis_base import AnalysisFactory
from config_loader import load_config

# Load configuration
config = load_config()

# Create behavioral analysis
behavioral = AnalysisFactory.create('behavioral', config=config)

# Run analysis on subset of subjects
results = behavioral.run_analysis(subjects=['subject_001', 'subject_002'])

print(f'Analyzed {len(results[\"results\"])} subjects')
print(f'Success rate: {results[\"summary\"][\"success_rate\"]:.1%}')
"
```

---

## 📈 Step 7: Understanding Your Results

### Results Directory Structure

After running analysis, you'll see:

```
results/
├── behavioral_analysis/
│   ├── behavioral_summary.csv          # Summary statistics
│   ├── behavioral_parameters.png       # Distribution plots
│   └── individual_subjects/            # Per-subject results
├── mvpa_analysis/
│   ├── mvpa_summary.csv               # Decoding accuracies
│   ├── roi_decoding_results.png       # Accuracy by ROI
│   └── subject_results/               # Detailed results
└── analysis_report.txt                # Overall summary
```

### Key Output Files

#### **1. Behavioral Summary (`behavioral_summary.csv`)**
```csv
subject_id,k_value,pseudo_r2,n_trials,choice_rate,processing_time
subject_001,0.025,0.45,120,0.63,2.1
subject_002,0.018,0.52,118,0.58,1.9
...
```

**What this means:**
- `k_value`: Discount rate (higher = more impatient)
- `pseudo_r2`: Model fit quality (higher = better fit)
- `choice_rate`: Proportion of larger-later choices

#### **2. MVPA Summary (`mvpa_summary.csv`)**
```csv
subject_id,roi_name,analysis_type,score,p_value,n_trials,n_voxels
subject_001,striatum,choice_decoding,0.68,0.021,120,1247
subject_001,dlpfc,choice_decoding,0.72,0.008,120,892
...
```

**What this means:**
- `score`: Decoding accuracy (>0.5 = above chance)
- `p_value`: Statistical significance (p < 0.05 = significant)
- `roi_name`: Brain region analyzed

#### **3. Analysis Report (`analysis_report.txt`)**
```
Delay Discounting MVPA Analysis Report
=====================================

Dataset Summary:
- Total subjects analyzed: 25
- Behavioral analysis success rate: 96% (24/25)
- MVPA analysis success rate: 92% (23/25)

Behavioral Results:
- Mean discount rate (k): 0.032 ± 0.028
- Mean model fit (R²): 0.48 ± 0.15

MVPA Results:
- Striatum choice decoding: 65.2% ± 8.1% (p < 0.001)
- DLPFC choice decoding: 67.8% ± 9.3% (p < 0.001)
- VMPFC choice decoding: 61.4% ± 7.9% (p = 0.023)
```

### Interpreting Results

#### **Behavioral Analysis:**
- **k-values 0.01-0.05**: Normal range for discount rates
- **k-values >0.1**: Very impatient (steep discounting)
- **k-values <0.005**: Very patient (shallow discounting)
- **R² >0.3**: Good model fit
- **R² <0.2**: Poor model fit (consider excluding)

#### **MVPA Analysis:**
- **Accuracy >60%**: Strong decoding
- **Accuracy 55-60%**: Moderate decoding  
- **Accuracy 50-55%**: Weak decoding
- **Accuracy <50%**: No decoding (below chance)
- **p < 0.05**: Statistically significant
- **p < 0.001**: Highly significant

---

## 📊 Step 8: Visualize Your Results

### Generate Standard Plots

```bash
# Create comprehensive visualization report
python3 analyze_results.py
```

**Generated plots:**
- `behavioral_distributions.png` - Histogram of discount rates
- `mvpa_accuracy_by_roi.png` - Bar plot of decoding accuracies
- `subject_performance_heatmap.png` - Individual subject results

### Custom Visualization

```python
# Create custom plots
import pandas as pd
import matplotlib.pyplot as plt

# Load your results
behavioral = pd.read_csv('results/behavioral_analysis/behavioral_summary.csv')
mvpa = pd.read_csv('results/mvpa_analysis/mvpa_summary.csv')

# Plot discount rate distribution
plt.figure(figsize=(10, 6))
plt.hist(behavioral['k_value'], bins=20, alpha=0.7)
plt.xlabel('Discount Rate (k)')
plt.ylabel('Number of Subjects')
plt.title('Distribution of Discount Rates')
plt.savefig('my_custom_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🔧 Step 9: Advanced Usage

### Running Tests

```bash
# Run full test suite
python3 run_tests.py --coverage --html

# Run specific test categories
python3 run_tests.py --behavioral
python3 run_tests.py --mvpa
python3 run_tests.py --integration

# Generate test coverage report
open htmlcov/index.html  # View coverage in browser
```

### Batch Processing

```bash
# Process specific subjects
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_005 subject_010

# Run on compute cluster (if available)
sbatch submit_analysis_job.sh
```

### Analysis Framework Usage

```python
from analysis_base import AnalysisFactory

# Create different analysis types
behavioral = AnalysisFactory.create('behavioral', config=config)
mvpa = AnalysisFactory.create('mvpa', config=config)
geometry = AnalysisFactory.create('geometry', config=config)

# Run analyses
behavioral_results = behavioral.run_analysis(['subject_001'])
mvpa_results = mvpa.run_analysis(['subject_001'])

# Save results
behavioral.save_results('my_behavioral_results.pkl')
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### **1. Import Errors**
```
Error: ModuleNotFoundError: No module named 'sklearn'
```
**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### **2. Data Loading Errors**
```
Error: FileNotFoundError: No such file or directory
```
**Solution:**
- Check that your data paths in `config.yaml` are correct
- Verify file permissions
- Make sure files exist with correct naming

#### **3. Memory Errors**
```
Error: MemoryError: Unable to allocate array
```
**Solution:**
```bash
# Use memory-efficient options
python3 delay_discounting_mvpa_pipeline.py --memory-efficient

# Or process fewer subjects at once
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_002
```

#### **4. SLURM Job Failures**
```
Error: Job failed with exit code 1
```
**Solution:**
- Check SLURM log files in `logs/` directory
- Reduce memory requirements in submit script
- Check data accessibility on compute nodes

#### **5. ROI Mask Issues**
```
Error: Cannot load mask file
```
**Solution:**
```bash
# Validate masks
python3 validate_roi_masks.py

# Check mask files exist
ls -la masks/
```

### Getting Help

#### **Check Log Files**
```bash
# Look for error logs
ls logs/
cat logs/latest_analysis.log
```

#### **Test Individual Components**
```bash
# Test behavioral analysis only
python3 run_tests.py --behavioral

# Test data loading
python3 -c "from data_utils import check_data_integrity; check_data_integrity(['subject_001'])"
```

#### **Contact Support**
- Check existing documentation in `TESTING_GUIDE.md`
- Review configuration examples in demo scripts
- Contact: Cognitive Neuroscience Lab, Stanford University

---

## 📝 Step 10: What's Next?

### Immediate Next Steps

1. **Analyze Your Full Dataset**
   ```bash
   # Run on all subjects
   python3 delay_discounting_mvpa_pipeline.py
   ```

2. **Explore Results**
   - Review statistical significance of findings
   - Compare results across brain regions
   - Examine individual subject patterns

3. **Generate Publication Figures**
   ```bash
   # Create high-quality plots
   python3 analyze_results.py --publication-quality
   ```

### Advanced Analyses

#### **1. Geometry Analysis**
```bash
# Run neural geometry analysis
python3 delay_discounting_geometry_analysis.py --example
```

#### **2. Custom ROIs**
- Add your own brain region masks to `masks/` directory
- Update `config.yaml` with new ROI paths
- Re-run MVPA analysis

#### **3. Additional Variables**
- Modify behavioral analysis to extract new variables
- Add custom MVPA decoding targets
- Implement new visualization methods

### Learn More

#### **Documentation to Read Next:**
- `ANALYSIS_CLASSES_README.md` - Advanced analysis framework usage
- `TESTING_GUIDE.md` - Comprehensive testing documentation
- `DATA_UTILS_README.md` - Data handling utilities
- `MVPA_UTILS_README.md` - Machine learning procedures

#### **Example Scripts to Explore:**
- `demo_analysis_classes.py` - Analysis framework examples
- `demo_data_utils.py` - Data utilities demonstration
- `demo_mvpa_utils.py` - MVPA procedures examples

---

## 📚 Quick Reference

### Essential Commands
```bash
# Setup
pip install -r requirements.txt

# Validate setup
python3 demo_testing.py

# Run analysis
python3 delay_discounting_mvpa_pipeline.py

# Check results
python3 analyze_results.py

# Run tests
python3 run_tests.py --fast
```

### Key Configuration Files
- `config.yaml` - Main configuration
- `requirements.txt` - Python dependencies
- `pytest.ini` - Test configuration

### Important Output Files
- `results/behavioral_summary.csv` - Behavioral results
- `results/mvpa_summary.csv` - MVPA decoding results
- `results/analysis_report.txt` - Summary report

### Help Commands
```bash
# Get help for scripts
python3 delay_discounting_mvpa_pipeline.py --help
python3 run_tests.py --help

# Check pipeline status
python3 demo_testing.py
```

---

**🎉 Congratulations!** You're now ready to run delay discounting MVPA analyses. Start with a small subset of subjects to familiarize yourself with the pipeline, then scale up to your full dataset.

For questions or issues, refer to the troubleshooting section above or check the comprehensive documentation provided with the pipeline. 