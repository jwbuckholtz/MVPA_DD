
# Quick Reference Guide

This document provides a quick reference for running the analysis pipeline.

## Running Locally (Detailed Steps)

Follow these steps to run the entire analysis pipeline on your local machine.

### 1. Setup Your Environment

First, ensure you have `uv` installed. Then, use it to create and sync your environment.

```bash
# Create and sync the virtual environment (do this once)
uv venv
uv sync
```

### 2. Activate the Environment
```bash
# Activate the virtual environment
source .venv/bin/activate
```
### 3. Configure Your Analysis

All settings for the pipeline are located in the `config.py` file. Before running the analysis, you may want to review and edit this file to:

-   **Set Data Paths**: Update `Paths.DATA_ROOT`, `Paths.FMRIPREP_DIR`, and other path variables to match the location of your data.
-   **Adjust Parameters**: Modify analysis parameters in the `Behavioral`, `MVPA`, and `Geometry` classes as needed.
-   **Select ROIs**: Change the `ROI.CORE_ROIS` list to specify which regions of interest to include in the analysis.

### 4. Run the Pipeline

The `main.py` script is the single entry point for running the pipeline.

**To run all analyses on a default set of subjects:**
```bash
python main.py
```

**To run on specific subjects:**
```bash
python main.py --subjects s001 s002 s003
```

**To skip specific analyses:**
```bash
# Example: Run only MVPA
python main.py --skip-behavioral --skip-geometry
```

### 5. View the Results

The pipeline will create output directories and save the results as CSV files.

-   **Behavioral Results**: `[OUTPUT_DIR]/behavioral_analysis/behavioral_summary.csv`
-   **MVPA Results**: `[OUTPUT_DIR]/mvpa_analysis/mvpa_summary.csv`
-   **Geometry Results**: `[OUTPUT_DIR]/geometry_analysis/geometry_summary.csv`

You can open these files with any spreadsheet program or data analysis tool (like Pandas) to view the results.

## Running on a SLURM Cluster (e.g., Sherlock)

To run the analysis on a SLURM cluster, you'll need a submission script.

### 1. SLURM Submission Script

Create a file named `submit_analysis.sh` with the following content. Be sure to replace `<your_email@domain.com>` with your email address.

```bash
#!/bin/bash
#
#SBATCH --job-name=fmri_analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email@domain.com>

# Load necessary modules
module load python/3.9

# Activate your virtual environment
source your_venv/bin/activate

# Run the analysis for a specific set of subjects
python main.py --subjects s001 s002 s003 s004 s005

echo "Analysis complete."
```

### 2. Submit the Job

To submit the job to the SLURM scheduler, use the `sbatch` command:
```bash
sbatch submit_analysis.sh
```

### 3. Check Job Status

You can check the status of your job using `squeue`:
```bash
squeue -u $USER
``` 