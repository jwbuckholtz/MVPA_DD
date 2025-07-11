# Complete Pipeline Guide for Sherlock HPC

## 🎯 Overview

This guide walks you through running the complete MVPA and geometry analysis pipeline on Stanford's Sherlock HPC cluster. This assumes you have:

- ✅ **Configuration ready**: Your `config_sherlock.yaml` file with correct data paths
- ✅ **SLURM script ready**: Your `submit_analysis_job.sh` script configured
- ✅ **Data access**: Permissions to access your fMRI and behavioral data
- ✅ **Sherlock account**: Active Stanford Sherlock computing account

**Time to complete**: 4-12 hours (depending on dataset size)  
**Expected output**: Complete MVPA results + comprehensive geometry analysis

---

## 📋 Pre-Flight Checklist

Before starting, verify you have these ready:

### **1. Configuration File (`config_sherlock.yaml`)**
```yaml
# Your config should have these key sections:
data_paths:
  fmri_data_dir: "/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/fmriprep"
  behavioral_data_dir: "/oak/stanford/groups/russpold/data/uh2/aim1/behavioral_data/event_files"
  mask_dir: "./masks"

cluster_settings:
  use_slurm: true
  max_parallel_jobs: 8
  memory_per_job: "32GB"
  time_limit: "08:00:00"
  partition: "normal"

output:
  results_dir: "$SCRATCH/delay_discounting_analysis/results"
  save_plots: true
  verbose: true
```

### **2. SLURM Submission Script (`submit_analysis_job.sh`)**
```bash
#!/bin/bash
#SBATCH --job-name=mvpa_complete
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

# Environment setup
cd $SCRATCH/delay_discounting_analysis
source mvpa_env/bin/activate
mkdir -p logs

# Run complete pipeline
echo "=== Starting Complete MVPA + Geometry Pipeline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Main MVPA pipeline (includes basic geometry)
python3 delay_discounting_mvpa_pipeline.py --config config_sherlock.yaml

# Advanced geometry analysis
echo "=== Starting Advanced Geometry Analysis ==="
python3 delay_discounting_geometry_analysis.py --example --comprehensive

echo "=== Pipeline completed at $(date) ==="
```

---

## 🚀 Step-by-Step Execution

### **Step 1: Login and Environment Setup**

```bash
# 1. Login to Sherlock
ssh username@login.sherlock.stanford.edu

# 2. Navigate to your analysis directory
cd $SCRATCH/delay_discounting_analysis

# 3. Check your setup
ls -la  # Should see your pipeline files
cat config_sherlock.yaml  # Verify your configuration
cat submit_analysis_job.sh  # Verify your SLURM script

# 4. Load required modules
module load python/3.9.0
module load py-pip/21.1.2_py39

# 5. Activate your virtual environment
source mvpa_env/bin/activate

# 6. Verify environment
which python  # Should show path in mvpa_env
pip list | grep -E "(numpy|sklearn|nilearn)"  # Check key packages
```

### **Step 2: Pre-Job Validation (Optional but Recommended)**

Before submitting the full job, run a quick validation:

```bash
# 1. Request a short interactive session
sdev -t 30:00 -m 8GB -c 2

# 2. Test with minimal data
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 --config config_sherlock.yaml

# 3. Verify outputs are created
ls results/
ls results/behavioral_analysis/
ls results/mvpa_analysis/
ls results/geometry_analysis/

# 4. Test geometry analysis
python3 delay_discounting_geometry_analysis.py --example --config config_sherlock.yaml

# 5. Exit interactive session
exit
```

### **Step 3: Submit Production Job**

```bash
# 1. Double-check your SLURM script
nano submit_analysis_job.sh  # Make any final edits

# 2. Check cluster status
sinfo -p normal  # Check partition availability
squeue  # Check current queue load

# 3. Submit your job
sbatch submit_analysis_job.sh

# 4. Note your job ID
# Example output: "Submitted batch job 12345678"
export JOB_ID=12345678  # Replace with your actual job ID
```

### **Step 4: Monitor Job Progress**

```bash
# 1. Check job status
squeue -u $USER

# Job status meanings:
# PD = Pending (waiting for resources)
# R  = Running
# CG = Completing
# CD = Completed

# 2. Monitor detailed job info
scontrol show job $JOB_ID

# 3. Watch live output (optional)
tail -f logs/mvpa_${JOB_ID}.out

# 4. Check for errors
tail -f logs/mvpa_${JOB_ID}.err

# 5. Check resource usage (while running)
sstat -j $JOB_ID --format=MaxRSS,AveCPU,Elapsed
```

### **Step 5: Track Pipeline Progress**

Your job will progress through these stages:

#### **Stage 1: MVPA Pipeline (60-80% of total time)**
```bash
# Monitor pipeline progress
grep -E "(Starting|Completed|Processing)" logs/mvpa_${JOB_ID}.out

# Look for these key milestones:
# - "Loading behavioral data"
# - "Processing subject XXX"
# - "Running MVPA analysis"
# - "Geometry analysis complete"
# - "MVPA pipeline completed"
```

#### **Stage 2: Advanced Geometry Analysis (20-40% of total time)**
```bash
# Monitor geometry analysis progress
grep -E "(geometry|trajectory|manifold)" logs/mvpa_${JOB_ID}.out

# Look for these milestones:
# - "Starting Advanced Geometry Analysis"
# - "Computing manifold alignment"
# - "Analyzing trajectory dynamics"
# - "Creating advanced visualizations"
# - "Geometry analysis completed"
```

### **Step 6: Check Job Completion**

```bash
# 1. Check final job status
squeue -u $USER  # Should show no running jobs

# 2. Check job completion details
sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Start,End,Elapsed

# 3. Check resource efficiency
seff $JOB_ID

# Look for:
# - State: COMPLETED
# - Exit Code: 0:0 (success)
# - Memory Efficiency: >60%
# - CPU Efficiency: >70%
```

### **Step 7: Verify Results**

```bash
# 1. Check results directory structure
ls -la results/
du -sh results/  # Check total size

# 2. Verify MVPA results
ls results/mvpa_analysis/
cat results/mvpa_analysis/mvpa_summary.csv  # Quick results check

# 3. Verify geometry results
ls results/geometry_analysis/
ls results/geometry_analysis/*.png  # Check visualizations

# 4. Check for advanced geometry results
ls results/geometry_analysis/*_advanced_geometry.png
cat results/geometry_analysis/trajectory_analysis_summary.json
```

---

## 📊 Understanding Your Results

### **MVPA Analysis Results**
```bash
# Key files to check:
results/mvpa_analysis/
├── mvpa_summary.csv              # Main results table
├── roi_decoding_accuracy.png     # ROI decoding performance
├── classification_results.png    # Choice classification results
├── regression_results.png        # Value regression results
└── statistical_summary.txt       # Significance testing
```

### **Geometry Analysis Results**
```bash
# Standard geometry results:
results/geometry_analysis/
├── geometry_summary.csv          # Quantitative metrics
├── ROI_choice_geometry.png        # Choice comparisons
├── ROI_delay_*_geometry.png       # Delay comparisons
├── ROI_sv_*_geometry.png          # Subjective value comparisons

# Advanced trajectory results (NEW!):
├── trajectory_analysis_summary.json           # Comprehensive metrics
├── ROI_choice_advanced_geometry.png           # 6-panel trajectory plots
├── ROI_delay_*_advanced_geometry.png          # Advanced visualizations
└── manifold_alignment_matrices.npz            # Alignment data
```

### **Quality Control Checks**
```bash
# 1. Check for successful completion
grep -i "error\|fail\|exception" logs/mvpa_${JOB_ID}.err

# 2. Verify all subjects processed
grep "Processing subject" logs/mvpa_${JOB_ID}.out | wc -l

# 3. Check MVPA accuracy
python3 -c "
import pandas as pd
df = pd.read_csv('results/mvpa_analysis/mvpa_summary.csv')
print('Mean accuracy:', df['accuracy'].mean())
print('Significant results:', (df['p_value'] < 0.05).sum())
"

# 4. Check geometry analysis completion
grep "trajectory analysis completed" logs/mvpa_${JOB_ID}.out
```

---

## 📁 Results Management

### **Archive Results to Permanent Storage**

```bash
# 1. Create timestamped archive directory
DATE=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="/oak/stanford/groups/russpold/users/$USER/mvpa_results_${DATE}"
mkdir -p $ARCHIVE_DIR

# 2. Copy results with compression
tar -czf "${ARCHIVE_DIR}/complete_analysis_results.tar.gz" results/
cp logs/mvpa_${JOB_ID}.out "${ARCHIVE_DIR}/analysis_log.txt"
cp logs/mvpa_${JOB_ID}.err "${ARCHIVE_DIR}/analysis_errors.txt"
cp config_sherlock.yaml "${ARCHIVE_DIR}/config_used.yaml"

# 3. Create results summary
echo "Analysis completed: $(date)" > "${ARCHIVE_DIR}/analysis_summary.txt"
echo "Job ID: $JOB_ID" >> "${ARCHIVE_DIR}/analysis_summary.txt"
echo "Total subjects: $(grep 'Processing subject' logs/mvpa_${JOB_ID}.out | wc -l)" >> "${ARCHIVE_DIR}/analysis_summary.txt"
echo "Results location: $ARCHIVE_DIR" >> "${ARCHIVE_DIR}/analysis_summary.txt"

# 4. Verify archive
ls -la $ARCHIVE_DIR
du -sh $ARCHIVE_DIR
```

### **Generate Analysis Report**

```bash
# Create comprehensive analysis report
python3 analyze_results.py --config config_sherlock.yaml --output-dir $ARCHIVE_DIR

# This generates:
# - Comprehensive results report
# - Statistical summaries
# - Quality control metrics
# - Visualization gallery
```

### **Clean Up Scratch Space (Optional)**

```bash
# After archiving, clean up large temporary files
rm -rf results/temp_files/  # Remove temporary processing files
rm -rf results/*/intermediate_*  # Remove intermediate results

# Keep main results and logs for immediate access
# Archive to OAK handles long-term storage
```

---

## 🚨 Troubleshooting Common Issues

### **Job Won't Start (Status: PD)**

```bash
# Check partition availability
sinfo -p normal

# Check your resource requests
scontrol show job $JOB_ID | grep -E "(Partition|TimeLimit|MinMemory)"

# Consider adjusting resources:
# - Reduce memory: #SBATCH --mem=16GB
# - Reduce time: #SBATCH --time=06:00:00
# - Try different partition: #SBATCH --partition=bigmem
```

### **Job Fails Early**

```bash
# Check error log immediately
cat logs/mvpa_${JOB_ID}.err

# Common issues and solutions:
# "Module not found" → Check module loading in script
# "Permission denied" → Check data access permissions
# "Out of memory" → Increase memory allocation
# "Import error" → Check virtual environment setup
```

### **Job Runs But No Results**

```bash
# Check if pipeline actually started
grep "Starting" logs/mvpa_${JOB_ID}.out

# Check configuration loading
grep "config" logs/mvpa_${JOB_ID}.out

# Verify data paths
ls /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/fmriprep/
```

### **Partial Results**

```bash
# Check which subjects completed
grep "Processing subject" logs/mvpa_${JOB_ID}.out
grep "completed successfully" logs/mvpa_${JOB_ID}.out

# Resume analysis for failed subjects
python3 delay_discounting_mvpa_pipeline.py --subjects failed_subject_001 failed_subject_002
```

### **Out of Memory Errors**

```bash
# Check actual memory usage
seff $JOB_ID

# Solutions:
# 1. Increase memory allocation
#SBATCH --mem=64GB

# 2. Use memory-efficient mode
python3 delay_discounting_mvpa_pipeline.py --memory-efficient --config config_sherlock.yaml

# 3. Process fewer subjects per job
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_020 --config config_sherlock.yaml
```

---

## 🎯 Next Steps After Completion

### **1. Results Validation**
```bash
# Run comprehensive validation
python3 run_tests.py --integration --config config_sherlock.yaml

# Validate specific components
python3 demo_testing.py --config config_sherlock.yaml
```

### **2. Generate Publication Figures**
```bash
# Create publication-ready visualizations
python3 analyze_results.py --publication-mode --config config_sherlock.yaml
```

### **3. Statistical Analysis**
```bash
# Run additional statistical tests
python3 analyze_results.py --statistical-analysis --config config_sherlock.yaml
```

### **4. Data Export**
```bash
# Export results for external analysis
python3 -c "
import pandas as pd
import numpy as np

# Load results
mvpa_results = pd.read_csv('results/mvpa_analysis/mvpa_summary.csv')
geometry_results = pd.read_csv('results/geometry_analysis/geometry_summary.csv')

# Export for R/MATLAB/etc
mvpa_results.to_csv('${ARCHIVE_DIR}/mvpa_results_for_export.csv')
geometry_results.to_csv('${ARCHIVE_DIR}/geometry_results_for_export.csv')

print('Results exported for external analysis')
"
```

---

## 💡 Pro Tips for Sherlock Success

### **Resource Optimization**
- **Memory**: Start with 32GB, increase if needed
- **Time**: Allow 8-12 hours for full dataset
- **CPUs**: 8 cores optimal for most analyses
- **Partition**: Use `normal` for most jobs, `bigmem` for large datasets

### **Workflow Efficiency**
- **Test first**: Always run interactive test before batch jobs
- **Monitor actively**: Check job progress every 1-2 hours
- **Archive immediately**: Move results to OAK as soon as job completes
- **Document everything**: Keep detailed notes of settings and results

### **Data Management**
- **Use SCRATCH**: Copy frequently accessed data to $SCRATCH for faster I/O
- **Compress archives**: Use tar.gz for long-term storage
- **Regular cleanup**: Remove old temporary files to save space
- **Backup configs**: Always save the exact configuration used

---

## 📞 Getting Help

### **Self-Diagnosis Steps**
1. Check this guide's troubleshooting section
2. Review error logs carefully
3. Test with single subject interactively
4. Verify environment and dependencies

### **Sherlock-Specific Support**
- **Documentation**: [Sherlock User Guide](https://www.sherlock.stanford.edu/docs/)
- **Status**: Check [Sherlock Status Page](https://status.sherlock.stanford.edu/)
- **Support**: Email research-computing-support@stanford.edu

### **Pipeline-Specific Issues**
- **Documentation**: Check README.md and other guides
- **Testing**: Run diagnostic tests with `python3 run_tests.py`
- **Configuration**: Verify all paths and settings in config_sherlock.yaml

---

**🎉 Congratulations!** You've successfully completed the full MVPA and geometry analysis pipeline on Sherlock! Your results are now ready for scientific analysis and publication. 