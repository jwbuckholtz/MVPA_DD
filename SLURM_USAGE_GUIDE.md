# SLURM Usage Guide for Mass Univariate Analysis

This guide explains how to use the `submit_mass_univariate.sh` SLURM batch script for running delay discounting mass univariate analysis on Stanford's Oak cluster.

## Quick Start

### **1. Basic Submission (All Subjects)**
```bash
sbatch submit_mass_univariate.sh
```

### **2. Test Run (First 3 Subjects)**
```bash
sbatch --export=MODE=test submit_mass_univariate.sh
```

### **3. Specific Subjects**
```bash
sbatch --export=MODE=custom,SUBJECTS=sub-s001,sub-s002,sub-s003 submit_mass_univariate.sh
```

## Configuration Options

### **Environment Variables**
You can customize the analysis by setting environment variables:

```bash
# Custom project directory
sbatch --export=PROJECT_DIR=/path/to/your/analysis submit_mass_univariate.sh

# Custom Python environment
sbatch --export=PYTHON_ENV=/path/to/your/venv submit_mass_univariate.sh

# Combined options
sbatch --export=MODE=test,PYTHON_ENV=~/my_env submit_mass_univariate.sh
```

### **Available Modes**
- **`full`** (default): Process all available subjects
- **`test`**: Process first 3 subjects only (for testing)
- **`custom`**: Process specific subjects (requires SUBJECTS variable)

## Resource Allocation

### **Default Resources**
```bash
#SBATCH --time=24:00:00          # 24 hours
#SBATCH --cpus-per-task=16       # 16 CPU cores
#SBATCH --mem=64G                # 64 GB RAM
#SBATCH --partition=russpold     # Russpold lab partition
```

### **Adjusting Resources**
For different data sizes, you can modify resources:

```bash
# For smaller datasets (< 20 subjects)
sbatch --time=8:00:00 --mem=32G --cpus-per-task=8 submit_mass_univariate.sh

# For larger datasets (> 50 subjects) 
sbatch --time=48:00:00 --mem=128G --cpus-per-task=24 submit_mass_univariate.sh

# For test runs
sbatch --time=2:00:00 --mem=16G --cpus-per-task=4 --export=MODE=test submit_mass_univariate.sh
```

## Monitoring Jobs

### **Check Job Status**
```bash
# View your jobs
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# Check job efficiency
seff <JOB_ID>
```

### **View Output**
```bash
# Real-time output (while job is running)
tail -f /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/slurm_<JOB_ID>.out

# View errors
cat /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/slurm_<JOB_ID>.err

# Analysis log (generated during run)
tail -f analysis_output.log
```

## Job Management

### **Cancel Job**
```bash
scancel <JOB_ID>
```

### **Job Priority**
```bash
# High priority (if available)
sbatch --qos=high submit_mass_univariate.sh

# Nice (lower priority)
sbatch --nice=100 submit_mass_univariate.sh
```

### **Job Dependencies**
```bash
# Run after another job completes
sbatch --dependency=afterok:<PREV_JOB_ID> submit_mass_univariate.sh

# Run test first, then full analysis
TEST_JOB=$(sbatch --export=MODE=test submit_mass_univariate.sh | cut -d' ' -f4)
sbatch --dependency=afterok:$TEST_JOB submit_mass_univariate.sh
```

## Troubleshooting

### **Common Issues**

#### **Job Fails Immediately**
```bash
# Check SLURM output for errors
cat slurm_<JOB_ID>.err

# Verify paths and permissions
ls -la /oak/stanford/groups/russpold/data/uh2/aim1/

# Test environment manually
srun --partition=russpold --time=1:00:00 --pty bash
```

#### **Out of Memory Errors**
```bash
# Increase memory allocation
sbatch --mem=128G submit_mass_univariate.sh

# Or reduce subjects per run
sbatch --export=MODE=custom,SUBJECTS=sub-s001,sub-s002 submit_mass_univariate.sh
```

#### **Time Limit Exceeded**
```bash
# Increase time limit
sbatch --time=48:00:00 submit_mass_univariate.sh

# Or run in batches
sbatch --export=MODE=custom,SUBJECTS=sub-s001,sub-s002,sub-s003 submit_mass_univariate.sh
sbatch --export=MODE=custom,SUBJECTS=sub-s004,sub-s005,sub-s006 submit_mass_univariate.sh
```

### **Debug Mode**
To debug issues, run an interactive session:

```bash
# Start interactive session
srun --partition=russpold --time=2:00:00 --mem=16G --cpus-per-task=4 --pty bash

# Load environment
module load python/3.9.0 fsl/6.0.4
source ~/venv/bin/activate

# Test manually
cd /oak/stanford/groups/russpold/data/uh2/aim1/analysis/MVPA_DD
python test_mass_univariate.py
python run_mass_univariate.py --check-only
```

## Performance Tips

### **Optimize for Your Data**
```bash
# For many small subjects: more CPUs, less memory
sbatch --cpus-per-task=20 --mem=32G submit_mass_univariate.sh

# For few large subjects: fewer CPUs, more memory  
sbatch --cpus-per-task=8 --mem=128G submit_mass_univariate.sh
```

### **Parallel Processing**
```bash
# Run multiple small batches in parallel
for i in {1..4}; do
    SUBJECTS="sub-s00$i"
    sbatch --export=MODE=custom,SUBJECTS=$SUBJECTS submit_mass_univariate.sh
done
```

### **Check Resource Usage**
After job completion:
```bash
# View resource usage
seff <JOB_ID>

# Detailed accounting
sacct -j <JOB_ID> --format=JobID,JobName,MaxRSS,Elapsed,CPUTime,ReqMem
```

## Expected Runtime

| Dataset Size | Estimated Time | Recommended Resources |
|--------------|----------------|----------------------|
| 1-5 subjects | 30-60 minutes | 4 CPUs, 16GB RAM |
| 10-20 subjects | 2-4 hours | 8 CPUs, 32GB RAM |
| 30-50 subjects | 6-12 hours | 16 CPUs, 64GB RAM |
| 50+ subjects | 12-24 hours | 24 CPUs, 128GB RAM |

## Output Files

### **SLURM Outputs**
- `slurm_<JOB_ID>.out`: Standard output from SLURM
- `slurm_<JOB_ID>.err`: Error output from SLURM  
- `analysis_output.log`: Detailed analysis log

### **Analysis Results**
Results are saved to:
```
/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/
├── smoothed_data/           # 4mm smoothed fMRI data
├── first_level/            # Subject-level results
├── second_level/           # Group-level maps
│   ├── *_group_zstat.nii.gz          # Unthresholded maps
│   ├── *_fdr0.05.nii.gz              # FDR corrected
│   └── *_cluster0.05.nii.gz          # Cluster corrected
└── complete_results.pkl     # Complete results object
```

## Email Notifications

The script automatically sends email notifications:
- **Job starts**: Job submission confirmation
- **Job ends**: Success/failure notification with summary
- **Job fails**: Error details and troubleshooting tips

To disable email notifications:
```bash
sbatch --mail-type=NONE submit_mass_univariate.sh
```

## Example Workflows

### **Development Workflow**
```bash
# 1. Test with 3 subjects
sbatch --export=MODE=test submit_mass_univariate.sh

# 2. Run small batch
sbatch --export=MODE=custom,SUBJECTS=sub-s001,sub-s002,sub-s003,sub-s004 submit_mass_univariate.sh

# 3. Full analysis
sbatch submit_mass_univariate.sh
```

### **Production Workflow**
```bash
# 1. Check data availability
python run_mass_univariate.py --check-only

# 2. Submit with dependencies
TEST_JOB=$(sbatch --export=MODE=test submit_mass_univariate.sh | cut -d' ' -f4)
FULL_JOB=$(sbatch --dependency=afterok:$TEST_JOB submit_mass_univariate.sh | cut -d' ' -f4)

# 3. Monitor progress
watch "squeue -u $USER"
```

For additional help or issues specific to your analysis, check the analysis logs or contact the cluster administrators. 