# MVPA Pipeline Quick Reference

## 🚀 Essential Commands

### Setup & Installation

#### Local/Farmshare
```bash
# Create and activate virtual environment
python3 -m venv mvpa_env
source mvpa_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate setup
python3 demo_testing.py
```

#### Sherlock HPC
```bash
# Log into Sherlock
ssh username@login.sherlock.stanford.edu

# Setup environment
cd $SCRATCH
mkdir delay_discounting_analysis && cd delay_discounting_analysis

# Load modules and create environment
module load python/3.9.0 py-pip/21.1.2_py39
python3 -m venv mvpa_env
source mvpa_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Analyses

#### Local/Interactive
```bash
# Interactive analysis (recommended for beginners)
python3 delay_discounting_mvpa_pipeline.py

# Specific analysis types
python3 delay_discounting_mvpa_pipeline.py --behavioral-only
python3 delay_discounting_mvpa_pipeline.py --mvpa-only

# Specific subjects
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 subject_002
```

#### Geometry Analysis (Enhanced!)
```bash
# Standard geometry analysis
python3 delay_discounting_geometry_analysis.py --example

# Advanced trajectory analysis only
python3 delay_discounting_geometry_analysis.py --example --trajectory-only

# Comprehensive analysis (standard + trajectory)
python3 delay_discounting_geometry_analysis.py --example --comprehensive

# Standard with trajectory as bonus
python3 delay_discounting_geometry_analysis.py --example --trajectory
```

#### Sherlock HPC
```bash
# Interactive testing session
sdev -t 2:00:00 -m 16GB -c 4
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 --config config_sherlock.yaml

# Submit batch job
sbatch submit_mvpa_sherlock.sh

# Check job status
squeue -u $USER

# Monitor job progress
tail -f logs/mvpa_JOBID.out
```

### Testing & Validation
```bash
# Run all tests
python3 run_tests.py

# Run specific test categories
python3 run_tests.py --behavioral
python3 run_tests.py --mvpa
python3 run_tests.py --fast

# Generate coverage report
python3 run_tests.py --coverage --html
```

### Results & Visualization
```bash
# Generate result plots
python3 analyze_results.py

# Validate ROI masks
python3 validate_roi_masks.py
```

---

## 🖥️ Sherlock HPC Quick Commands

### SLURM Job Management
```bash
# Submit job
sbatch submit_mvpa_sherlock.sh

# Check job queue
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# Cancel job
scancel JOBID

# Job efficiency (after completion)
seff JOBID

# Job history
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

### Interactive Sessions
```bash
# Request interactive session
sdev -t 2:00:00 -m 16GB -c 4

# Interactive with GPU
sdev -t 1:00:00 -m 8GB -c 2 --gres=gpu:1

# Exit interactive session
exit
```

### Resource Monitoring
```bash
# Check partition status
sinfo

# Check available resources
sinfo -p normal -o "%20N %10c %10m %25f %10a"

# Monitor job progress
tail -f logs/mvpa_JOBID.out

# Check node utilization
sstat -j JOBID --format=MaxRSS,AveCPU
```

### Data Management
```bash
# Copy data to SCRATCH for faster access
cp -r /oak/stanford/groups/russpold/data/subset/ $SCRATCH/

# Archive results to OAK
cp -r results/ /oak/stanford/groups/russpold/users/$USER/mvpa_results_$(date +%Y%m%d)/

# Compress large datasets
tar -czf results_$(date +%Y%m%d).tar.gz results/

# Check storage usage
du -sh $SCRATCH $HOME
df -h /oak/stanford/groups/russpold/
```

### Module Management
```bash
# List available modules
module avail python
module avail py-

# Load modules
module load python/3.9.0 py-pip/21.1.2_py39

# Check loaded modules
module list

# Unload modules
module purge
```

---

## 📁 Key Files & Directories

### Configuration
- `config.yaml` - Main configuration file
- `requirements.txt` - Python dependencies

### Scripts
- `delay_discounting_mvpa_pipeline.py` - Main analysis pipeline
- `delay_discounting_geometry_analysis.py` - Comprehensive geometry analysis (enhanced!)
- `geometry_utils.py` - Consolidated geometry functions library (NEW!)
- `analyze_results.py` - Results visualization
- `run_tests.py` - Test launcher

### Results Structure
```
results/
├── behavioral_analysis/
│   ├── behavioral_summary.csv
│   └── behavioral_parameters.png
├── mvpa_analysis/
│   ├── mvpa_summary.csv
│   └── roi_decoding_results.png
├── geometry_analysis/  (enhanced!)
│   ├── geometry_summary.csv
│   ├── *_geometry.png (standard visualizations)
│   ├── trajectory_analysis_summary.json (NEW!)
│   └── *_advanced_geometry.png (NEW! 6-panel trajectory plots)
└── analysis_report.txt
```

---

## 🎯 Common Workflows

### First-Time Setup

#### Local/Farmshare
1. `python3 -m venv mvpa_env && source mvpa_env/bin/activate`
2. `pip install -r requirements.txt`
3. `python3 demo_testing.py`
4. `python3 run_tests.py --fast`
5. Edit `config.yaml` with your data paths
6. `python3 delay_discounting_mvpa_pipeline.py`

#### Sherlock HPC
1. `ssh username@login.sherlock.stanford.edu`
2. `cd $SCRATCH && mkdir delay_discounting_analysis && cd delay_discounting_analysis`
3. `module load python/3.9.0 py-pip/21.1.2_py39`
4. `python3 -m venv mvpa_env && source mvpa_env/bin/activate`
5. `pip install -r requirements.txt`
6. Create `config_sherlock.yaml` with Sherlock-specific paths
7. `sdev -t 1:00:00 -m 8GB && python3 delay_discounting_mvpa_pipeline.py --subjects subject_001`

### Daily Analysis Workflow

#### Local/Farmshare
1. `source mvpa_env/bin/activate`
2. `python3 delay_discounting_mvpa_pipeline.py --subjects [subject_list]`
3. `python3 analyze_results.py`
4. Review results in `results/` directory

#### Local/Farmshare + Geometry Focus
1. `source mvpa_env/bin/activate`
2. `python3 delay_discounting_mvpa_pipeline.py --subjects [subject_list]` (includes basic geometry)
3. `python3 delay_discounting_geometry_analysis.py --example --comprehensive` (advanced geometry)
4. `python3 analyze_results.py`
5. Review both MVPA and advanced geometry results

#### Sherlock HPC
1. `ssh username@login.sherlock.stanford.edu && cd $SCRATCH/delay_discounting_analysis`
2. `module load python/3.9.0 py-pip/21.1.2_py39 && source mvpa_env/bin/activate`
3. `sbatch submit_mvpa_sherlock.sh`
4. `squeue -u $USER` (monitor jobs)
5. `cp -r results/ /oak/stanford/groups/russpold/users/$USER/` (archive results)

### Troubleshooting Workflow

#### Local/Farmshare
1. `python3 demo_testing.py`
2. `python3 run_tests.py --behavioral`
3. Check `logs/` directory for error files
4. Validate data paths in `config.yaml`
5. `python3 validate_roi_masks.py`

#### Sherlock HPC
1. `squeue -u $USER` (check job status)
2. `cat logs/mvpa_JOBID.err` (check error logs)
3. `scontrol show job JOBID` (detailed job info)
4. `sinfo -p normal` (check partition availability)
5. `seff JOBID` (check resource usage after completion)

---

## 🧪 Analysis Framework Usage

### Creating Analysis Objects
```python
from analysis_base import AnalysisFactory

# Create analysis instances
behavioral = AnalysisFactory.create('behavioral', config=config)
mvpa = AnalysisFactory.create('mvpa', config=config)
geometry = AnalysisFactory.create('geometry', config=config)
```

### Running Analyses
```python
# Run analysis on subjects
results = behavioral.run_analysis(['subject_001', 'subject_002'])

# Process single subject
result = behavioral.process_subject('subject_001')

# Save/load results
behavioral.save_results('behavioral_results.pkl')
loaded = behavioral.load_results('behavioral_results.pkl')
```

### Geometry Utilities (NEW!)
```python
from geometry_utils import (
    compute_manifold_alignment,
    compute_information_geometry_metrics,
    compute_geodesic_distances,
    analyze_trajectory_dynamics
)

# Manifold alignment between conditions
alignment = compute_manifold_alignment(X1, X2, method='procrustes')

# Information geometry metrics
info_metrics = compute_information_geometry_metrics(X1, X2)

# Geodesic distances for non-linear manifold analysis
geodesic_dist = compute_geodesic_distances(neural_data, k=5)

# Trajectory dynamics across conditions
trajectory_results = analyze_trajectory_dynamics(embeddings, conditions)
```

---

## 📊 Result Interpretation

### Behavioral Analysis
| Metric | Good | Poor | Meaning |
|--------|------|------|---------|
| k-value | 0.01-0.05 | >0.1 or <0.005 | Discount rate |
| R² | >0.3 | <0.2 | Model fit quality |
| Choice rate | 0.3-0.7 | <0.1 or >0.9 | Response variability |

### MVPA Analysis
| Metric | Strong | Weak | Meaning |
|--------|--------|------|---------|
| Accuracy | >60% | <55% | Decoding performance |
| p-value | <0.001 | >0.05 | Statistical significance |
| N trials | >100 | <50 | Data sufficiency |

---

## 🔧 Configuration Examples

### Basic config.yaml
```yaml
data_paths:
  fmri_data_dir: "/path/to/fmri_data"
  behavioral_data_dir: "/path/to/behavioral_data"

analysis_params:
  cv_folds: 5
  n_permutations: 1000
  tr: 2.0

roi_masks:
  striatum: "./masks/striatum_mask.nii.gz"
  dlpfc: "./masks/dlpfc_mask.nii.gz"
  vmpfc: "./masks/vmpfc_mask.nii.gz"
```

### Sherlock config_sherlock.yaml
```yaml
data_paths:
  fmri_data_dir: "/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/fmriprep"
  behavioral_data_dir: "/oak/stanford/groups/russpold/data/uh2/aim1/behavioral_data/event_files"

cluster_settings:
  use_slurm: true
  max_parallel_jobs: 8
  memory_per_job: "32GB"
  partition: "normal"

output:
  results_dir: "$SCRATCH/delay_discounting_analysis/results"
```

### Environment Variables
```bash
# Use custom configuration
export MVPA_CONFIG_ENV=development

# Custom config file (local)
export MVPA_CONFIG_FILE=/path/to/custom_config.yaml

# Sherlock-specific config
export MVPA_CONFIG_FILE=config_sherlock.yaml
```

---

## 🐛 Quick Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Data Loading Errors
1. Check file paths in `config.yaml`
2. Verify file permissions
3. Test with single subject

### Memory Errors
```bash
# Use memory-efficient mode
python3 delay_discounting_mvpa_pipeline.py --memory-efficient

# Process fewer subjects
python3 delay_discounting_mvpa_pipeline.py --subjects subject_001
```

### Test Failures
```bash
# Run specific tests
python3 run_tests.py --verbose --tb=long

# Check test collection
pytest test_pipeline_pytest.py --collect-only
```

### Sherlock-Specific Issues
```bash
# Job pending/not starting
squeue -u $USER  # Check job status
sinfo -p normal  # Check partition availability

# Out of memory errors
seff JOBID  # Check actual memory usage
# Increase memory: #SBATCH --mem=64GB

# Job timeout
scontrol show job JOBID | grep TimeLimit
# Increase time: #SBATCH --time=12:00:00

# Module loading issues
module avail python  # Check available modules
module list  # Check loaded modules
module purge && module load python/3.9.0  # Reset modules

# Data access problems
ls /oak/stanford/groups/russpold/  # Check OAK access
df -h /oak/stanford/groups/russpold/  # Check OAK space
```

---

## 📈 Performance Tips

### Speed Up Analysis
- Use `--fast` option for testing
- Process subjects in batches
- Use parallel processing when available
- **Sherlock**: Use SCRATCH storage for faster I/O
- **Sherlock**: Use array jobs for multiple subjects

### Memory Optimization
- Enable memory-efficient mode
- Process fewer subjects simultaneously
- Clear results between runs if needed
- **Sherlock**: Request appropriate memory (don't over-allocate)
- **Sherlock**: Use bigmem partition for memory-intensive analyses

### Sherlock Resource Optimization
- **Interactive**: Use `sdev` for testing only (max 2 hours)
- **Batch jobs**: Submit production analyses via SLURM
- **Storage**: Copy frequently-used data to SCRATCH
- **Monitoring**: Use `seff JOBID` to check resource efficiency
- **Partitions**: Choose appropriate partition (normal/bigmem/gpu)

### Quality Control
- Always validate masks first
- Check behavioral data quality
- Review log files for warnings
- **Sherlock**: Monitor job efficiency with `seff`
- **Sherlock**: Check error logs in `logs/mvpa_JOBID.err`

---

## 📚 Documentation Quick Links

- **New Users**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Setup Verification**: [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)
- **Testing Guide**: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Analysis Framework**: [ANALYSIS_CLASSES_README.md](ANALYSIS_CLASSES_README.md)
- **Full Documentation**: [README.md](README.md)

---

## 🆘 Getting Help

### Self-Help Steps
1. Check this quick reference
2. Run diagnostic commands
3. Review error logs
4. Check documentation

### Contact Information
- **Documentation**: Check README.md and guides
- **Issues**: Review troubleshooting sections
- **Support**: Cognitive Neuroscience Lab, Stanford University

---

**💡 Pro Tip**: Bookmark this page for quick access to common commands and workflows! 