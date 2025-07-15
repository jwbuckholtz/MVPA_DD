# MVPA Pipeline Setup Checklist

Use this checklist to ensure you've completed all setup steps correctly.

## ✅ Environment Setup

### Basic Requirements (Local/Farmshare)
- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Virtual environment created (`python3 -m venv mvpa_env`)
- [ ] Virtual environment activated (`source mvpa_env/bin/activate`)
- [ ] Prompt shows `(mvpa_env)` when activated

### Sherlock HPC Setup
- [ ] Can log into Sherlock (`ssh username@login.sherlock.stanford.edu`)
- [ ] Working in SCRATCH directory (`cd $SCRATCH`)
- [ ] Required modules loaded (`module load python/3.9.0 py-pip/21.1.2_py39`)
- [ ] Virtual environment created in SCRATCH
- [ ] Can access OAK storage (`ls /oak/stanford/groups/russpold/`)

### Directory Structure
- [ ] Created project directory (`mkdir delay_discounting_analysis`)
- [ ] Pipeline files in project directory
- [ ] In correct working directory (`pwd` shows project path)
- [ ] **Sherlock**: Using SCRATCH for analysis directory

## ✅ Dependencies Installation

### Package Installation
- [ ] Requirements file exists (`ls requirements.txt`)
- [ ] Pip upgraded (`pip install --upgrade pip`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Core packages work (`python3 -c "import numpy, pandas, sklearn, nibabel, nilearn"`)

### Testing Framework
- [ ] Pytest available (`python3 -c "import pytest"`)
- [ ] Testing system demo works (`python3 demo_testing.py`)
- [ ] Fast tests pass (`python3 run_tests.py --fast --no-cov`)

## ✅ Data Setup

### Data Structure Understanding
- [ ] Understand expected data structure (see GETTING_STARTED.md Step 4)
- [ ] Know location of your fMRI data
- [ ] Know location of your behavioral data
- [ ] ROI masks available in `masks/` directory

### Data Validation
- [ ] ROI masks validated (`python3 validate_roi_masks.py`)
- [ ] Can load behavioral data for test subject
- [ ] File permissions are correct
- [ ] Data paths are accessible

## ✅ Configuration

### Basic Configuration
- [ ] Configuration file exists (`ls config.yaml`)
- [ ] Data paths updated in config.yaml
- [ ] ROI mask paths correct
- [ ] Analysis parameters appropriate for your data

### Sherlock Configuration
- [ ] Created Sherlock-specific config (`config_sherlock.yaml`)
- [ ] OAK data paths correct in Sherlock config
- [ ] SCRATCH output paths configured
- [ ] SLURM settings appropriate for dataset size
- [ ] Can load Sherlock config (`export MVPA_CONFIG_FILE=config_sherlock.yaml`)

### Advanced Configuration (Optional)
- [ ] Environment-specific config created if needed
- [ ] Configuration validation passes
- [ ] Environment variables set if using custom config

## ✅ First Analysis Run

### Test Analysis (Local/Farmshare)
- [ ] Interactive pipeline starts (`python3 delay_discounting_mvpa_pipeline.py`)
- [ ] Can process one test subject successfully
- [ ] No import errors or missing dependencies
- [ ] Results directory created

### Test Analysis (Sherlock)
- [ ] Can request interactive session (`sdev -t 1:00:00 -m 8GB -c 2`)
- [ ] Test analysis runs (`python3 delay_discounting_mvpa_pipeline.py --subjects subject_001 --config config_sherlock.yaml`)
- [ ] Can submit batch job (`sbatch submit_mvpa_sherlock.sh`)
- [ ] Job appears in queue (`squeue -u $USER`)
- [ ] Can monitor job progress (`tail -f logs/mvpa_JOBID.out`)

### Results Validation
- [ ] Results files generated in expected locations
- [ ] Can open and view result files
- [ ] Summary statistics look reasonable
- [ ] No error messages in log files
- [ ] **Sherlock**: Results copied to OAK for permanent storage

## ✅ Understanding Outputs

### Result Files
- [ ] Behavioral summary CSV exists and is readable
- [ ] MVPA summary CSV exists and is readable
- [ ] Analysis report TXT file exists and is readable
- [ ] Plots generated successfully

### Result Interpretation
- [ ] Understand behavioral metrics (k-values, R²)
- [ ] Understand MVPA metrics (accuracy, p-values)
- [ ] Can identify significant vs. non-significant results
- [ ] Know what constitutes good vs. poor results

## ✅ Advanced Features (Optional)

### Testing System
- [ ] Can run different test categories
- [ ] Coverage reports generate correctly
- [ ] Understand test output and failures

### Analysis Framework
- [ ] Can create analysis objects using factory
- [ ] Can run analyses programmatically
- [ ] Can save and load results

### Visualization
- [ ] Standard plots generate correctly
- [ ] Can create custom visualizations
- [ ] Plots are publication-quality

## 🚨 Troubleshooting Checklist

If you encounter issues, check these common problems:

### Import/Installation Issues
- [ ] Virtual environment is activated
- [ ] All packages in requirements.txt installed
- [ ] Python version is 3.8+
- [ ] No conflicting package versions

### Data Issues
- [ ] File paths in config are correct and absolute
- [ ] Files exist at specified locations
- [ ] File permissions allow reading
- [ ] Data format matches expected structure

### Memory/Performance Issues
- [ ] Sufficient RAM available (8GB minimum)
- [ ] Sufficient disk space (10GB minimum)
- [ ] Not running too many subjects simultaneously
- [ ] Consider using memory-efficient options

### Analysis Issues
- [ ] Input data is valid and complete
- [ ] Configuration parameters are appropriate
- [ ] No corrupted data files
- [ ] Log files checked for specific errors

### Sherlock-Specific Issues
- [ ] Jobs not starting: Check partition availability (`sinfo -p normal`)
- [ ] Out of memory: Check actual usage (`seff JOBID`) and increase allocation
- [ ] Job timeout: Check time limits (`scontrol show job JOBID`) and increase if needed
- [ ] Module issues: Check available modules (`module avail python`)
- [ ] Data access: Verify OAK permissions (`ls /oak/stanford/groups/russpold/`)
- [ ] Storage space: Check SCRATCH usage (`du -sh $SCRATCH`)

## ✅ Ready for Production

### Final Verification
- [ ] Successfully analyzed at least 2 test subjects
- [ ] Results look reasonable and interpretable
- [ ] No persistent error messages
- [ ] Understand how to run full dataset

### Documentation
- [ ] Read relevant documentation (GETTING_STARTED.md)
- [ ] Understand analysis framework (ANALYSIS_CLASSES_README.md)
- [ ] Know how to run tests (TESTING_GUIDE.md)
- [ ] Familiar with troubleshooting steps

### Next Steps Planned
- [ ] Have plan for full dataset analysis
- [ ] Know how to customize for your needs
- [ ] Have backup/save strategy for results
- [ ] Know where to get help if needed

---

## 📞 Getting Help

If any checklist items fail:

1. **Check log files**: `ls logs/` and `cat logs/latest_analysis.log`
2. **Run diagnostic tests**: `python3 run_tests.py --behavioral`
3. **Validate setup**: `python3 demo_testing.py`
4. **Review documentation**: `GETTING_STARTED.md` section 9 (Troubleshooting)
5. **Contact support**: Cognitive Neuroscience Lab, Stanford University

---

**🎯 Success Criteria:**
- All ✅ boxes checked
- No persistent error messages
- Successfully analyzed test subjects
- Results files generated and interpretable
- Ready to scale to full dataset

**🚀 You're ready to go!** Proceed with confidence to analyze your full dataset. 