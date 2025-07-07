# Mass Univariate Analysis Module

This module provides comprehensive whole-brain GLM analysis for delay discounting fMRI data, including individual subject-level modeling, group-level random effects analysis, and multiple comparisons correction.

## Features

### **Preprocessing**
- **Spatial smoothing**: 4mm FWHM Gaussian kernel
- **Confound regression**: Motion, physiological, and AROMA components from fMRIPrep
- **Temporal filtering**: High-pass filtering and drift removal

### **GLM Models**
The pipeline implements 5 different GLM models:

1. **Model 1 - Choice**: Binary regressor contrasting sooner-smaller vs larger-later choices
2. **Model 2 - SV Chosen**: Parametric regressor for subjective value of chosen option  
3. **Model 3 - SV Unchosen**: Parametric regressor for subjective value of unchosen option
4. **Model 4 - SV Both**: Both chosen and unchosen SV with direct contrast
5. **Model 5 - Choice Difficulty**: Choice difficulty based on SV difference

### **Statistical Analysis**
- **First-level**: Subject-wise GLM with trial onset + 4000ms duration
- **Second-level**: Random effects group analysis
- **Multiple comparisons**: FDR and cluster-based correction (α = 0.05)

## Quick Start

### **1. Test the Module**
```bash
# Run comprehensive tests
python test_mass_univariate.py
```

### **2. Check Available Data**
```bash
# Check which subjects have complete data
python run_mass_univariate.py --check-only
```

### **3. Run Test Analysis**
```bash
# Test with first 3 subjects
python run_mass_univariate.py --test
```

### **4. Run Full Analysis**
```bash
# All available subjects
python run_mass_univariate.py

# Specific subjects
python run_mass_univariate.py --subjects sub-s001,sub-s002,sub-s003
```

### **5. Cluster Submission**
```bash
# Create SLURM script for HPC
python run_mass_univariate.py --create-slurm
sbatch /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/submit_mass_univariate.sh
```

## Prerequisites

### **Data Requirements**
- **fMRI data**: fMRIPrep-preprocessed files in MNI space
- **Behavioral data**: Trial-wise event files with timing and choices
- **Directory structure**: Organized according to your Oak storage configuration

### **Expected File Structure**
```
/oak/stanford/groups/russpold/data/uh2/aim1/
├── derivatives/fmriprep/
│   └── {subject}/ses-2/func/
│       ├── {subject}_ses-2_task-discountFix_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
│       └── {subject}_ses-2_task-discountFix_*_desc-confounds_timeseries.tsv
└── behavioral_data/event_files/
    └── {subject}_discountFix_events.tsv
```

### **Behavioral Data Columns**
Your behavioral files should include:
- `onset`: Trial onset time (seconds)
- `response`: Choice response (1=SS, 2=LL)  
- `delay_days`: Delay to larger reward
- `large_amount`: Amount of larger reward
- Additional columns computed automatically: `sv_chosen`, `sv_unchosen`, `choice`, `sv_difference`

## Output Structure

### **Directory Organization**
```
{OUTPUT_DIR}/mass_univariate/
├── smoothed_data/           # 4mm smoothed fMRI data
├── first_level/            # Subject-level results
│   └── {subject}_{model}_{contrast}_zstat.nii.gz
├── second_level/           # Group-level results
│   ├── {model}_{contrast}_group_zstat.nii.gz          # Unthresholded
│   ├── {model}_{contrast}_group_zstat_fdr0.05.nii.gz   # FDR corrected
│   ├── {model}_{contrast}_group_zstat_cluster0.05.nii.gz # Cluster corrected
│   ├── complete_mass_univariate_results.pkl           # Complete results
│   └── mass_univariate_summary_report.txt             # Summary report
```

### **Statistical Maps Generated**

| Model | Contrasts | Description |
|-------|-----------|-------------|
| model_1_choice | choice_effect | SS vs LL choice activation |
| model_2_sv_chosen | sv_chosen_effect | Chosen option value coding |
| model_3_sv_unchosen | sv_unchosen_effect | Unchosen option value coding |
| model_4_sv_both | sv_chosen_effect<br/>sv_unchosen_effect<br/>sv_chosen_vs_unchosen | Individual SV effects<br/>Direct contrast |
| model_5_difficulty | difficulty_effect<br/>choice_effect | Choice difficulty<br/>Choice main effect |

## Technical Details

### **GLM Parameters**
- **TR**: 0.68 seconds (from acquisition parameters)
- **HRF model**: SPM canonical HRF
- **Drift model**: Cosine basis functions
- **High-pass filter**: 128 seconds (0.0078 Hz)
- **Trial duration**: 4000ms from onset

### **Confounds Applied**
- **Motion**: 6 rigid-body parameters + derivatives
- **Physiological**: CSF, white matter, global signal
- **Artifacts**: AROMA components (if available)

### **Multiple Comparisons Correction**
- **FDR**: False Discovery Rate at q < 0.05
- **Cluster**: Cluster-extent based with p < 0.05
- **Permutation**: Available for future implementation

## Configuration

### **Customizing Analysis**
Modify `mass_univariate_config.json` to customize:
- Smoothing kernel size
- GLM parameters
- Confound selection
- Multiple comparisons methods
- Quality control thresholds

### **Key Parameters**
```json
{
  "preprocessing": {
    "smoothing_fwhm": 4.0
  },
  "glm_parameters": {
    "tr": 0.68,
    "trial_duration": 4.0
  },
  "multiple_comparisons": {
    "alpha": 0.05,
    "methods": ["fdr", "cluster"]
  }
}
```

## Quality Control

### **Automatic Checks**
- Minimum behavioral accuracy (60%)
- Maximum reaction time (10 seconds)  
- Motion threshold monitoring
- File existence verification

### **Exclusion Criteria**
- Missing fMRI or behavioral data
- Insufficient valid trials
- Excessive motion artifacts
- Behavioral model fitting failures

## Troubleshooting

### **Common Issues**

**No subjects found:**
```bash
# Check data paths in oak_storage_config.py
python run_mass_univariate.py --check-only
```

**GLM fitting errors:**
- Check behavioral data format
- Verify trial timing alignment
- Ensure confounds file exists

**Memory issues:**
- Reduce number of parallel jobs
- Use cluster submission script
- Check available disk space on Oak

### **Log Files**
- SLURM output: `{OUTPUT_DIR}/mass_univariate/slurm_*.out`
- Error logs: `{OUTPUT_DIR}/mass_univariate/slurm_*.err`
- Analysis logs: Console output during execution

## Integration with Existing Pipeline

This module integrates seamlessly with your existing delay discounting analysis pipeline:

1. **Behavioral modeling**: Uses same hyperbolic discounting functions
2. **Data paths**: Leverages Oak storage configuration
3. **Quality control**: Applies consistent exclusion criteria
4. **Output format**: Compatible with visualization tools

## Performance Notes

### **Runtime Estimates**
- **Per subject**: ~5-10 minutes (first-level modeling)
- **Group analysis**: ~2-5 minutes per contrast
- **Total time**: ~1-3 hours for 20-50 subjects

### **Resource Requirements**
- **Memory**: 8-16 GB recommended  
- **Storage**: ~500 MB per subject (smoothed data)
- **CPUs**: Parallel processing supported

## Next Steps

After running the analysis:

1. **Visualization**: Use nilearn plotting functions for results visualization
2. **ROI analysis**: Extract activation from specific regions
3. **Connectivity**: Use first-level results for connectivity analysis
4. **Meta-analysis**: Combine with other delay discounting studies

## Support

For issues or questions:
1. Run the test suite: `python test_mass_univariate.py`
2. Check log files for specific error messages
3. Verify data paths and file formats
4. Consult nilearn documentation for GLM details 