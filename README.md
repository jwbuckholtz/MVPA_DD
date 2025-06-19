# Delay Discounting MVPA Analysis Pipeline

A comprehensive pipeline for analyzing delay discounting fMRI data, including behavioral modeling, MVPA decoding, and neural geometry analysis.

## Overview

This pipeline is designed for cognitive neuroscience researchers studying delay discounting using fMRI. It provides:

1. **Behavioral Analysis**: Hyperbolic discounting parameter estimation
2. **MVPA Decoding**: Classification and regression on neural data  
3. **Neural Geometry**: Low-dimensional embedding analysis
4. **Visualization**: Comprehensive results plotting and reporting

## Dataset

This analysis is designed for the dataset described in [Eisenberg et al. (2024), Nature Scientific Data](https://www.nature.com/articles/s41597-024-03636-y).

### Data Structure Expected

- **fMRI Data**: `/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/fmriprep/{worker_id}/ses-2/func/`
- **Behavioral Data**: `/oak/stanford/groups/russpold/data/uh2/aim1/behavioral_data/event_files/{worker_id}_discountFix_events.tsv`

## Analysis Components

### 1. Behavioral Modeling

The pipeline fits hyperbolic discounting models to choice data:

- **Hyperbolic Function**: V = A / (1 + k × delay)
- **Parameters Estimated**: Discount rate (k), model fit (pseudo-R²)
- **Trial-wise Variables**: 
  - Subjective value of larger-later option
  - Subjective value of smaller-sooner option  
  - Difference in subjective values
  - Sum of subjective values
  - Chosen option subjective value
  - Unchosen option subjective value

### 2. MVPA Decoding

**Regions of Interest**:
- Striatum (caudate, putamen, nucleus accumbens)
- DLPFC (dorsolateral prefrontal cortex)
- VMPFC (ventromedial prefrontal cortex)
- Whole brain

**Decoding Targets**:
- Choice classification (smaller-sooner vs. larger-later)
- Continuous variable regression:
  - Subjective value difference
  - Subjective value sum  
  - Delay length
  - Chosen option value

**Methods**:
- Linear SVM for classification
- Ridge regression for continuous variables
- Cross-validation with permutation testing
- Multiple comparisons correction (FDR)

### 3. Neural Geometry Analysis

- **Dimensionality Reduction**: PCA, MDS, t-SNE, Isomap
- **Behavioral Correlations**: Map trial-wise behavior to neural geometry
- **Representational Analysis**: Compute representational dissimilarity matrices

## Files

### Core Scripts

- `delay_discounting_mvpa_pipeline.py` - Main analysis pipeline
- `create_roi_masks.py` - Generate anatomical ROI masks
- `analyze_results.py` - Results visualization and statistics
- `submit_analysis_job.sh` - SLURM job submission script

### Configuration

- `requirements.txt` - Python package dependencies
- `Dataset_Descriptor_Files/` - Data descriptors and quality control

## Usage

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create ROI Masks

```bash
python create_roi_masks.py
```

This creates anatomical masks in `./masks/`:
- `striatum_mask.nii.gz`
- `dlpfc_mask.nii.gz` 
- `vmpfc_mask.nii.gz`

### 3. Run Analysis

#### Interactive Mode
```bash
python delay_discounting_mvpa_pipeline.py
```

#### HPC Cluster (SLURM)
```bash
# Edit email address in submit script
nano submit_analysis_job.sh

# Submit job
sbatch submit_analysis_job.sh
```

### 4. Analyze Results

```bash
python analyze_results.py
```

## Output Structure

```
delay_discounting_results/
├── all_results.pkl              # Complete results pickle
├── behavioral_analysis/         # Individual behavioral files
├── mvpa_analysis/              # MVPA results by subject/ROI
└── geometry_analysis/          # Neural geometry results

analysis_outputs/
├── behavioral_distributions.png # Behavioral parameter plots
├── group_mvpa_statistics.csv   # Group-level MVPA stats
└── summary_report.txt          # Comprehensive text report

masks/
├── striatum_mask.nii.gz        # Anatomical ROI masks
├── dlpfc_mask.nii.gz
├── vmpfc_mask.nii.gz
└── roi_masks_visualization.png # Mask visualization
```

## Key Features

### Quality Control
- Behavioral data validation (minimum trials, choice variability)
- fMRI data availability checking
- Suggested exclusions from dataset descriptors

### Statistical Rigor
- Cross-validation for all decoding analyses
- Permutation testing for significance
- Multiple comparisons correction
- Group-level statistical summaries

### Computational Efficiency
- Parallel processing support
- Memory-efficient data handling
- Modular design for easy customization

### Visualization
- Behavioral parameter distributions
- MVPA decoding accuracy plots
- Neural geometry correlation heatmaps
- Comprehensive statistical reporting

## Customization

### Adding New ROIs

1. Create mask file in `./masks/`
2. Add to `Config.ROI_MASKS` dictionary in pipeline script
3. Re-run analysis

### Adding New Behavioral Variables

1. Modify `BehavioralAnalysis.process_subject_behavior()` 
2. Add variables to MVPA and geometry analysis sections
3. Update visualization scripts

### Changing Analysis Parameters

Edit the `Config` class in `delay_discounting_mvpa_pipeline.py`:

```python
class Config:
    TR = 1.0                    # Repetition time
    HEMI_LAG = 4               # Hemodynamic lag (TRs)
    CV_FOLDS = 5               # Cross-validation folds
    N_PERMUTATIONS = 1000      # Permutation test iterations
```

## Dependencies

- **Python**: ≥3.8
- **Core**: numpy, scipy, pandas, matplotlib, seaborn
- **ML**: scikit-learn, statsmodels
- **Neuroimaging**: nibabel, nilearn
- **Utilities**: joblib, tqdm, pathlib

## Citation

If you use this pipeline, please cite:

1. The original dataset: Eisenberg et al. (2024). "A large-scale examination of self-regulation and mental health." *Nature Scientific Data*, 11, 741.

2. Key neuroimaging tools:
   - Nilearn: Abraham et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*, 8, 14.
   - Scikit-learn: Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.

## Support

For questions or issues:
1. Check the error logs in `logs/` directory
2. Verify data paths and permissions
3. Ensure all dependencies are installed
4. Contact: Cognitive Neuroscience Lab, Stanford University

## License

This pipeline is provided for academic research use. Please respect the terms of use for the underlying dataset and software dependencies. 