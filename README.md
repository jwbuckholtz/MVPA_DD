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

#### **Standard Analyses**
- **Dimensionality Reduction**: PCA, MDS, t-SNE
- **Representational Similarity Analysis (RSA)**: Neural pattern similarity matrices
- **Distance Analysis**: Within vs between condition separability with permutation testing

#### **Advanced Geometric Analyses**
- **Manifold Alignment**: Procrustes analysis and Canonical Correlation Analysis (CCA)
- **Geodesic Distance Analysis**: Non-linear manifold distances using Isomap
- **Manifold Curvature**: Local geometric complexity estimation
- **Information Geometry**: KL divergence, Jensen-Shannon divergence, Wasserstein distance
- **Specialized Delay Discounting Comparisons**: Choice types, delay lengths, subjective values

## Files

### Core Scripts

- `delay_discounting_mvpa_pipeline.py` - Main analysis pipeline
- `create_roi_masks.py` - Generate anatomical ROI masks
- `analyze_results.py` - Results visualization and statistics
- `submit_analysis_job.sh` - SLURM job submission script

### Geometry Analysis Scripts

- `delay_discounting_geometry_analysis.py` - Specialized delay discounting geometry analysis
- `geometric_transformation_analysis.py` - Advanced geometric transformation methods
- `dd_geometry_config.json` - Configuration for geometry analyses

### Configuration

- `requirements.txt` - Python package dependencies
- `Dataset_Descriptor_Files/` - Data descriptors and quality control

### Documentation

- `DELAY_DISCOUNTING_GEOMETRY_README.md` - Comprehensive geometry analysis documentation
- `Prompts/` - Development prompts and specifications

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
# Mass Univariate Analysis (NEW)
sbatch submit_mass_univariate.sh                    # All subjects
sbatch --export=MODE=test submit_mass_univariate.sh # Test run

# Original MVPA Analysis
nano submit_analysis_job.sh  # Edit email address
sbatch submit_analysis_job.sh
```

### 4. Analyze Results

```bash
python analyze_results.py
```

### 5. Mass Univariate Analysis (NEW)

#### Run Whole-Brain GLM Analysis
```bash
# Test the module first
python test_mass_univariate.py

# Check available subjects
python run_mass_univariate.py --check-only

# Run test analysis (3 subjects)
python run_mass_univariate.py --test

# Run full analysis
python run_mass_univariate.py

# SLURM cluster submission
sbatch submit_mass_univariate.sh
```

This module performs:
- **Spatial smoothing** (4mm FWHM)
- **5 GLM models**: Choice, SV chosen, SV unchosen, SV both, Choice difficulty
- **Random effects** group analysis
- **Multiple comparisons correction** (FDR + cluster)

See `MASS_UNIVARIATE_README.md` and `SLURM_USAGE_GUIDE.md` for detailed documentation.

### 6. Neural Geometry Analysis (Optional)

#### Run Specialized Delay Discounting Geometry Analysis
```bash
# For extracted ROI data
python delay_discounting_geometry_analysis.py \
    --neural-data roi_timeseries.npy \
    --behavioral-data trial_data.csv \
    --roi-name "DLPFC" \
    --config dd_geometry_config.json

# Test with example data
python delay_discounting_geometry_analysis.py --example
```

#### Available Geometry Comparisons
- `choice` - Sooner-smaller vs larger-later choices
- `delay_short_vs_long` - Short vs long delays  
- `delay_immediate_vs_delayed` - Immediate vs delayed trials
- `sv_chosen_median` - High vs low chosen option values
- `sv_unchosen_median` - High vs low unchosen option values
- `sv_difference_median` - High vs low subjective value differences
- `value_diff_terciles` - Similar vs different option values (choice difficulty)

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

dd_geometry_results/            # Specialized geometry analysis outputs
├── {ROI}_{comparison}_results.json      # Detailed geometry results
├── {ROI}_summary_report.txt            # Text summary
└── visualizations/                     # Advanced geometry plots
    └── {ROI}_{comparison}_advanced_geometry.png

masks/
├── striatum_mask.nii.gz        # Anatomical ROI masks
├── dlpfc_mask.nii.gz
├── vmpfc_mask.nii.gz
└── roi_masks_visualization.png # Mask visualization
```

## Key Features

### Advanced Neural Geometry Analysis
- **Specialized Delay Discounting Comparisons**: Choice types, delay categories, subjective value splits
- **Manifold Alignment Analysis**: Procrustes and CCA methods for comparing neural manifolds
- **Information-Theoretic Measures**: KL/JS divergence and Wasserstein distance between conditions  
- **Geodesic Distance Analysis**: Non-linear manifold distances using Isomap
- **Curvature Estimation**: Local geometric complexity of neural representations
- **Comprehensive Visualizations**: 6-panel advanced geometry plots for each comparison
- **Statistical Rigor**: Permutation testing for geometric separation measures

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
- **Advanced geometry visualizations**: Manifold alignment, information geometry, curvature analysis
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

### Customizing Geometry Analysis

1. **Add new comparison types**: Modify `DelayDiscountingGeometryAnalyzer` class
2. **Adjust advanced methods**: Configure parameters in `dd_geometry_config.json`
3. **Custom visualizations**: Extend `visualize_advanced_geometry_results()` method
4. **Performance tuning**: Adjust neighbor counts and enable/disable specific analyses

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
- **ML**: scikit-learn (≥0.24.0), statsmodels
- **Neuroimaging**: nibabel, nilearn
- **Utilities**: joblib, tqdm, pathlib

### Advanced Geometry Requirements
- **scikit-learn ≥0.24.0**: For Isomap and CCA analysis
- **scipy ≥1.7.0**: For advanced statistical functions (KDE, information theory)
- **matplotlib ≥3.3.0**: For 3D visualizations and advanced plotting

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

## Additional Notes

### Monitoring Job Progress

To monitor the progress of your job, you can use the `squeue` command to check the job queue and `tail` to follow the output:

```bash
squeue -u $USER
tail -f /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/slurm_*.out
```

This will help you track the progress of your analysis and ensure that it completes successfully. 

## New Output Files

### Second-Level Analysis Results
```bash
ls /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results/mass_univariate/second_level/
```

This directory contains results from the second-level analysis, which includes the results of the GLM models for each subject. 