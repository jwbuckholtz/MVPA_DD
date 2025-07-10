# Delay Discounting MVPA Analysis Pipeline

A comprehensive pipeline for analyzing delay discounting fMRI data, including behavioral modeling, MVPA decoding, and neural geometry analysis.

## Overview

This pipeline is designed for cognitive neuroscience researchers studying delay discounting using fMRI. It provides:

1. **Behavioral Analysis**: Hyperbolic discounting parameter estimation
2. **MVPA Decoding**: Classification and regression on neural data  
3. **Neural Geometry**: Low-dimensional embedding analysis
4. **Data Utilities**: Centralized data loading and validation system
5. **MVPA Utilities**: Centralized machine learning procedures (NEW!)
6. **Visualization**: Comprehensive results plotting and reporting

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

### Utility Modules

- `data_utils.py` - Centralized data loading, validation, and integrity checking
- `mvpa_utils.py` - Centralized MVPA procedures and machine learning operations
- `logger_utils.py` - Standardized logging, argument parsing, and import management (NEW!)
- `demo_data_utils.py` - Demonstration script for data utilities
- `demo_mvpa_utils.py` - Demonstration script for MVPA utilities
- `demo_logger_utils.py` - Demonstration script for logger utilities (NEW!)

### Configuration

- `requirements.txt` - Python package dependencies
- `Dataset_Descriptor_Files/` - Data descriptors and quality control

### Documentation

- `DELAY_DISCOUNTING_GEOMETRY_README.md` - Comprehensive geometry analysis documentation
- `DATA_UTILS_README.md` - Complete data utilities documentation and usage guide
- `MVPA_UTILS_README.md` - Complete MVPA utilities documentation and usage guide
- `LOGGER_UTILS_README.md` - Complete logger utilities documentation and usage guide (NEW!)
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
# Edit email address in submit script
nano submit_analysis_job.sh

# Submit job
sbatch submit_analysis_job.sh
```

### 4. Analyze Results

```bash
python analyze_results.py
```

### 5. Data Utilities and Quality Control (NEW!)

The pipeline now includes a centralized data utilities module (`data_utils.py`) that provides:

#### Check Data Integrity
```bash
# Check data availability and quality for all subjects
python analyze_results.py --check_data

# Demonstrate all data utilities features
python demo_data_utils.py

# Quick demo (faster)
python demo_data_utils.py --quick
```

#### Key Features
- **Centralized Data Loading**: Single functions for behavioral, fMRI, and confounds data
- **Automatic Validation**: Quality control for all data types
- **Subject Discovery**: Automated finding of subjects with complete data
- **ROI Operations**: Streamlined mask loading and time series extraction
- **Error Handling**: Robust error handling with custom exceptions

#### Usage in Scripts
```python
from data_utils import (
    load_behavioral_data, load_fmri_data, get_complete_subjects,
    check_data_integrity
)

# Find subjects with complete data
subjects = get_complete_subjects()

# Load data with validation and preprocessing
behavioral_data = load_behavioral_data(subject_id, config, validate=True)
fmri_img = load_fmri_data(subject_id, config)

# Check data integrity
integrity_report = check_data_integrity(subjects)
```

See `DATA_UTILS_README.md` for comprehensive documentation.

### 6. Logger Utilities and Standardization (NEW!)

The pipeline now includes standardized logging, argument parsing, and import management (`logger_utils.py`) that provides:

#### Core Features
- **Standardized Logging**: Consistent logging across all scripts with configurable levels and formats
- **Common Argument Parsing**: Reusable argument patterns for different script types
- **Import Management**: Centralized import handling with optional dependency fallback
- **Environment Setup**: Complete environment initialization for pipeline scripts
- **Progress Tracking**: Built-in progress logging for long-running operations
- **Error Handling**: Comprehensive error logging with tracebacks

#### Usage Examples
```python
# Basic script setup
from logger_utils import setup_script_logging, create_analysis_parser

# Parse arguments
parser = create_analysis_parser('my_analysis', 'mvpa')
args = parser.parse_args()

# Setup logging
logger = setup_script_logging('my_analysis', verbose=args.verbose)

# Advanced complete setup
from logger_utils import setup_pipeline_environment

env = setup_pipeline_environment('my_script', args, ['numpy', 'pandas'])
logger = env['logger']
config = env['config']
```

#### Key Benefits
- **60-90% reduction** in setup code across scripts
- **Consistent logging** and error handling
- **Standardized argument parsing** patterns
- **Automatic environment** validation and setup
- **Progress tracking** for long-running operations

#### Demo and Testing
```bash
# Demonstrate all logger utilities features
python demo_logger_utils.py

# Test with different demo types
python demo_logger_utils.py --demo-type basic
python demo_logger_utils.py --demo-type advanced
```

See `LOGGER_UTILS_README.md` for comprehensive documentation.

### 7. Neural Geometry Analysis (Optional)

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