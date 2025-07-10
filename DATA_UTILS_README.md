# Data Utilities Module for Delay Discounting MVPA Pipeline

The `data_utils.py` module provides centralized, optimized functions for loading, validating, and managing data across the entire delay discounting analysis pipeline. This module eliminates code duplication and ensures consistent data handling throughout all analysis scripts.

## Overview

### What This Module Provides

- **Centralized Data Loading**: Single functions for loading behavioral data, fMRI data, and confounds
- **Data Validation**: Comprehensive quality control and integrity checking  
- **Subject Management**: Automated discovery and validation of available subjects
- **ROI Operations**: Streamlined mask loading and time series extraction
- **Error Handling**: Robust error handling with custom exceptions
- **Metadata Management**: Consistent data saving/loading with metadata

### Benefits

- **Reduced Code Duplication**: No more copy-pasted data loading code
- **Consistent Quality Control**: Standardized validation across all scripts
- **Easier Maintenance**: Update data handling logic in one place
- **Better Error Handling**: Clear error messages and graceful failure handling
- **Improved Documentation**: Self-documenting functions with comprehensive help

## Quick Start

```python
from data_utils import (
    load_behavioral_data, load_fmri_data, get_complete_subjects,
    SubjectManager, check_data_integrity
)
from oak_storage_config import OAKConfig

# Get configuration
config = OAKConfig()

# Find subjects with complete data
subjects = get_complete_subjects(config)
print(f"Found {len(subjects)} subjects")

# Load data for a subject
subject_id = subjects[0]
behavioral_data = load_behavioral_data(subject_id, config)
fmri_img = load_fmri_data(subject_id, config)

# Check data integrity
integrity_report = check_data_integrity(subjects[:5], config)
print(integrity_report)
```

## Core Functions

### Subject Discovery and Management

#### `get_complete_subjects(config=None)`
Get list of subjects with complete, validated data.

```python
subjects = get_complete_subjects()
# Returns: ['sub-001', 'sub-002', ...]
```

#### `SubjectManager`
Class for advanced subject discovery and management.

```python
manager = SubjectManager(config)

# Get subjects with both fMRI and behavioral data
complete = manager.get_available_subjects(require_both=True)

# Get subjects with any available data
any_data = manager.get_available_subjects(require_both=False)

# Get detailed summary
summary = manager.get_subject_summary()
print(summary)  # DataFrame with data availability and quality metrics
```

### Data Loading Functions

#### `load_behavioral_data(subject_id, config=None, validate=True, compute_sv=True)`
Load and process behavioral data with validation and subjective value computation.

```python
# Basic loading
behavioral_data = load_behavioral_data('sub-001', config)

# With full processing (recommended)
behavioral_data = load_behavioral_data(
    'sub-001', 
    config, 
    validate=True,     # Run quality control
    compute_sv=True    # Compute subjective values
)

# Columns available: onset, choice, delay_days, sv_chosen, sv_unchosen, etc.
```

#### `load_fmri_data(subject_id, config=None, smoothed=False, validate=True)`
Load fMRI data with validation.

```python
# Load raw fMRI data
fmri_img = load_fmri_data('sub-001', config)

# Load smoothed data (if available)
fmri_img = load_fmri_data('sub-001', config, smoothed=True)

# Returns nibabel.Nifti1Image object
print(f"Shape: {fmri_img.shape}")
```

#### `load_confounds(subject_id, config=None, selected_confounds=None)`
Load fMRIPrep confounds with automatic selection.

```python
# Load default confounds (motion, CSF, WM, global signal)
confounds = load_confounds('sub-001', config)

# Load specific confounds
confounds = load_confounds('sub-001', config, 
                          selected_confounds=['trans_x', 'trans_y', 'rot_x'])

# Returns pandas DataFrame or None
```

### ROI and Mask Operations

#### `check_mask_exists(mask_path, create_if_missing=False)`
Check if mask file exists and optionally create it.

```python
from oak_storage_config import OAKConfig
config = OAKConfig()

# Check if mask exists
exists = check_mask_exists(config.ROI_MASKS['striatum'])

# Attempt to create if missing
exists = check_mask_exists(config.ROI_MASKS['striatum'], create_if_missing=True)
```

#### `load_mask(mask_path, validate=True)`
Load ROI mask with validation.

```python
mask_img = load_mask(config.ROI_MASKS['dlpfc'])
print(f"Mask has {(mask_img.get_fdata() > 0).sum()} voxels")
```

#### `extract_roi_timeseries(subject_id, roi_name, config=None, **kwargs)`
Extract time series data from ROI.

```python
# Extract striatum time series
timeseries = extract_roi_timeseries(
    'sub-001', 
    'striatum',
    config,
    smoothed=False,
    standardize=True,
    detrend=True
)

# Returns numpy array: (n_timepoints, n_voxels)
print(f"Extracted time series: {timeseries.shape}")
```

### Data Validation

#### `DataValidator`
Class for comprehensive data validation.

```python
validator = DataValidator(config)

# Validate behavioral data
validation = validator.validate_behavioral_data(behavioral_df, 'sub-001')
print(f"Valid: {validation['valid']}")
print(f"Warnings: {validation['warnings']}")
print(f"Metrics: {validation['metrics']}")

# Validate fMRI data
validation = validator.validate_fmri_data(fmri_path, 'sub-001')
```

#### `check_data_integrity(subject_ids=None, config=None)`
Run comprehensive data integrity check.

```python
# Check all subjects
report = check_data_integrity()

# Check specific subjects
report = check_data_integrity(['sub-001', 'sub-002'], config)

# Returns detailed pandas DataFrame
print(report[['subject_id', 'complete', 'n_trials', 'accuracy']])
```

### Data Persistence

#### `save_processed_data(data, output_path, subject_id=None)`
Save processed data with metadata.

```python
results = {
    'mvpa_accuracy': 0.75,
    'behavioral_k': 0.1,
    'analysis_date': '2024-01-15'
}

save_processed_data(results, 'subject_results.pkl', subject_id='sub-001')
```

#### `load_processed_data(input_path)`
Load processed data with metadata.

```python
data, metadata = load_processed_data('subject_results.pkl')
print(f"Data: {data}")
print(f"Metadata: {metadata}")
```

## Integration with Existing Scripts

### Refactored Scripts

The following scripts have been updated to use `data_utils`:

1. **`delay_discounting_mvpa_pipeline.py`** - Main MVPA pipeline
2. **`analyze_results.py`** - Results analysis  
3. **`create_roi_masks.py`** - ROI mask creation

### Usage Examples

#### In MVPA Pipeline

```python
# OLD way (before data_utils)
behavior_file = f"{config.BEHAVIOR_DIR}/{subject_id}_discountFix_events.tsv"
if not os.path.exists(behavior_file):
    return {'success': False, 'error': 'File not found'}
events_df = pd.read_csv(behavior_file, sep='\t')
# ... manual processing ...

# NEW way (with data_utils)
try:
    events_df = load_behavioral_data(subject_id, config, validate=True, compute_sv=True)
    # Data is already processed and validated!
except DataError as e:
    return {'success': False, 'error': str(e)}
```

#### In Results Analysis

```python
# OLD way
with open(results_file, 'rb') as f:
    results = pickle.load(f)

# NEW way  
results, metadata = load_processed_data(results_file)
print(f"Analysis metadata: {metadata}")
```

#### In ROI Mask Creation

```python
# OLD way
if not os.path.exists(mask_path):
    print(f"Mask not found: {mask_path}")

# NEW way
if not check_mask_exists(mask_path, create_if_missing=True):
    print(f"Could not create mask: {mask_path}")
```

## Error Handling

The module uses a custom `DataError` exception for data-related issues:

```python
from data_utils import DataError

try:
    data = load_behavioral_data('sub-999', config)
except DataError as e:
    print(f"Data loading failed: {e}")
    # Handle gracefully
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## Configuration Integration

All functions work seamlessly with `OAKConfig`:

```python
from oak_storage_config import OAKConfig
from data_utils import *

config = OAKConfig()

# All functions automatically use config paths
subjects = get_complete_subjects(config)
data = load_behavioral_data(subjects[0], config)
```

## Quality Control Features

### Automatic Validation

- **Behavioral Data**: Checks for required columns, missing responses, reaction times
- **fMRI Data**: Validates intensity ranges, checks for NaN/infinite values
- **File Integrity**: Ensures files exist and are readable

### Quality Metrics

```python
# Get detailed quality metrics
manager = SubjectManager(config)
summary = manager.get_subject_summary()

# Columns include:
# - subject_id: Subject identifier
# - has_fmri, has_behavior: Data availability
# - behavior_valid, fmri_valid: Data validity
# - n_trials: Number of behavioral trials
# - accuracy: Behavioral accuracy
# - complete: Overall data completeness
```

## Performance Features

### Caching and Memory Management

- **Nilearn Caching**: Automatic caching of preprocessing steps
- **Lazy Loading**: Data loaded only when needed
- **Memory Efficient**: Minimal memory footprint

### Parallel Processing Ready

All functions are designed to work with parallel processing:

```python
from joblib import Parallel, delayed

# Process multiple subjects in parallel
def process_subject(subject_id):
    behavioral_data = load_behavioral_data(subject_id, config)
    fmri_data = load_fmri_data(subject_id, config)
    return analyze_subject(behavioral_data, fmri_data)

results = Parallel(n_jobs=4)(
    delayed(process_subject)(subject_id) 
    for subject_id in subjects
)
```

## Testing and Debugging

### Demo Script

Run `demo_data_utils.py` to test all functionality:

```bash
# Full demonstration
python demo_data_utils.py

# Quick demo (faster)
python demo_data_utils.py --quick

# Skip data loading (even faster)
python demo_data_utils.py --skip-loading
```

### Built-in Testing

The module includes built-in testing capabilities:

```bash
# Test data utilities directly
python data_utils.py
```

### Data Integrity Checking

Use the analysis script to check data integrity:

```bash
# Check all subjects
python analyze_results.py --check_data
```

## Migration Guide

### For Existing Scripts

1. **Add Import**:
   ```python
   from data_utils import load_behavioral_data, load_fmri_data, DataError
   ```

2. **Replace Manual Loading**:
   ```python
   # Replace this
   events_df = pd.read_csv(behavior_file, sep='\t')
   
   # With this
   events_df = load_behavioral_data(subject_id, config)
   ```

3. **Add Error Handling**:
   ```python
   try:
       data = load_behavioral_data(subject_id, config)
   except DataError as e:
       print(f"Data error: {e}")
       continue
   ```

### Benefits of Migration

- **Reduced Lines of Code**: 50-80% reduction in data loading code
- **Improved Reliability**: Standardized error handling and validation
- **Better Performance**: Optimized loading and caching
- **Easier Maintenance**: Single source of truth for data operations

## Advanced Usage

### Custom Subject Discovery

```python
manager = SubjectManager(config)

# Find subjects with specific criteria
all_subjects = manager.get_available_subjects(require_both=False)
valid_subjects = []

for subject_id in all_subjects:
    try:
        behavioral_data = load_behavioral_data(subject_id, config)
        if len(behavioral_data) >= 50:  # At least 50 trials
            valid_subjects.append(subject_id)
    except DataError:
        continue

print(f"Found {len(valid_subjects)} subjects with >=50 trials")
```

### Custom Validation

```python
validator = DataValidator(config)

def custom_validation(behavioral_data, subject_id):
    """Custom validation logic"""
    validation = validator.validate_behavioral_data(behavioral_data, subject_id)
    
    # Add custom checks
    if behavioral_data['choice'].mean() < 0.1:
        validation['warnings'].append("Very low choice rate")
    
    return validation
```

### Batch Processing

```python
# Process all subjects efficiently
config = OAKConfig()
subjects = get_complete_subjects(config)

results = []
for subject_id in subjects:
    try:
        # Load data
        behavioral_data = load_behavioral_data(subject_id, config)
        fmri_img = load_fmri_data(subject_id, config)
        
        # Extract ROI time series
        timeseries = extract_roi_timeseries(subject_id, 'striatum', config)
        
        # Analyze...
        result = your_analysis_function(behavioral_data, timeseries)
        results.append(result)
        
    except DataError as e:
        print(f"Skipping {subject_id}: {e}")
        continue

# Save results
save_processed_data(results, 'batch_results.pkl')
```

## Troubleshooting

### Common Issues

1. **Missing Masks**: Run `python create_roi_masks.py`
2. **Path Issues**: Check `oak_storage_config.py` paths
3. **Permission Errors**: Ensure read access to data directories
4. **Memory Issues**: Use `smoothed=False` and process subjects individually

### Debug Information

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all data_utils operations will show detailed logs
data = load_behavioral_data(subject_id, config)
```

### Getting Help

1. Run `python demo_data_utils.py` for working examples
2. Check function docstrings: `help(load_behavioral_data)`
3. Review this documentation for usage patterns
4. Check the refactored scripts for integration examples

## Summary

The `data_utils` module transforms the delay discounting analysis pipeline by:

- **Centralizing** all data operations in one module
- **Standardizing** data loading and validation across scripts  
- **Simplifying** script development and maintenance
- **Improving** error handling and data quality control
- **Enabling** efficient batch processing and parallel analysis

Use this module as the foundation for all data operations in your delay discounting analyses! 