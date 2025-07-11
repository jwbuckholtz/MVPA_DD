# ROI Mask Validation Guide

## Overview

The delay discounting MVPA pipeline now uses **pre-existing ROI masks** stored on OAK instead of creating them on-the-fly. This approach provides better consistency, performance, and reliability for the analysis pipeline.

## Key Changes

### ✅ New Approach: Pre-existing Masks on OAK
- **Masks are stored centrally** on OAK storage
- **Validation before use** ensures quality and consistency
- **Faster pipeline startup** - no mask creation delays
- **Better reproducibility** - same masks across all analyses

### ❌ Deprecated Approach: On-the-fly Creation
- `create_roi_masks.py` is now **DEPRECATED**
- No longer creates masks from Harvard-Oxford atlas
- Mask creation functionality removed from pipeline

## ROI Masks Configuration

### Core ROI Masks (Required)
These masks are **required** for the pipeline to run:

```python
CORE_ROI_MASKS = ['striatum', 'dlpfc', 'vmpfc']
```

- **Striatum**: Caudate, putamen, nucleus accumbens regions
- **DLPFC**: Dorsolateral prefrontal cortex (BA 9/46)
- **VMPFC**: Ventromedial prefrontal cortex (BA 10/11/32)

### Optional ROI Masks (Available if present)
These masks are used if available:

```python
OPTIONAL_ROI_MASKS = [
    'left_striatum', 'right_striatum',
    'left_dlpfc', 'right_dlpfc', 
    'acc',  # Anterior Cingulate Cortex
    'ofc'   # Orbitofrontal Cortex
]
```

### Mask Locations
All masks are stored in:
```
/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/masks/
```

## Mask Validation Script

### `validate_roi_masks.py`

This script replaces `create_roi_masks.py` and provides comprehensive mask validation.

#### Basic Usage
```bash
# Validate all masks
python validate_roi_masks.py

# Check OAK connectivity only
python validate_roi_masks.py --check-connectivity

# Create inventory of all mask files
python validate_roi_masks.py --inventory

# Specify output directory
python validate_roi_masks.py --output-dir ./validation_results
```

#### Key Features

1. **OAK Connectivity Check**
   - Verifies access to OAK storage
   - Checks data root and masks directory

2. **Comprehensive Validation**
   - Validates all core and optional masks
   - Checks file existence, integrity, and quality
   - Validates mask dimensions, resolution, and voxel counts

3. **Detailed Reporting**
   - Creates validation summary CSV
   - Generates detailed text report
   - Provides visualization of valid masks

4. **Inventory Management**
   - Lists all mask files in directory
   - Identifies configured vs. unconfigured masks
   - Reports file sizes and properties

### Example Output

```
Validating ROI masks...

Core ROI masks:
  ✓ striatum: 2847 voxels
  ✓ dlpfc: 1523 voxels
  ✓ vmpfc: 891 voxels

Optional ROI masks:
  ✓ left_striatum: 1402 voxels
  ✓ right_striatum: 1445 voxels
  ✗ acc: File not found
  ✗ ofc: File not found

Validation Summary:
Core masks valid: 3/3
Optional masks available: 2/6

✓ All core ROI masks are valid - pipeline ready!
Available ROIs: striatum, dlpfc, vmpfc, left_striatum, right_striatum
```

## Integration with Pipeline

### Updated `oak_storage_config.py`

The configuration now includes:
- `CORE_ROI_MASKS`: Required masks for analysis
- `OPTIONAL_ROI_MASKS`: Additional masks used if available
- Expanded `ROI_MASKS` dictionary with more mask options

### Updated `data_utils.py`

Key changes:
- `check_mask_exists()`: Simplified to only check existence
- Removed `create_if_missing` parameter
- Enhanced error handling for missing masks

### Updated Pipeline Scripts

- **`delay_discounting_mvpa_pipeline.py`**: Enhanced ROI masker creation with validation
- **`submit_analysis_job.sh`**: Uses mask validation instead of creation
- **`analyze_results.py`**: Works with validated masks

## Usage in Analysis Pipeline

### 1. Validate Masks Before Analysis

```bash
# Always validate masks first
python validate_roi_masks.py
```

### 2. Check Available ROIs

```python
from validate_roi_masks import MaskValidator
from oak_storage_config import OAKConfig

validator = MaskValidator(OAKConfig())
available_rois = validator.get_available_rois()
print(f"Available ROIs: {', '.join(available_rois)}")
```

### 3. Use in Analysis Scripts

```python
from data_utils import check_mask_exists, load_mask
from oak_storage_config import OAKConfig

config = OAKConfig()

# Check if core masks are available
for roi_name in config.CORE_ROI_MASKS:
    mask_path = config.ROI_MASKS[roi_name]
    if check_mask_exists(mask_path):
        mask_img = load_mask(mask_path, validate=True)
        print(f"✓ {roi_name} mask loaded: {(mask_img.get_fdata() > 0).sum()} voxels")
    else:
        print(f"✗ {roi_name} mask not found")
```

## SLURM Job Integration

The updated `submit_analysis_job.sh` now:

1. **Validates masks** instead of creating them
2. **Checks OAK connectivity** before proceeding
3. **Reports available ROIs** for the analysis
4. **Fails early** if core masks are missing

```bash
# Updated SLURM job step
echo "Step 1: Validating pre-existing ROI masks on OAK..."
python validate_roi_masks.py --check-connectivity
```

## Troubleshooting

### Common Issues

1. **OAK Not Accessible**
   ```
   Error: Cannot access OAK storage
   ```
   - Ensure you're connected to Stanford network
   - Check VPN connection if working remotely
   - Verify OAK mount points

2. **Core Masks Missing**
   ```
   Error: Not all core ROI masks are valid
   ```
   - Contact data administrator
   - Verify mask file paths in configuration
   - Check file permissions

3. **Mask Validation Failures**
   ```
   Error: Empty mask (no voxels)
   ```
   - Mask file may be corrupted
   - Re-download or regenerate mask
   - Check mask creation pipeline

### Debugging Steps

1. **Check OAK connectivity**:
   ```bash
   python validate_roi_masks.py --check-connectivity
   ```

2. **Create mask inventory**:
   ```bash
   python validate_roi_masks.py --inventory
   ```

3. **Run full validation**:
   ```bash
   python validate_roi_masks.py --output-dir ./debug_validation
   ```

4. **Check individual masks**:
   ```python
   from data_utils import check_mask_exists, load_mask
   from oak_storage_config import OAKConfig
   
   config = OAKConfig()
   for roi_name, mask_path in config.ROI_MASKS.items():
       exists = check_mask_exists(mask_path)
       print(f"{roi_name}: {'✓' if exists else '✗'}")
   ```

## Migration Guide

### For Existing Users

1. **Update your workflow**:
   ```bash
   # OLD: Create masks
   python create_roi_masks.py
   
   # NEW: Validate masks
   python validate_roi_masks.py
   ```

2. **Update script imports**:
   ```python
   # OLD: Import mask creation
   from create_roi_masks import create_all_masks
   
   # NEW: Import mask validation
   from validate_roi_masks import MaskValidator
   ```

3. **Update configuration usage**:
   ```python
   # NEW: Use core/optional mask categories
   config = OAKConfig()
   core_masks = config.CORE_ROI_MASKS
   optional_masks = config.OPTIONAL_ROI_MASKS
   ```

### For New Users

1. **Start with validation**:
   ```bash
   python validate_roi_masks.py
   ```

2. **Check available ROIs**:
   ```bash
   python validate_roi_masks.py --inventory
   ```

3. **Run analysis**:
   ```bash
   sbatch submit_analysis_job.sh
   ```

## Best Practices

1. **Always validate masks** before running analyses
2. **Check OAK connectivity** first when troubleshooting
3. **Use core masks** for consistent analyses across studies
4. **Leverage optional masks** for additional regions of interest
5. **Monitor validation reports** for mask quality issues
6. **Document mask versions** used in your analyses

## Demo Script

Use the demo script to test the new functionality:

```bash
python demo_mask_validation.py
```

This will demonstrate:
- OAK connectivity checking
- Mask inventory creation
- Comprehensive validation
- Individual mask checking
- Visualization generation

## Support

For questions or issues:
1. Check this documentation
2. Run the demo script
3. Contact the Cognitive Neuroscience Lab
4. Submit an issue on the project repository

---

**Last updated**: January 2025  
**Version**: 2.0 (OAK Migration)  
**Authors**: Cognitive Neuroscience Lab, Stanford University 