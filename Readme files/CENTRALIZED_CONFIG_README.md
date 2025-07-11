# Centralized Configuration System

## Overview

The centralized configuration system consolidates all scattered configuration values across the MVPA analysis pipeline into a single, comprehensive YAML file. This replaces the previous system where configuration was scattered across multiple files (`oak_storage_config.py`, `dd_geometry_config.json`, hardcoded values in scripts, etc.).

## Key Benefits

✅ **Single source of truth** - All configuration in one place  
✅ **Better maintainability** - Easy to update and track changes  
✅ **Backward compatibility** - Existing code continues to work  
✅ **Environment overrides** - Runtime configuration changes via environment variables  
✅ **Validation** - Built-in validation and error checking  
✅ **Documentation** - Self-documenting configuration with comments  
✅ **Flexibility** - Easy to extend and customize  

## Files in the Configuration System

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration file with all parameters |
| `config_loader.py` | Configuration loader with structured access |
| `config_migration.py` | Migration utility for existing configs |
| `demo_config_system.py` | Demonstration of usage |
| `CENTRALIZED_CONFIG_README.md` | This documentation |

## Quick Start

### 1. Basic Usage

```python
from config_loader import Config

# Load configuration
config = Config()

# Access configuration values
print(f"TR: {config.fmri.tr}")
print(f"ROI masks: {config.roi_masks.core_rois}")
print(f"N permutations: {config.mvpa.n_permutations}")
```

### 2. Legacy Compatibility

```python
# Your existing code continues to work
oak_config = config.get_legacy_oak_config()
mvpa_config = config.get_legacy_mvpa_config()

# Use exactly like before
tr = oak_config.TR
roi_masks = oak_config.ROI_MASKS
cv_folds = mvpa_config.CV_FOLDS
```

### 3. Migration from Existing Configs

```bash
# Migrate all existing configuration files
python config_migration.py --migrate-all

# Or migrate specific files
python config_migration.py --migrate-from oak_storage_config.py

# Validate migration
python config_migration.py --validate-migration
```

## Configuration Structure

The configuration is organized into logical sections:

### Core Sections

| Section | Purpose | Examples |
|---------|---------|----------|
| `study` | Study information | name, PI, contact |
| `paths` | Data and output paths | data_root, output_dir, masks_dir |
| `fmri` | fMRI parameters | tr, hemi_lag, smoothing |
| `roi_masks` | ROI mask configuration | core_rois, mask_files |
| `behavioral` | Behavioral analysis | min_accuracy, variables |
| `mvpa` | MVPA parameters | cv_folds, n_permutations, algorithms |
| `geometry` | Neural geometry analysis | dimensionality_reduction, comparisons |
| `parallel` | Parallel processing | n_jobs, resource_management |
| `memory` | Memory efficiency | memory_mapping, monitoring |
| `caching` | Caching system | cache_dir, invalidation |
| `logging` | Logging configuration | level, components |
| `slurm` | HPC configuration | job_name, resources |

### Example Configuration Access

```python
config = Config()

# fMRI parameters
tr = config.fmri.tr
hemi_lag = config.fmri.hemi_lag

# ROI masks
core_rois = config.roi_masks.core_rois
all_masks = config.get_roi_mask_paths()

# MVPA parameters
cv_folds = config.mvpa.cv_folds
n_permutations = config.mvpa.n_permutations
svm_c = config.mvpa.classification['algorithms']['svm']['C']

# Parallel processing
n_jobs = config.parallel.n_jobs
memory_limit = config.parallel.resource_management['memory_limit_gb']
```

## Advanced Features

### Environment Variable Overrides

You can override any configuration value using environment variables:

```bash
# Override TR
export MVPA_FMRI_TR=2.0

# Override CV folds
export MVPA_MVPA_CV_FOLDS=10

# Override parallel jobs
export MVPA_PARALLEL_N_JOBS=8
```

```python
# Load with environment overrides
config = Config(environment_overrides=True)
print(f"TR: {config.fmri.tr}")  # Will use overridden value
```

### Dynamic Configuration Updates

```python
config = Config()

# Update configuration at runtime
updates = {
    'fmri': {'tr': 2.0},
    'mvpa': {'n_permutations': 5000}
}
config.update_from_dict(updates)
```

### Configuration Serialization

```python
# Convert to dictionary
config_dict = config.to_dict()

# Convert to JSON
json_str = config.to_json()

# Save to file
config.save_yaml('my_config.yaml')
```

## Migration Guide

### From Scattered Configs to Centralized

**Before (scattered configs):**
```python
from oak_storage_config import OAKConfig
from mvpa_utils import MVPAConfig

config = OAKConfig()
mvpa_config = MVPAConfig()

tr = config.TR
roi_masks = config.ROI_MASKS
cv_folds = mvpa_config.CV_FOLDS
```

**After (centralized config):**
```python
from config_loader import Config

config = Config()

# New way (structured access)
tr = config.fmri.tr
roi_masks = config.get_roi_mask_paths()
cv_folds = config.mvpa.cv_folds

# OR legacy compatibility way
oak_config = config.get_legacy_oak_config()
mvpa_config = config.get_legacy_mvpa_config()
tr = oak_config.TR  # Works exactly like before
```

### Migration Steps

1. **Backup existing configs:**
   ```bash
   python config_migration.py --create-backup
   ```

2. **Migrate all configs:**
   ```bash
   python config_migration.py --migrate-all
   ```

3. **Validate migration:**
   ```bash
   python config_migration.py --validate-migration
   ```

4. **Test your pipeline:**
   ```bash
   python demo_config_system.py
   ```

5. **Update your scripts** (optional - legacy compatibility means this isn't required)

## Pipeline Integration

### Minimal Changes Required

The centralized configuration system is designed for **minimal disruption**. Your existing pipeline scripts should continue to work with minimal or no changes.

### Option 1: Use Legacy Compatibility (Recommended)

```python
# In your existing pipeline scripts, just change the import:

# OLD:
# from oak_storage_config import OAKConfig
# config = OAKConfig()

# NEW:
from config_loader import get_oak_config
config = get_oak_config()

# Everything else stays the same!
```

### Option 2: Use New Structured Access

```python
# Gradually migrate to new structured access
from config_loader import Config

config = Config()

# Use structured access for new code
tr = config.fmri.tr
roi_masks = config.get_roi_mask_paths()
cv_folds = config.mvpa.cv_folds
```

### Option 3: Hybrid Approach

```python
# Use both approaches during transition
from config_loader import Config

config = Config()

# Use structured access for new features
memory_threshold = config.memory.memory_mapping['threshold_gb']

# Use legacy compatibility for existing code
oak_config = config.get_legacy_oak_config()
tr = oak_config.TR
```

## Configuration Validation

The system includes comprehensive validation:

```python
# Configuration is automatically validated on load
config = Config()  # Raises ConfigError if invalid

# Manual validation
config._validate_config()
```

**Validation checks:**
- fMRI TR must be positive
- MVPA CV folds must be ≥ 2
- Required paths must be specified
- ROI core masks cannot be empty
- Permutation count must be ≥ 1

## Customization

### Custom Configuration Files

```python
# Use custom configuration file
config = Config('my_custom_config.yaml')

# Use different config for different environments
config = Config('config_development.yaml')
config = Config('config_production.yaml')
```

### Extending Configuration

Add new sections to `config.yaml`:

```yaml
# Add custom analysis section
custom_analysis:
  enabled: true
  method: "my_method"
  parameters:
    alpha: 0.05
    iterations: 1000
```

Access in code:
```python
config = Config()
custom_settings = config.advanced.get('custom_analysis', {})
```

## Best Practices

### 1. Use Environment Variables for Deployment

```bash
# Development
export MVPA_PARALLEL_N_JOBS=1
export MVPA_LOGGING_LEVEL=DEBUG

# Production  
export MVPA_PARALLEL_N_JOBS=16
export MVPA_LOGGING_LEVEL=INFO
```

### 2. Create Environment-Specific Configs

```
config_development.yaml
config_testing.yaml
config_production.yaml
```

### 3. Use Configuration Versioning

```yaml
study:
  name: "Delay Discounting MVPA Analysis"
  version: "1.0"
```

### 4. Document Your Changes

```yaml
# Custom modification for experiment 2
mvpa:
  n_permutations: 5000  # Increased for higher precision
```

## Troubleshooting

### Common Issues

**1. Configuration file not found**
```
ConfigError: Configuration file not found: config.yaml
```
Solution: Make sure `config.yaml` exists in your working directory.

**2. YAML syntax errors**
```
ConfigError: Error parsing YAML configuration
```
Solution: Check YAML syntax. Use proper indentation and quotes.

**3. Validation errors**
```
ConfigError: Configuration validation failed: fMRI TR must be positive
```
Solution: Fix the invalid configuration values.

**4. Missing dependencies**
```
ImportError: No module named 'yaml'
```
Solution: Install PyYAML: `pip install PyYAML`

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

config = Config()
```

## Performance

The centralized configuration system is designed for performance:

- **Fast loading**: YAML parsing is optimized
- **Lazy evaluation**: Configuration sections loaded on demand
- **Caching**: Singleton pattern for main config instance
- **Memory efficient**: Structured objects minimize memory usage

## Testing

Test your configuration setup:

```bash
# Run full demo
python demo_config_system.py

# Test specific features
python -c "from config_loader import Config; print(Config().fmri.tr)"

# Test migration
python config_migration.py --validate-migration
```

## FAQ

**Q: Do I need to rewrite my existing pipeline scripts?**
A: No! The legacy compatibility system means your existing scripts should work without changes.

**Q: Can I still use the old config files?**
A: Yes, but we recommend migrating to the centralized system for better maintainability.

**Q: What if I need to add new configuration parameters?**
A: Add them to `config.yaml` and they'll be automatically available through the configuration system.

**Q: How do I handle different configurations for different environments?**
A: Use environment variables or create environment-specific config files.

**Q: Is the configuration system backwards compatible?**
A: Yes, the legacy compatibility functions ensure existing code continues to work.

## Support

For questions or issues:

1. Check this documentation
2. Run the demo script: `python demo_config_system.py`
3. Use the migration utility: `python config_migration.py --help`
4. Review the configuration file: `config.yaml`

## Migration Summary

| Task | Command | Description |
|------|---------|-------------|
| Backup configs | `python config_migration.py --create-backup` | Create backup of existing configs |
| Migrate all | `python config_migration.py --migrate-all` | Migrate all known config files |
| Validate | `python config_migration.py --validate-migration` | Validate migration results |
| Test system | `python demo_config_system.py` | Run comprehensive demo |
| Use in pipeline | `from config_loader import get_oak_config` | Minimal change to existing code |

The centralized configuration system provides a robust, maintainable solution for managing all your MVPA analysis parameters while maintaining full backward compatibility with existing code. 