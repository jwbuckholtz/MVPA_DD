#!/usr/bin/env python3
"""
Demonstration of Centralized Configuration System
===============================================

This script demonstrates how to use the new centralized YAML configuration
system that replaces the scattered configuration files.

Features demonstrated:
- Loading centralized configuration
- Accessing configuration values
- Backward compatibility with legacy config classes
- Environment variable overrides
- Configuration validation
- Dynamic configuration updates

Usage:
    python demo_config_system.py

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import Config, get_oak_config, get_mvpa_config, get_main_config


def demo_basic_usage():
    """Demonstrate basic configuration usage"""
    print("=" * 60)
    print("1. BASIC CONFIGURATION USAGE")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    
    # Access configuration values using structured objects
    print(f"Study Name: {config.study.name}")
    print(f"TR: {config.fmri.tr} seconds")
    print(f"Hemodynamic Lag: {config.fmri.hemi_lag} TRs")
    print(f"Data Root: {config.paths.data_root}")
    print(f"Core ROIs: {config.roi_masks.core_rois}")
    print(f"CV Folds: {config.mvpa.cv_folds}")
    print(f"N Permutations: {config.mvpa.n_permutations}")
    print(f"Parallel Jobs: {config.parallel.n_jobs}")
    
    print("\n✓ Basic configuration access works!")


def demo_roi_masks():
    """Demonstrate ROI mask configuration"""
    print("=" * 60)
    print("2. ROI MASKS CONFIGURATION")
    print("=" * 60)
    
    config = Config()
    
    # Get all ROI mask paths
    all_masks = config.get_roi_mask_paths()
    print(f"All ROI masks ({len(all_masks)}):")
    for roi, path in all_masks.items():
        print(f"  {roi}: {path}")
    
    # Get core ROI masks only
    core_masks = config.get_core_roi_mask_paths()
    print(f"\nCore ROI masks ({len(core_masks)}):")
    for roi, path in core_masks.items():
        print(f"  {roi}: {path}")
    
    print("\n✓ ROI mask configuration works!")


def demo_output_paths():
    """Demonstrate output path configuration"""
    print("=" * 60)
    print("3. OUTPUT PATHS CONFIGURATION")
    print("=" * 60)
    
    config = Config()
    
    # Get all output paths
    output_paths = config.get_output_paths()
    print("Output paths:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")
    
    print("\n✓ Output path configuration works!")


def demo_legacy_compatibility():
    """Demonstrate backward compatibility with legacy config classes"""
    print("=" * 60)
    print("4. LEGACY COMPATIBILITY")
    print("=" * 60)
    
    config = Config()
    
    # Get legacy OAKConfig-compatible object
    oak_config = config.get_legacy_oak_config()
    print("Legacy OAKConfig attributes:")
    print(f"  DATA_ROOT: {oak_config.DATA_ROOT}")
    print(f"  TR: {oak_config.TR}")
    print(f"  HEMI_LAG: {oak_config.HEMI_LAG}")
    print(f"  CV_FOLDS: {oak_config.CV_FOLDS}")
    print(f"  N_PERMUTATIONS: {oak_config.N_PERMUTATIONS}")
    print(f"  ROI_MASKS: {len(oak_config.ROI_MASKS)} masks")
    
    # Get legacy MVPAConfig-compatible object
    mvpa_config = config.get_legacy_mvpa_config()
    print("\nLegacy MVPAConfig attributes:")
    print(f"  CV_FOLDS: {mvpa_config.CV_FOLDS}")
    print(f"  N_PERMUTATIONS: {mvpa_config.N_PERMUTATIONS}")
    print(f"  DEFAULT_CLASSIFIER: {mvpa_config.DEFAULT_CLASSIFIER}")
    print(f"  DEFAULT_REGRESSOR: {mvpa_config.DEFAULT_REGRESSOR}")
    print(f"  SVM_C: {mvpa_config.SVM_C}")
    print(f"  RIDGE_ALPHA: {mvpa_config.RIDGE_ALPHA}")
    
    print("\n✓ Legacy compatibility works!")


def demo_convenience_functions():
    """Demonstrate convenience functions"""
    print("=" * 60)
    print("5. CONVENIENCE FUNCTIONS")
    print("=" * 60)
    
    # Use convenience functions
    oak_config = get_oak_config()
    mvpa_config = get_mvpa_config()
    main_config = get_main_config()
    
    print(f"OAK Config TR: {oak_config.TR}")
    print(f"MVPA Config CV Folds: {mvpa_config.CV_FOLDS}")
    print(f"Main Config Study: {main_config.study.name}")
    
    print("\n✓ Convenience functions work!")


def demo_environment_overrides():
    """Demonstrate environment variable overrides"""
    print("=" * 60)
    print("6. ENVIRONMENT VARIABLE OVERRIDES")
    print("=" * 60)
    
    # Set environment variables
    os.environ['MVPA_FMRI_TR'] = '2.0'
    os.environ['MVPA_MVPA_CV_FOLDS'] = '10'
    os.environ['MVPA_PARALLEL_N_JOBS'] = '4'
    
    # Load configuration with overrides
    config = Config(environment_overrides=True)
    
    print(f"Original TR: 0.68")
    print(f"Overridden TR: {config.fmri.tr}")
    print(f"Original CV Folds: 5")
    print(f"Overridden CV Folds: {config.mvpa.cv_folds}")
    print(f"Original N Jobs: -1")
    print(f"Overridden N Jobs: {config.parallel.n_jobs}")
    
    # Clean up environment
    del os.environ['MVPA_FMRI_TR']
    del os.environ['MVPA_MVPA_CV_FOLDS']
    del os.environ['MVPA_PARALLEL_N_JOBS']
    
    print("\n✓ Environment overrides work!")


def demo_configuration_updates():
    """Demonstrate dynamic configuration updates"""
    print("=" * 60)
    print("7. DYNAMIC CONFIGURATION UPDATES")
    print("=" * 60)
    
    config = Config()
    
    print(f"Original TR: {config.fmri.tr}")
    print(f"Original N Permutations: {config.mvpa.n_permutations}")
    
    # Update configuration
    updates = {
        'fmri': {'tr': 2.0},
        'mvpa': {'n_permutations': 5000}
    }
    config.update_from_dict(updates)
    
    print(f"Updated TR: {config.fmri.tr}")
    print(f"Updated N Permutations: {config.mvpa.n_permutations}")
    
    print("\n✓ Dynamic configuration updates work!")


def demo_configuration_serialization():
    """Demonstrate configuration serialization"""
    print("=" * 60)
    print("8. CONFIGURATION SERIALIZATION")
    print("=" * 60)
    
    config = Config()
    
    # Convert to dictionary
    config_dict = config.to_dict()
    print(f"Configuration as dictionary: {len(config_dict)} top-level keys")
    
    # Convert to JSON
    json_str = config.to_json()
    print(f"Configuration as JSON: {len(json_str)} characters")
    
    # Save to file
    config.save_yaml('demo_config_output.yaml')
    print("Configuration saved to demo_config_output.yaml")
    
    # Clean up
    if Path('demo_config_output.yaml').exists():
        Path('demo_config_output.yaml').unlink()
    
    print("\n✓ Configuration serialization works!")


def demo_pipeline_integration():
    """Demonstrate how to integrate with existing pipeline"""
    print("=" * 60)
    print("9. PIPELINE INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Example of how to use in existing pipeline
    print("Example pipeline integration:")
    print("""
# OLD WAY (scattered configs):
from oak_storage_config import OAKConfig
from mvpa_utils import MVPAConfig

config = OAKConfig()
mvpa_config = MVPAConfig()

# Use scattered config values
tr = config.TR
roi_masks = config.ROI_MASKS
cv_folds = mvpa_config.CV_FOLDS

# NEW WAY (centralized config):
from config_loader import Config

config = Config()

# Use centralized config values
tr = config.fmri.tr
roi_masks = config.get_roi_mask_paths()
cv_folds = config.mvpa.cv_folds

# OR use legacy compatibility
oak_config = config.get_legacy_oak_config()
mvpa_config = config.get_legacy_mvpa_config()

# Existing code works unchanged
tr = oak_config.TR
roi_masks = oak_config.ROI_MASKS
cv_folds = mvpa_config.CV_FOLDS
""")
    
    print("✓ Pipeline integration examples shown!")


def demo_advanced_features():
    """Demonstrate advanced configuration features"""
    print("=" * 60)
    print("10. ADVANCED FEATURES")
    print("=" * 60)
    
    config = Config()
    
    # Access nested configuration
    print("Nested configuration access:")
    print(f"  SVM C parameter: {config.mvpa.classification['algorithms']['svm']['C']}")
    print(f"  PCA components: {config.geometry.dimensionality_reduction['pca']['n_components']}")
    print(f"  Memory threshold: {config.memory.memory_mapping['threshold_gb']} GB")
    
    # Access visualization settings
    print("\nVisualization settings:")
    print(f"  Style: {config.visualization.get('style', 'default')}")
    print(f"  DPI: {config.visualization.get('figure', {}).get('dpi', 300)}")
    
    # Access SLURM configuration
    print("\nSLURM configuration:")
    print(f"  Job name: {config.slurm.job_name}")
    print(f"  CPUs per task: {config.slurm.cpus_per_task}")
    print(f"  Memory: {config.slurm.memory_gb} GB")
    
    print("\n✓ Advanced features work!")


def main():
    """Run all demonstrations"""
    print("CENTRALIZED CONFIGURATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows how to use the new centralized YAML configuration")
    print("system that replaces scattered configuration files.")
    print()
    
    try:
        demo_basic_usage()
        demo_roi_masks()
        demo_output_paths()
        demo_legacy_compatibility()
        demo_convenience_functions()
        demo_environment_overrides()
        demo_configuration_updates()
        demo_configuration_serialization()
        demo_pipeline_integration()
        demo_advanced_features()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY! ✅")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the centralized config.yaml file")
        print("2. Use config_migration.py to migrate your existing configs")
        print("3. Update your pipeline scripts to use the new config system")
        print("4. Enjoy cleaner, more maintainable configuration!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure config.yaml exists and is properly formatted.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 