#!/usr/bin/env python3
"""
Unified Configuration Loader for MVPA Analysis
==============================================

This module provides a centralized configuration system that replaces
the scattered configuration files with a single YAML-based configuration.

Features:
- Centralized YAML configuration
- Backward compatibility with existing config classes
- Environment variable overrides
- Validation and error checking
- Dynamic configuration updates

Usage:
    from config_loader import Config
    
    # Load default configuration
    config = Config()
    
    # Load custom configuration
    config = Config('custom_config.yaml')
    
    # Access configuration values
    print(f"TR: {config.fmri.tr}")
    print(f"ROI masks: {config.roi_masks.core_rois}")
    print(f"N permutations: {config.mvpa.n_permutations}")

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from copy import deepcopy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


@dataclass
class StudyConfig:
    """Study information configuration"""
    name: str = "Delay Discounting MVPA Analysis"
    description: str = "Stanford Delay Discounting fMRI Analysis"
    version: str = "1.0"
    pi: str = "Russell Poldrack"
    contact: str = "your_email@stanford.edu"


@dataclass
class PathsConfig:
    """Paths configuration"""
    data_root: str = "/oak/stanford/groups/russpold/data/uh2/aim1"
    fmriprep_dir: str = ""
    behavior_dir: str = ""
    output_root: str = ""
    output_dir: str = ""
    behavioral_output: str = "behavioral_analysis"
    mvpa_output: str = "mvpa_analysis"
    geometry_output: str = "geometry_analysis"
    dd_geometry_output: str = "dd_geometry_results"
    masks_dir: str = ""
    cache_dir: str = ""
    log_dir: str = "./logs"
    
    def __post_init__(self):
        """Auto-generate derived paths"""
        if not self.fmriprep_dir:
            self.fmriprep_dir = f"{self.data_root}/derivatives/fmriprep"
        if not self.behavior_dir:
            self.behavior_dir = f"{self.data_root}/behavioral_data/event_files"
        if not self.output_root:
            self.output_root = f"{self.data_root}/derivatives/mvpa_analysis"
        if not self.output_dir:
            self.output_dir = f"{self.output_root}/delay_discounting_results"
        if not self.masks_dir:
            self.masks_dir = f"{self.data_root}/derivatives/masks"
        if not self.cache_dir:
            self.cache_dir = f"{self.output_root}/analysis_cache"


@dataclass
class FMRIConfig:
    """fMRI acquisition and preprocessing configuration"""
    tr: float = 0.68
    hemi_lag: int = 0
    smoothing_fwhm: float = 6.0
    high_pass_filter: float = 0.01
    standardize: bool = True
    detrend: bool = True
    confound_strategy: str = "auto"
    slice_time_ref: float = 0.5


@dataclass
class ROIMasksConfig:
    """ROI masks configuration"""
    core_rois: List[str] = field(default_factory=lambda: ["striatum", "dlpfc", "vmpfc"])
    optional_rois: List[str] = field(default_factory=list)
    mask_files: Dict[str, str] = field(default_factory=dict)
    
    def get_mask_path(self, roi_name: str, masks_dir: str) -> str:
        """Get full path to mask file"""
        if roi_name not in self.mask_files:
            raise ConfigError(f"ROI '{roi_name}' not found in mask_files")
        return os.path.join(masks_dir, self.mask_files[roi_name])
    
    def get_all_mask_paths(self, masks_dir: str) -> Dict[str, str]:
        """Get all mask paths as dictionary"""
        return {roi: self.get_mask_path(roi, masks_dir) for roi in self.mask_files.keys()}


@dataclass
class BehavioralConfig:
    """Behavioral analysis configuration"""
    min_accuracy: float = 0.6
    max_rt: float = 10.0
    min_rt: float = 0.1
    discount_model: str = "hyperbolic"
    fit_method: str = "least_squares"
    variables: List[str] = field(default_factory=list)


@dataclass
class MVPAConfig:
    """MVPA analysis configuration"""
    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 42
    cv_strategy: str = "stratified"
    n_permutations: int = 1000
    perm_random_state: int = 42
    alpha: float = 0.05
    multiple_comparisons: str = "fdr_bh"
    classification: Dict[str, Any] = field(default_factory=dict)
    regression: Dict[str, Any] = field(default_factory=dict)
    feature_selection: Dict[str, Any] = field(default_factory=dict)
    data_preparation: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, List[str]] = field(default_factory=dict)
    quality_control: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryConfig:
    """Geometry analysis configuration"""
    output_dir: str = "./dd_geometry_results"
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    n_permutations: int = 1000
    random_state: int = 42
    alpha: float = 0.05
    dimensionality_reduction: Dict[str, Any] = field(default_factory=dict)
    data_preparation: Dict[str, Any] = field(default_factory=dict)
    delay_discounting: Dict[str, Any] = field(default_factory=dict)
    comparisons: Dict[str, Any] = field(default_factory=dict)
    advanced_methods: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelConfig:
    """Parallel processing configuration"""
    n_jobs: int = -1
    backend: str = "loky"
    subjects: Dict[str, Any] = field(default_factory=dict)
    rois: Dict[str, Any] = field(default_factory=dict)
    nested_parallel: Dict[str, Any] = field(default_factory=dict)
    resource_management: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Memory efficiency configuration"""
    memory_mapping: Dict[str, Any] = field(default_factory=dict)
    memory_buffer: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachingConfig:
    """Caching system configuration"""
    enabled: bool = True
    cache_dir: str = "analysis_cache"
    version: str = "1.0"
    management: Dict[str, Any] = field(default_factory=dict)
    precision: Dict[str, Any] = field(default_factory=dict)
    levels: Dict[str, bool] = field(default_factory=dict)
    invalidation: Dict[str, bool] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    console: bool = True
    file: bool = True
    log_file: str = "mvpa_analysis.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    components: Dict[str, str] = field(default_factory=dict)
    performance: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SlurmConfig:
    """SLURM/HPC configuration"""
    job_name: str = "delay_discounting_mvpa"
    partition: str = "normal"
    time: str = "08:00:00"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 16
    memory_gb: int = 32
    output_dir: str = "./logs"
    mail_type: str = "ALL"
    mail_user: str = "your_email@stanford.edu"
    environment: Dict[str, Any] = field(default_factory=dict)
    auto_configure: Dict[str, Any] = field(default_factory=dict)


class Config:
    """
    Unified configuration class that loads from YAML and provides
    structured access to all configuration parameters.
    """
    
    def __init__(self, config_file: str = "config.yaml", 
                 environment_overrides: bool = True,
                 validate: bool = True):
        """
        Initialize configuration
        
        Parameters:
        -----------
        config_file : str
            Path to YAML configuration file
        environment_overrides : bool
            Whether to allow environment variable overrides
        validate : bool
            Whether to validate configuration
        """
        self.config_file = config_file
        self.environment_overrides = environment_overrides
        
        # Load configuration
        self._raw_config = self._load_yaml_config(config_file)
        
        # Apply environment overrides
        if environment_overrides:
            self._apply_environment_overrides()
        
        # Parse into structured objects
        self._parse_config()
        
        # Validate configuration
        if validate:
            self._validate_config()
        
        logger.info(f"Configuration loaded from {config_file}")
    
    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML configuration: {e}")
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_prefix = "MVPA_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                # Convert environment variable to nested dict path
                # e.g., MVPA_FMRI_TR -> fmri.tr
                parts = config_key.split('_')
                
                # Navigate to nested dict
                current = self._raw_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set value (try to convert to appropriate type)
                try:
                    if value.lower() in ['true', 'false']:
                        current[parts[-1]] = value.lower() == 'true'
                    elif value.isdigit():
                        current[parts[-1]] = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        current[parts[-1]] = float(value)
                    else:
                        current[parts[-1]] = value
                except (ValueError, AttributeError):
                    current[parts[-1]] = value
                
                logger.info(f"Applied environment override: {key} = {value}")
    
    def _parse_config(self):
        """Parse raw configuration into structured objects"""
        # Study information
        self.study = StudyConfig(**self._raw_config.get('study', {}))
        
        # Paths
        self.paths = PathsConfig(**self._raw_config.get('paths', {}))
        
        # fMRI parameters
        self.fmri = FMRIConfig(**self._raw_config.get('fmri', {}))
        
        # ROI masks
        self.roi_masks = ROIMasksConfig(**self._raw_config.get('roi_masks', {}))
        
        # Behavioral analysis
        self.behavioral = BehavioralConfig(**self._raw_config.get('behavioral', {}))
        
        # MVPA analysis
        self.mvpa = MVPAConfig(**self._raw_config.get('mvpa', {}))
        
        # Geometry analysis
        self.geometry = GeometryConfig(**self._raw_config.get('geometry', {}))
        
        # Parallel processing
        self.parallel = ParallelConfig(**self._raw_config.get('parallel', {}))
        
        # Memory efficiency
        self.memory = MemoryConfig(**self._raw_config.get('memory', {}))
        
        # Caching
        self.caching = CachingConfig(**self._raw_config.get('caching', {}))
        
        # Logging
        self.logging = LoggingConfig(**self._raw_config.get('logging', {}))
        
        # SLURM
        self.slurm = SlurmConfig(**self._raw_config.get('slurm', {}))
        
        # Store visualization, validation, and advanced settings as dicts
        self.visualization = self._raw_config.get('visualization', {})
        self.validation = self._raw_config.get('validation', {})
        self.advanced = self._raw_config.get('advanced', {})
    
    def _validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Validate fMRI parameters
        if self.fmri.tr <= 0:
            errors.append("fMRI TR must be positive")
        if self.fmri.hemi_lag < 0:
            errors.append("fMRI hemi_lag must be non-negative")
        
        # Validate MVPA parameters
        if self.mvpa.cv_folds < 2:
            errors.append("MVPA cv_folds must be >= 2")
        if self.mvpa.n_permutations < 1:
            errors.append("MVPA n_permutations must be >= 1")
        
        # Validate paths
        if not self.paths.data_root:
            errors.append("paths.data_root is required")
        
        # Validate ROI masks
        if not self.roi_masks.core_rois:
            errors.append("roi_masks.core_rois cannot be empty")
        
        if errors:
            raise ConfigError(f"Configuration validation failed: {', '.join(errors)}")
    
    def get_roi_mask_paths(self) -> Dict[str, str]:
        """Get all ROI mask paths"""
        return self.roi_masks.get_all_mask_paths(self.paths.masks_dir)
    
    def get_core_roi_mask_paths(self) -> Dict[str, str]:
        """Get core ROI mask paths only"""
        all_masks = self.get_roi_mask_paths()
        return {roi: all_masks[roi] for roi in self.roi_masks.core_rois if roi in all_masks}
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get all output paths"""
        return {
            'main': self.paths.output_dir,
            'behavioral': os.path.join(self.paths.output_dir, self.paths.behavioral_output),
            'mvpa': os.path.join(self.paths.output_dir, self.paths.mvpa_output),
            'geometry': os.path.join(self.paths.output_dir, self.paths.geometry_output),
            'dd_geometry': os.path.join(self.paths.output_dir, self.paths.dd_geometry_output),
            'cache': self.paths.cache_dir,
            'logs': self.paths.log_dir
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return deepcopy(self._raw_config)
    
    def to_json(self, file_path: str = None) -> str:
        """Convert configuration to JSON"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
        return json_str
    
    def save_yaml(self, file_path: str):
        """Save configuration to YAML file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._raw_config = update_nested_dict(self._raw_config, updates)
        self._parse_config()
    
    def get_legacy_oak_config(self):
        """Get legacy OAKConfig-compatible object for backward compatibility"""
        from types import SimpleNamespace
        
        # Create a namespace object with all the legacy attributes
        legacy_config = SimpleNamespace()
        
        # Basic paths
        legacy_config.DATA_ROOT = self.paths.data_root
        legacy_config.FMRIPREP_DIR = self.paths.fmriprep_dir
        legacy_config.BEHAVIOR_DIR = self.paths.behavior_dir
        legacy_config.OUTPUT_DIR = self.paths.output_dir
        legacy_config.BEHAVIOR_OUTPUT = os.path.join(self.paths.output_dir, self.paths.behavioral_output)
        legacy_config.MVPA_OUTPUT = os.path.join(self.paths.output_dir, self.paths.mvpa_output)
        legacy_config.GEOMETRY_OUTPUT = os.path.join(self.paths.output_dir, self.paths.geometry_output)
        legacy_config.MASKS_DIR = self.paths.masks_dir
        
        # fMRI parameters
        legacy_config.TR = self.fmri.tr
        legacy_config.HEMI_LAG = self.fmri.hemi_lag
        
        # ROI masks
        legacy_config.ROI_MASKS = self.get_roi_mask_paths()
        legacy_config.CORE_ROI_MASKS = self.roi_masks.core_rois
        legacy_config.OPTIONAL_ROI_MASKS = self.roi_masks.optional_rois
        
        # MVPA parameters
        legacy_config.N_JOBS = self.parallel.n_jobs
        legacy_config.CV_FOLDS = self.mvpa.cv_folds
        legacy_config.N_PERMUTATIONS = self.mvpa.n_permutations
        
        # Quality control
        legacy_config.MIN_ACCURACY = self.behavioral.min_accuracy
        legacy_config.MAX_RT = self.behavioral.max_rt
        
        return legacy_config
    
    def get_legacy_mvpa_config(self):
        """Get legacy MVPAConfig-compatible object for backward compatibility"""
        from types import SimpleNamespace
        
        legacy_config = SimpleNamespace()
        
        # Cross-validation
        legacy_config.CV_FOLDS = self.mvpa.cv_folds
        legacy_config.CV_SHUFFLE = self.mvpa.cv_shuffle
        legacy_config.CV_RANDOM_STATE = self.mvpa.cv_random_state
        
        # Permutation testing
        legacy_config.N_PERMUTATIONS = self.mvpa.n_permutations
        legacy_config.PERM_RANDOM_STATE = self.mvpa.perm_random_state
        
        # Algorithms
        legacy_config.DEFAULT_CLASSIFIER = self.mvpa.classification.get('default_algorithm', 'svm')
        legacy_config.DEFAULT_REGRESSOR = self.mvpa.regression.get('default_algorithm', 'ridge')
        
        # SVM parameters
        svm_params = self.mvpa.classification.get('algorithms', {}).get('svm', {})
        legacy_config.SVM_C = svm_params.get('C', 1.0)
        legacy_config.SVM_KERNEL = svm_params.get('kernel', 'linear')
        
        # Ridge parameters
        ridge_params = self.mvpa.regression.get('algorithms', {}).get('ridge', {})
        legacy_config.RIDGE_ALPHA = ridge_params.get('alpha', 1.0)
        
        # Feature selection
        legacy_config.FEATURE_SELECTION = self.mvpa.feature_selection.get('enabled', False)
        legacy_config.N_FEATURES = self.mvpa.feature_selection.get('n_features', 1000)
        
        # Preprocessing
        legacy_config.STANDARDIZE = self.mvpa.data_preparation.get('standardize', True)
        
        # Parallel processing
        legacy_config.N_JOBS = self.parallel.n_jobs
        
        # Quality control
        legacy_config.MIN_SAMPLES_PER_CLASS = self.mvpa.quality_control.get('min_samples_per_class', 5)
        legacy_config.MIN_VARIANCE_THRESHOLD = self.mvpa.data_preparation.get('variance_threshold', 1e-8)
        
        return legacy_config
    
    def __repr__(self):
        """String representation of configuration"""
        return f"Config(file='{self.config_file}', study='{self.study.name}')"


# Convenience functions for backward compatibility
def load_config(config_file: str = "config.yaml") -> Config:
    """Load configuration from file"""
    return Config(config_file)


def get_oak_config(config_file: str = "config.yaml"):
    """Get OAKConfig-compatible object"""
    config = Config(config_file)
    return config.get_legacy_oak_config()


def get_mvpa_config(config_file: str = "config.yaml"):
    """Get MVPAConfig-compatible object"""
    config = Config(config_file)
    return config.get_legacy_mvpa_config()


# Environment-based configuration selection
def get_config_file() -> str:
    """Get configuration file based on environment"""
    # Check for environment-specific config file
    env_config = os.environ.get('MVPA_CONFIG_FILE')
    if env_config and os.path.exists(env_config):
        return env_config
    
    # Check for common config file names
    config_files = [
        'config.yaml',
        'mvpa_config.yaml',
        'analysis_config.yaml',
        'config.yml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            return config_file
    
    # If no config file found, use default
    return 'config.yaml'


# Main configuration instance (singleton pattern)
_main_config = None

def get_main_config() -> Config:
    """Get main configuration instance (singleton)"""
    global _main_config
    if _main_config is None:
        config_file = get_config_file()
        _main_config = Config(config_file)
    return _main_config


if __name__ == "__main__":
    # Example usage
    print("Testing configuration loader...")
    
    try:
        # Load configuration
        config = Config()
        
        # Test structured access
        print(f"Study: {config.study.name}")
        print(f"TR: {config.fmri.tr}")
        print(f"Core ROIs: {config.roi_masks.core_rois}")
        print(f"N permutations: {config.mvpa.n_permutations}")
        
        # Test legacy compatibility
        oak_config = config.get_legacy_oak_config()
        print(f"Legacy TR: {oak_config.TR}")
        print(f"Legacy ROI masks: {list(oak_config.ROI_MASKS.keys())}")
        
        print("✓ Configuration loader test passed!")
        
    except Exception as e:
        print(f"✗ Configuration loader test failed: {e}") 