#!/usr/bin/env python3
"""
Unified Configuration Loader for MVPA Analysis
==============================================
Features:
- Centralized YAML config
- Backward compatibility with legacy config classes
- Environment variable overrides
- Validation and error checking
- Dynamic updates

Usage:
    from config_loader import Config

    # Load default or custom config file
    config = Config()  # or Config('custom_config.yaml')

    # Access values:
    print(f"TR: {config.fmri.tr}")
    print(f"ROI masks: {config.roi_masks.core_rois}")
    print(f"N permutations: {config.mvpa.n_permutations}")

"""

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Set up logging system to capture and display informational messages.
# This configures the default logging level to INFO and creates a logger
# specific to this module (__name__) for organized log messages.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """
    Exception raised specifically for configuration-related errors.
    This helps to differentiate config errors from other exceptions,
    allowing for clearer error handling and debugging.
    """

    pass


@dataclass
class StudyConfig:
    """
    Configuration for study metadata.

    This class corresponds to the 'study' section in the YAML config file.
    Each attribute maps to a key in that section with sensible default values.

    Access example:
        config.study.name  # returns the study name
    """

    name: str = 'Delay Discounting MVPA Analysis'  # Study name
    description: str = 'Stanford Delay Discounting fMRI Analysis'  # Brief description
    version: str = '1.0'  # Version of this config/study
    pi: str = 'Russell Poldrack'  # Principal Investigator
    contact: str = 'your_email@stanford.edu'  # Contact email address


@dataclass
class PathsConfig:
    """
    Configuration for setting up key directory and file paths for data input,
    output, caching, logging, and ROI masks. Derived paths are computed in __post_init__.
    """

    data_root: str = '/oak/stanford/groups/russpold/data/uh2/aim1'
    fmriprep_dir: str = ''
    behavior_dir: str = ''
    output_root: str = ''
    output_dir: str = ''
    behavioral_output: str = 'behavioral_analysis'
    mvpa_output: str = 'mvpa_analysis'
    geometry_output: str = 'geometry_analysis'
    dd_geometry_output: str = 'dd_geometry_results'
    masks_dir: str = ''
    cache_dir: str = ''
    log_dir: str = './logs'

    task_name: str = 'discountFix'
    bold_func_glob: str = 'ses*/func/'
    bold_data_suffix: str = '*preproc_bold.nii.gz'
    bold_file_glob: str = field(init=False)

    core_mask_files: Dict[str, str] = field(default_factory=dict)
    optional_mask_files: Dict[str, str] = field(default_factory=dict)

    resolved_core_masks: Dict[str, str] = field(init=False)
    resolved_optional_masks: Dict[str, str] = field(init=False)
    output_paths: Dict[str, str] = field(init=False)

    def __post_init__(self):
        """Compute derived paths and resolved ROI file paths."""
        if not self.fmriprep_dir:
            self.fmriprep_dir = str(Path(self.data_root) / 'derivatives' / 'fmriprep')
        if not self.behavior_dir:
            self.behavior_dir = str(
                Path(self.data_root) / 'behavioral_data' / 'event_files'
            )
        if not self.output_root:
            self.output_root = str(
                Path(self.data_root) / 'derivatives' / 'mvpa_analysis'
            )
        if not self.output_dir:
            self.output_dir = str(Path(self.output_root) / 'delay_discounting_results')
        if not self.masks_dir:
            self.masks_dir = str(Path(self.data_root) / 'derivatives' / 'masks')
        if not self.cache_dir:
            self.cache_dir = str(Path(self.output_root) / 'analysis_cache')

        # Ensure functional glob ends with slash
        if not self.bold_func_glob.endswith('/'):
            self.bold_func_glob += '/'

        # Generate glob pattern for finding BOLD files
        self.bold_file_glob = (
            f'{self.bold_func_glob}*{self.task_name}{self.bold_data_suffix}'
        )

        # Resolve full paths to ROI masks
        self.resolved_core_masks = {
            roi: f'{self.masks_dir}/{fname}'
            for roi, fname in self.core_mask_files.items()
        }

        self.resolved_optional_masks = {
            roi: f'{self.masks_dir}/{fname}'
            for roi, fname in self.optional_mask_files.items()
        }
        # Construct output paths dictionary
        self.output_paths = {
            'main': self.output_dir,
            'behavioral': f'{self.output_dir}/{self.behavioral_output}',
            'mvpa': f'{self.output_dir}/{self.mvpa_output}',
            'geometry': f'{self.output_dir}/{self.geometry_output}',
            'dd_geometry': f'{self.output_dir}/{self.dd_geometry_output}',
            'cache': self.cache_dir,
            'logs': self.log_dir,
        }


@dataclass
class FMRIConfig:
    """
    Configuration for fMRI acquisition parameters and preprocessing settings.

    Attributes:
        tr (float): Repetition time (in seconds) for the BOLD fMRI acquisition.
        hemi_lag (int): Number of TRs to shift data for temporal alignment between hemispheres.
        smoothing_fwhm (float): Full width at half maximum (FWHM) for spatial smoothing (in mm).
        high_pass_filter (float): High-pass filter cutoff frequency (in Hz) to remove low-frequency drifts.
        standardize (bool): Whether to z-score features (e.g., voxels or ROIs) during preprocessing.
        detrend (bool): Whether to remove linear trends from time series data.
        confound_strategy (str): Strategy for selecting confound regressors.
        slice_time_ref (float): Reference slice timing as a fraction of TR
            (e.g., 0.5 refers to the middle of the TR).
    """

    tr: float = 0.68
    hemi_lag: int = 0
    smoothing_fwhm: float = 6.0
    high_pass_filter: float = 0.01
    standardize: bool = True
    detrend: bool = True
    confound_strategy: str = 'auto'
    slice_time_ref: float = 0.5


@dataclass
class ROIMasksConfig:
    """Configuration for ROI masks.

    Attributes:
        core_rois: List of core region names.
        optional_rois: List of optional region names.
        mask_files: Mapping from ROI name to mask filename.
    """

    core_rois: List[str] = field(default_factory=lambda: ['striatum', 'dlpfc', 'vmpfc'])
    optional_rois: List[str] = field(default_factory=list)
    mask_files: Dict[str, str] = field(default_factory=dict)

    def get_mask_path(self, roi_name: str, masks_dir: str) -> str:
        """Return full path to mask file for the given ROI."""
        if roi_name not in self.mask_files:
            raise ConfigError(f"ROI '{roi_name}' not found in mask_files")
        return os.path.join(masks_dir, self.mask_files[roi_name])

    def get_all_mask_paths(self, masks_dir: str) -> Dict[str, str]:
        """Return a dictionary of all ROI names to their full mask file paths."""
        return {
            roi: self.get_mask_path(roi, masks_dir) for roi in self.mask_files.keys()
        }


@dataclass
class BehavioralConfig:
    """Behavioral analysis configuration parameters.

    Attributes:
        min_accuracy: Minimum accuracy threshold to include trials.
        max_rt: Maximum allowable reaction time (seconds).
        min_rt: Minimum allowable reaction time (seconds).
        discount_model: Type of discounting model to fit (e.g., 'hyperbolic').
        fit_method: Method used for fitting (e.g., 'least_squares').
        variables: List of behavioral variable names to include.
    """

    min_accuracy: float = 0.6
    max_rt: float = 10.0
    min_rt: float = 0.1
    discount_model: str = 'hyperbolic'
    fit_method: str = 'least_squares'
    variables: List[str] = field(default_factory=list)


# These are likely not good defaults, but are placeholders for now.
@dataclass
class MVPAConfig:
    """MVPA analysis configuration"""

    cv_folds: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 42
    cv_strategy: str = 'stratified'
    n_permutations: int = 1000
    perm_random_state: int = 42
    alpha: float = 0.05
    multiple_comparisons: str = 'fdr_bh'
    classification: Dict[str, Any] = field(default_factory=dict)
    regression: Dict[str, Any] = field(default_factory=dict)
    feature_selection: Dict[str, Any] = field(default_factory=dict)
    data_preparation: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, List[str]] = field(default_factory=dict)
    quality_control: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryConfig:
    """Configuration for geometry-based analyses.

    Attributes:
        output_dir: Directory to save geometry analysis results.
        save_plots: Whether to save generated plots.
        plot_format: File format for saved plots ('png', 'pdf', 'svg', etc.).
        dpi: Resolution for saved plots.
        n_permutations: Number of permutations for permutation tests.
        random_state: Random seed for reproducibility.
        alpha: Significance threshold for tests.
        dimensionality_reduction: Parameters for dimensionality reduction methods.
        data_preparation: Settings for data preprocessing steps.
        delay_discounting: Specific parameters related to delay discounting models.
        comparisons: Settings for statistical comparisons.
        advanced_methods: Configuration for advanced geometry analysis techniques.
    """

    output_dir: str = './dd_geometry_results'
    save_plots: bool = True
    plot_format: str = 'png'
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
    """Configuration for parallel processing.

    Attributes:
        n_jobs: Number of parallel jobs (-1 uses all available CPUs).
        backend: Parallelization backend (e.g., 'loky', 'threading').
        subjects: Configuration for subject-level parallelization.
        rois: Configuration for ROI-level parallelization.
        nested_parallel: Settings for nested parallel processing.
        resource_management: Controls for managing CPU/memory resources.
    """

    n_jobs: int = -1
    backend: str = 'loky'
    subjects: Dict[str, Any] = field(default_factory=dict)
    rois: Dict[str, Any] = field(default_factory=dict)
    nested_parallel: Dict[str, Any] = field(default_factory=dict)
    resource_management: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachingConfig:
    """Configuration for caching system used to speed up repeated analyses.

    Attributes:
        enabled: Whether caching is enabled.
        cache_dir: Directory where cache files are stored.
        version: Cache version identifier to manage cache invalidation.
        management: Settings for cache management policies (e.g., max size, cleanup).
        precision: Settings related to numeric precision in caching (e.g., float rounding).
        levels: Dict specifying which caching levels are active (e.g., subject, run).
        invalidation: Dict controlling cache invalidation flags (e.g., force refresh).
    """

    enabled: bool = True
    cache_dir: str = 'analysis_cache'
    version: str = '1.0'
    management: Dict[str, Any] = field(default_factory=dict)
    precision: Dict[str, Any] = field(default_factory=dict)
    levels: Dict[str, bool] = field(default_factory=dict)
    invalidation: Dict[str, bool] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Configuration for logging system.

    Attributes:
        level: Logging verbosity level (e.g., 'DEBUG', 'INFO', 'WARNING').
        console: Enable logging output to the console.
        file: Enable logging output to a file.
        log_file: Path to the log file.
        max_file_size_mb: Maximum size of a log file before rotation.
        backup_count: Number of backup log files to keep.
        components: Dict mapping component names to specific log levels.
        performance: Flags to enable logging of performance metrics or profiling.
    """

    level: str = 'INFO'
    console: bool = True
    file: bool = True
    log_file: str = 'mvpa_analysis.log'
    max_file_size_mb: int = 100
    backup_count: int = 5
    components: Dict[str, str] = field(default_factory=dict)
    performance: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SlurmConfig:
    """Configuration for SLURM HPC job submission.

    Attributes:
        job_name: Name of the SLURM job.
        partition: SLURM partition/queue name.
        time: Maximum walltime for job in HH:MM:SS format.
        nodes: Number of nodes to request.
        ntasks: Number of tasks to run.
        cpus_per_task: CPUs allocated per task.
        memory_gb: Memory in gigabytes to request.
        output_dir: Directory to save SLURM output logs.
        mail_type: SLURM mail notification type (e.g., 'BEGIN', 'END', 'FAIL', 'ALL').
        mail_user: Email address for SLURM notifications.
        environment: Environment variables or modules to load for the job.
        auto_configure: Settings for automatic resource configuration or job tuning.
    """

    job_name: str = 'delay_discounting_mvpa'
    partition: str = 'normal'
    time: str = '08:00:00'
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 16
    memory_gb: int = 32
    output_dir: str = './logs'
    mail_type: str = 'ALL'
    mail_user: str = 'your_email@stanford.edu'
    environment: Dict[str, Any] = field(default_factory=dict)
    auto_configure: Dict[str, Any] = field(default_factory=dict)


class Config:
    """
    Unified configuration class that loads from YAML and provides
    structured access to all configuration parameters.
    """

    def __init__(self, config_file: str = 'config.yaml', validate: bool = True):
        """
        Initialize configuration by loading and parsing a YAML file.

        Parameters:
        -----------
        config_file : str
            Path to the YAML configuration file to load.
        validate : bool
            Whether to validate the configuration after loading.
        """
        self.config_file = config_file
        self._raw_config = self._load_yaml_config(config_file)
        self._parse_config()
        if validate:
            self._validate_config()

    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load YAML configuration file into a dictionary.

        Parameters:
        -----------
        config_file : str
            Path to the YAML configuration file.

        Returns:
        --------
        Dict[str, Any]
            The loaded configuration as a nested dictionary.

        Raises:
        -------
        ConfigError
            If the file does not exist or if there is a YAML parsing error.
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise ConfigError(f'Configuration file not found: {config_file}')
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                return data
        except yaml.YAMLError as e:
            raise ConfigError(f'Error parsing YAML configuration: {e}')

    def _parse_config(self):
        """Parse raw config into structured dataclasses dynamically"""
        config_map = {
            'study': StudyConfig,
            'paths': PathsConfig,
            'fmri': FMRIConfig,
            'roi_masks': ROIMasksConfig,
            'behavioral': BehavioralConfig,
            'mvpa': MVPAConfig,
            'geometry': GeometryConfig,
            'parallel': ParallelConfig,
            'caching': CachingConfig,
            'logging': LoggingConfig,
            'slurm': SlurmConfig,
        }

        for section, cls in config_map.items():
            setattr(self, section, cls(**self._raw_config.get(section, {})))

    def _validate_config(self):
        """Validate configuration parameters"""
        errors = []

        # --- fMRI validation ---
        if not isinstance(self.fmri.tr, (int, float)) or self.fmri.tr <= 0:
            errors.append('fmri.tr must be a positive number')
        if not isinstance(self.fmri.hemi_lag, int) or self.fmri.hemi_lag < 0:
            errors.append('fmri.hemi_lag must be a non-negative integer')

        # --- MVPA validation ---
        if not isinstance(self.mvpa.cv_folds, int) or self.mvpa.cv_folds < 2:
            errors.append('mvpa.cv_folds must be an integer >= 2')
        if (
            not isinstance(self.mvpa.n_permutations, int)
            or self.mvpa.n_permutations < 1
        ):
            errors.append('mvpa.n_permutations must be an integer >= 1')

        # --- Path validation ---
        if not self.paths.data_root:
            errors.append('paths.data_root is required')
        elif not os.path.exists(self.paths.data_root):
            errors.append(f'paths.data_root does not exist: {self.paths.data_root}')

        # --- ROI masks ---
        if not self.roi_masks.core_rois:
            errors.append('roi_masks.core_rois cannot be empty')

        if errors:
            raise ConfigError(
                'Configuration validation failed:\n  - ' + '\n  - '.join(errors)
            )

    def __repr__(self):
        """String representation of configuration"""
        return f"Config(file='{self.config_file}', study='{self.study.name}')"


# Convenience functions for backward compatibility
def load_config(config_file: str = 'config.yaml') -> Config:
    """Load configuration from file"""
    return Config(config_file)


if __name__ == '__main__':
    # Example usage
    print('Testing configuration loader...')

    try:
        # Load configuration
        config = Config()

        # Test structured access
        print(f'Study: {config.study.name}')
        print(f'TR: {config.fmri.tr}')
        print(f'Core ROIs: {config.roi_masks.core_rois}')
        print(f'N permutations: {config.mvpa.n_permutations}')

        print('✓ Configuration loader test passed!')

    except Exception as e:
        print(f'✗ Configuration loader test failed: {e}')
