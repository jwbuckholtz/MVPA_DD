#!/usr/bin/env python3
"""
Configuration Migration Utility
==============================

This utility helps migrate from scattered configuration files to the 
centralized YAML configuration system.

Features:
- Migrate from existing config files (oak_storage_config.py, dd_geometry_config.json, etc.)
- Validate migration results
- Create backup of old configurations
- Generate migration reports
- Support for custom migration rules

Usage:
    python config_migration.py --migrate-all
    python config_migration.py --migrate-from oak_storage_config.py
    python config_migration.py --validate-migration
    python config_migration.py --create-backup

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import json
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import importlib.util
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigMigrationError(Exception):
    """Custom exception for configuration migration errors"""
    pass


class ConfigMigrator:
    """Main configuration migration utility"""
    
    def __init__(self, output_file: str = "config.yaml", backup_dir: str = "config_backup"):
        self.output_file = output_file
        self.backup_dir = backup_dir
        self.migration_report = {
            'timestamp': datetime.now().isoformat(),
            'source_files': [],
            'migrated_values': {},
            'warnings': [],
            'errors': []
        }
        
        # Create backup directory
        Path(self.backup_dir).mkdir(exist_ok=True)
    
    def migrate_all(self) -> Dict[str, Any]:
        """Migrate all known configuration files"""
        logger.info("Starting complete configuration migration...")
        
        # Initialize merged configuration
        merged_config = self._get_default_config()
        
        # Migrate from each known config file
        config_files = [
            ('oak_storage_config.py', self._migrate_oak_storage_config),
            ('dd_geometry_config.json', self._migrate_dd_geometry_config),
            ('mvpa_utils.py', self._migrate_mvpa_utils_config),
        ]
        
        for config_file, migrate_func in config_files:
            if Path(config_file).exists():
                try:
                    logger.info(f"Migrating from {config_file}...")
                    partial_config = migrate_func(config_file)
                    merged_config = self._merge_configs(merged_config, partial_config)
                    self.migration_report['source_files'].append(config_file)
                    logger.info(f"✓ Successfully migrated {config_file}")
                except Exception as e:
                    error_msg = f"Failed to migrate {config_file}: {str(e)}"
                    logger.error(error_msg)
                    self.migration_report['errors'].append(error_msg)
            else:
                logger.info(f"Skipping {config_file} (not found)")
        
        # Save merged configuration
        self._save_config(merged_config)
        
        # Generate migration report
        self._generate_migration_report()
        
        logger.info(f"Configuration migration complete. Output: {self.output_file}")
        return merged_config
    
    def migrate_from_file(self, config_file: str) -> Dict[str, Any]:
        """Migrate from a specific configuration file"""
        logger.info(f"Migrating from {config_file}...")
        
        if not Path(config_file).exists():
            raise ConfigMigrationError(f"Configuration file not found: {config_file}")
        
        # Determine migration method based on file extension
        if config_file.endswith('.py'):
            if 'oak_storage_config' in config_file:
                migrated_config = self._migrate_oak_storage_config(config_file)
            elif 'mvpa_utils' in config_file:
                migrated_config = self._migrate_mvpa_utils_config(config_file)
            else:
                migrated_config = self._migrate_generic_python_config(config_file)
        elif config_file.endswith('.json'):
            migrated_config = self._migrate_json_config(config_file)
        elif config_file.endswith(('.yaml', '.yml')):
            migrated_config = self._migrate_yaml_config(config_file)
        else:
            raise ConfigMigrationError(f"Unsupported configuration file format: {config_file}")
        
        self.migration_report['source_files'].append(config_file)
        self._save_config(migrated_config)
        self._generate_migration_report()
        
        logger.info(f"Migration from {config_file} complete. Output: {self.output_file}")
        return migrated_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure"""
        return {
            'study': {
                'name': 'Delay Discounting MVPA Analysis',
                'description': 'Stanford Delay Discounting fMRI Analysis',
                'version': '1.0',
                'pi': 'Russell Poldrack',
                'contact': 'your_email@stanford.edu'
            },
            'paths': {
                'data_root': '/oak/stanford/groups/russpold/data/uh2/aim1',
                'fmriprep_dir': '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/fmriprep',
                'behavior_dir': '/oak/stanford/groups/russpold/data/uh2/aim1/behavioral_data/event_files',
                'output_root': '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis',
                'output_dir': '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/delay_discounting_results',
                'behavioral_output': 'behavioral_analysis',
                'mvpa_output': 'mvpa_analysis',
                'geometry_output': 'geometry_analysis',
                'dd_geometry_output': 'dd_geometry_results',
                'masks_dir': '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/masks',
                'cache_dir': '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/mvpa_analysis/analysis_cache',
                'log_dir': './logs'
            },
            'fmri': {
                'tr': 0.68,
                'hemi_lag': 0,
                'smoothing_fwhm': 6.0,
                'high_pass_filter': 0.01,
                'standardize': True,
                'detrend': True,
                'confound_strategy': 'auto',
                'slice_time_ref': 0.5
            },
            'roi_masks': {
                'core_rois': ['striatum', 'dlpfc', 'vmpfc'],
                'optional_rois': ['left_striatum', 'right_striatum', 'left_dlpfc', 'right_dlpfc', 'acc', 'ofc'],
                'mask_files': {
                    'striatum': 'striatum_mask.nii.gz',
                    'dlpfc': 'dlpfc_mask.nii.gz',
                    'vmpfc': 'vmpfc_mask.nii.gz',
                    'left_striatum': 'left_striatum_mask.nii.gz',
                    'right_striatum': 'right_striatum_mask.nii.gz',
                    'left_dlpfc': 'left_dlpfc_mask.nii.gz',
                    'right_dlpfc': 'right_dlpfc_mask.nii.gz',
                    'acc': 'acc_mask.nii.gz',
                    'ofc': 'ofc_mask.nii.gz'
                }
            },
            'behavioral': {
                'min_accuracy': 0.6,
                'max_rt': 10.0,
                'min_rt': 0.1,
                'discount_model': 'hyperbolic',
                'fit_method': 'least_squares',
                'variables': ['choice', 'choice_binary', 'rt', 'onset', 'delay_days', 'amount_small', 'amount_large', 'sv_chosen', 'sv_unchosen', 'sv_diff', 'sv_sum', 'later_delay', 'discount_rate', 'model_fit']
            },
            'mvpa': {
                'cv_folds': 5,
                'cv_shuffle': True,
                'cv_random_state': 42,
                'cv_strategy': 'stratified',
                'n_permutations': 1000,
                'perm_random_state': 42,
                'alpha': 0.05,
                'multiple_comparisons': 'fdr_bh',
                'classification': {
                    'default_algorithm': 'svm',
                    'algorithms': {
                        'svm': {'C': 1.0, 'kernel': 'linear', 'class_weight': 'balanced'},
                        'logistic': {'C': 1.0, 'class_weight': 'balanced', 'solver': 'liblinear'},
                        'random_forest': {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'}
                    }
                },
                'regression': {
                    'default_algorithm': 'ridge',
                    'algorithms': {
                        'ridge': {'alpha': 1.0, 'normalize': False},
                        'lasso': {'alpha': 1.0, 'normalize': False},
                        'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5, 'normalize': False}
                    }
                },
                'feature_selection': {
                    'enabled': False,
                    'method': 'univariate',
                    'n_features': 1000,
                    'score_func': 'f_classif'
                },
                'data_preparation': {
                    'standardize': True,
                    'remove_mean': True,
                    'variance_threshold': 1e-8
                },
                'targets': {
                    'classification': ['choice_binary'],
                    'regression': ['sv_diff', 'sv_sum', 'sv_chosen', 'sv_unchosen', 'later_delay', 'discount_rate']
                },
                'quality_control': {
                    'min_samples_per_class': 5,
                    'min_trials_per_subject': 20,
                    'max_missing_data': 0.1
                }
            },
            'geometry': {
                'output_dir': './dd_geometry_results',
                'save_plots': True,
                'plot_format': 'png',
                'dpi': 300,
                'n_permutations': 1000,
                'random_state': 42,
                'alpha': 0.05,
                'dimensionality_reduction': {
                    'pca': {'n_components': 15, 'whiten': False},
                    'mds': {'n_components': 8, 'metric': True, 'dissimilarity': 'euclidean'},
                    'tsne': {'n_components': 3, 'perplexity': 30, 'learning_rate': 200},
                    'isomap': {'n_components': 5, 'n_neighbors': 10}
                },
                'data_preparation': {
                    'standardize_data': True,
                    'remove_mean': True
                },
                'delay_discounting': {
                    'delay_short_threshold': 7,
                    'delay_long_threshold': 30,
                    'value_percentile_split': 50,
                    'value_diff_percentile': 67
                }
            },
            'parallel': {
                'n_jobs': -1,
                'backend': 'loky',
                'subjects': {'enabled': True, 'n_jobs': 8, 'chunk_size': 1},
                'rois': {'enabled': True, 'n_jobs': 4, 'chunk_size': 1},
                'nested_parallel': {'enabled': True, 'max_workers': 16},
                'resource_management': {'memory_limit_gb': 64, 'cpu_limit': 16, 'timeout_minutes': 120}
            },
            'memory': {
                'memory_mapping': {'enabled': True, 'threshold_gb': 1.0, 'force_memmap': False},
                'memory_buffer': {'available_memory_buffer': 0.2, 'max_memory_per_process_gb': 8.0},
                'shared_memory': {'enabled': True, 'temp_dir': '/tmp/mvpa_memmap'},
                'monitoring': {'enabled': True, 'log_usage': True, 'warning_threshold': 0.8}
            },
            'caching': {
                'enabled': True,
                'cache_dir': 'analysis_cache',
                'version': '1.0',
                'management': {'max_cache_size_gb': 50, 'cleanup_threshold': 0.8, 'auto_cleanup': True},
                'precision': {'float_precision': 6, 'hash_precision': 16},
                'levels': {'behavioral_analysis': True, 'beta_extraction': True, 'mvpa_decoding': True, 'geometry_analysis': True},
                'invalidation': {'on_config_change': True, 'on_code_change': True, 'on_data_change': True}
            },
            'logging': {
                'level': 'INFO',
                'console': True,
                'file': True,
                'log_file': 'mvpa_analysis.log',
                'max_file_size_mb': 100,
                'backup_count': 5,
                'components': {'behavioral': 'INFO', 'mvpa': 'INFO', 'geometry': 'INFO', 'parallel': 'INFO', 'memory': 'INFO', 'caching': 'INFO'},
                'performance': {'enabled': True, 'log_memory_usage': True, 'log_timing': True, 'log_progress': True}
            },
            'slurm': {
                'job_name': 'delay_discounting_mvpa',
                'partition': 'normal',
                'time': '08:00:00',
                'nodes': 1,
                'ntasks': 1,
                'cpus_per_task': 16,
                'memory_gb': 32,
                'output_dir': './logs',
                'mail_type': 'ALL',
                'mail_user': 'your_email@stanford.edu',
                'environment': {'omp_num_threads': 'auto', 'pythonpath': '.'},
                'auto_configure': {'enabled': True, 'memory_multiplier': 0.8, 'cpu_multiplier': 1.0}
            }
        }
    
    def _migrate_oak_storage_config(self, config_file: str) -> Dict[str, Any]:
        """Migrate from oak_storage_config.py"""
        logger.info(f"Migrating OAK storage configuration from {config_file}")
        
        # Create backup
        self._create_backup(config_file)
        
        # Import the module
        spec = importlib.util.spec_from_file_location("oak_storage_config", config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract configuration values
        config = self._get_default_config()
        
        if hasattr(module, 'OAKConfig'):
            oak_config = module.OAKConfig()
            
            # Migrate paths
            if hasattr(oak_config, 'DATA_ROOT'):
                config['paths']['data_root'] = oak_config.DATA_ROOT
            if hasattr(oak_config, 'FMRIPREP_DIR'):
                config['paths']['fmriprep_dir'] = oak_config.FMRIPREP_DIR
            if hasattr(oak_config, 'BEHAVIOR_DIR'):
                config['paths']['behavior_dir'] = oak_config.BEHAVIOR_DIR
            if hasattr(oak_config, 'OUTPUT_DIR'):
                config['paths']['output_dir'] = oak_config.OUTPUT_DIR
            if hasattr(oak_config, 'MASKS_DIR'):
                config['paths']['masks_dir'] = oak_config.MASKS_DIR
            
            # Migrate fMRI parameters
            if hasattr(oak_config, 'TR'):
                config['fmri']['tr'] = oak_config.TR
            if hasattr(oak_config, 'HEMI_LAG'):
                config['fmri']['hemi_lag'] = oak_config.HEMI_LAG
            
            # Migrate ROI masks
            if hasattr(oak_config, 'ROI_MASKS'):
                roi_masks = oak_config.ROI_MASKS
                config['roi_masks']['mask_files'] = {}
                for roi, mask_path in roi_masks.items():
                    config['roi_masks']['mask_files'][roi] = os.path.basename(mask_path)
            
            if hasattr(oak_config, 'CORE_ROI_MASKS'):
                config['roi_masks']['core_rois'] = oak_config.CORE_ROI_MASKS
            if hasattr(oak_config, 'OPTIONAL_ROI_MASKS'):
                config['roi_masks']['optional_rois'] = oak_config.OPTIONAL_ROI_MASKS
            
            # Migrate MVPA parameters
            if hasattr(oak_config, 'CV_FOLDS'):
                config['mvpa']['cv_folds'] = oak_config.CV_FOLDS
            if hasattr(oak_config, 'N_PERMUTATIONS'):
                config['mvpa']['n_permutations'] = oak_config.N_PERMUTATIONS
            if hasattr(oak_config, 'N_JOBS'):
                config['parallel']['n_jobs'] = oak_config.N_JOBS
            
            # Migrate behavioral parameters
            if hasattr(oak_config, 'MIN_ACCURACY'):
                config['behavioral']['min_accuracy'] = oak_config.MIN_ACCURACY
            if hasattr(oak_config, 'MAX_RT'):
                config['behavioral']['max_rt'] = oak_config.MAX_RT
            
            self.migration_report['migrated_values']['oak_storage_config'] = {
                'tr': config['fmri']['tr'],
                'hemi_lag': config['fmri']['hemi_lag'],
                'roi_masks': len(config['roi_masks']['mask_files']),
                'cv_folds': config['mvpa']['cv_folds'],
                'n_permutations': config['mvpa']['n_permutations']
            }
        
        return config
    
    def _migrate_dd_geometry_config(self, config_file: str) -> Dict[str, Any]:
        """Migrate from dd_geometry_config.json"""
        logger.info(f"Migrating DD geometry configuration from {config_file}")
        
        # Create backup
        self._create_backup(config_file)
        
        # Load JSON configuration
        with open(config_file, 'r') as f:
            dd_config = json.load(f)
        
        # Start with default configuration
        config = self._get_default_config()
        
        # Migrate geometry-specific parameters
        if 'output_dir' in dd_config:
            config['geometry']['output_dir'] = dd_config['output_dir']
        if 'n_permutations' in dd_config:
            config['geometry']['n_permutations'] = dd_config['n_permutations']
        if 'random_state' in dd_config:
            config['geometry']['random_state'] = dd_config['random_state']
        if 'alpha' in dd_config:
            config['geometry']['alpha'] = dd_config['alpha']
        if 'n_components_pca' in dd_config:
            config['geometry']['dimensionality_reduction']['pca']['n_components'] = dd_config['n_components_pca']
        if 'n_components_mds' in dd_config:
            config['geometry']['dimensionality_reduction']['mds']['n_components'] = dd_config['n_components_mds']
        if 'standardize_data' in dd_config:
            config['geometry']['data_preparation']['standardize_data'] = dd_config['standardize_data']
        if 'plot_format' in dd_config:
            config['geometry']['plot_format'] = dd_config['plot_format']
        if 'dpi' in dd_config:
            config['geometry']['dpi'] = dd_config['dpi']
        if 'delay_short_threshold' in dd_config:
            config['geometry']['delay_discounting']['delay_short_threshold'] = dd_config['delay_short_threshold']
        if 'delay_long_threshold' in dd_config:
            config['geometry']['delay_discounting']['delay_long_threshold'] = dd_config['delay_long_threshold']
        
        # Migrate comparison descriptions
        if 'comparison_descriptions' in dd_config:
            config['geometry']['comparisons'] = {}
            for comp_name, description in dd_config['comparison_descriptions'].items():
                config['geometry']['comparisons'][comp_name] = {
                    'description': description,
                    'enabled': True
                }
        
        self.migration_report['migrated_values']['dd_geometry_config'] = {
            'output_dir': config['geometry']['output_dir'],
            'n_permutations': config['geometry']['n_permutations'],
            'pca_components': config['geometry']['dimensionality_reduction']['pca']['n_components'],
            'mds_components': config['geometry']['dimensionality_reduction']['mds']['n_components'],
            'comparisons': len(config['geometry']['comparisons'])
        }
        
        return config
    
    def _migrate_mvpa_utils_config(self, config_file: str) -> Dict[str, Any]:
        """Migrate from mvpa_utils.py MVPAConfig class"""
        logger.info(f"Migrating MVPA utils configuration from {config_file}")
        
        # Create backup
        self._create_backup(config_file)
        
        # Import the module
        spec = importlib.util.spec_from_file_location("mvpa_utils", config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Start with default configuration
        config = self._get_default_config()
        
        if hasattr(module, 'MVPAConfig'):
            mvpa_config = module.MVPAConfig()
            
            # Migrate MVPA parameters
            if hasattr(mvpa_config, 'CV_FOLDS'):
                config['mvpa']['cv_folds'] = mvpa_config.CV_FOLDS
            if hasattr(mvpa_config, 'CV_SHUFFLE'):
                config['mvpa']['cv_shuffle'] = mvpa_config.CV_SHUFFLE
            if hasattr(mvpa_config, 'CV_RANDOM_STATE'):
                config['mvpa']['cv_random_state'] = mvpa_config.CV_RANDOM_STATE
            if hasattr(mvpa_config, 'N_PERMUTATIONS'):
                config['mvpa']['n_permutations'] = mvpa_config.N_PERMUTATIONS
            if hasattr(mvpa_config, 'PERM_RANDOM_STATE'):
                config['mvpa']['perm_random_state'] = mvpa_config.PERM_RANDOM_STATE
            if hasattr(mvpa_config, 'DEFAULT_CLASSIFIER'):
                config['mvpa']['classification']['default_algorithm'] = mvpa_config.DEFAULT_CLASSIFIER
            if hasattr(mvpa_config, 'DEFAULT_REGRESSOR'):
                config['mvpa']['regression']['default_algorithm'] = mvpa_config.DEFAULT_REGRESSOR
            if hasattr(mvpa_config, 'SVM_C'):
                config['mvpa']['classification']['algorithms']['svm']['C'] = mvpa_config.SVM_C
            if hasattr(mvpa_config, 'SVM_KERNEL'):
                config['mvpa']['classification']['algorithms']['svm']['kernel'] = mvpa_config.SVM_KERNEL
            if hasattr(mvpa_config, 'RIDGE_ALPHA'):
                config['mvpa']['regression']['algorithms']['ridge']['alpha'] = mvpa_config.RIDGE_ALPHA
            if hasattr(mvpa_config, 'FEATURE_SELECTION'):
                config['mvpa']['feature_selection']['enabled'] = mvpa_config.FEATURE_SELECTION
            if hasattr(mvpa_config, 'N_FEATURES'):
                config['mvpa']['feature_selection']['n_features'] = mvpa_config.N_FEATURES
            if hasattr(mvpa_config, 'STANDARDIZE'):
                config['mvpa']['data_preparation']['standardize'] = mvpa_config.STANDARDIZE
            if hasattr(mvpa_config, 'N_JOBS'):
                config['parallel']['n_jobs'] = mvpa_config.N_JOBS
            if hasattr(mvpa_config, 'MIN_SAMPLES_PER_CLASS'):
                config['mvpa']['quality_control']['min_samples_per_class'] = mvpa_config.MIN_SAMPLES_PER_CLASS
            if hasattr(mvpa_config, 'MIN_VARIANCE_THRESHOLD'):
                config['mvpa']['data_preparation']['variance_threshold'] = mvpa_config.MIN_VARIANCE_THRESHOLD
            
            self.migration_report['migrated_values']['mvpa_utils'] = {
                'cv_folds': config['mvpa']['cv_folds'],
                'n_permutations': config['mvpa']['n_permutations'],
                'default_classifier': config['mvpa']['classification']['default_algorithm'],
                'default_regressor': config['mvpa']['regression']['default_algorithm'],
                'svm_c': config['mvpa']['classification']['algorithms']['svm']['C'],
                'ridge_alpha': config['mvpa']['regression']['algorithms']['ridge']['alpha']
            }
        
        return config
    
    def _migrate_json_config(self, config_file: str) -> Dict[str, Any]:
        """Migrate from a generic JSON configuration file"""
        logger.info(f"Migrating JSON configuration from {config_file}")
        
        # Create backup
        self._create_backup(config_file)
        
        # Load JSON configuration
        with open(config_file, 'r') as f:
            json_config = json.load(f)
        
        # Start with default configuration and merge
        config = self._get_default_config()
        config = self._merge_configs(config, json_config)
        
        return config
    
    def _migrate_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Migrate from a generic YAML configuration file"""
        logger.info(f"Migrating YAML configuration from {config_file}")
        
        # Create backup
        self._create_backup(config_file)
        
        # Load YAML configuration
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Start with default configuration and merge
        config = self._get_default_config()
        config = self._merge_configs(config, yaml_config)
        
        return config
    
    def _migrate_generic_python_config(self, config_file: str) -> Dict[str, Any]:
        """Migrate from a generic Python configuration file"""
        logger.info(f"Migrating generic Python configuration from {config_file}")
        
        # Create backup
        self._create_backup(config_file)
        
        # Import the module
        spec = importlib.util.spec_from_file_location("generic_config", config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Start with default configuration
        config = self._get_default_config()
        
        # Extract configuration values from module attributes
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr_value = getattr(module, attr_name)
                if not callable(attr_value):
                    # Map common attribute names to config structure
                    if attr_name.upper() == 'TR':
                        config['fmri']['tr'] = attr_value
                    elif attr_name.upper() == 'HEMI_LAG':
                        config['fmri']['hemi_lag'] = attr_value
                    elif attr_name.upper() == 'CV_FOLDS':
                        config['mvpa']['cv_folds'] = attr_value
                    elif attr_name.upper() == 'N_PERMUTATIONS':
                        config['mvpa']['n_permutations'] = attr_value
                    # Add more mappings as needed
        
        return config
    
    def _merge_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries"""
        def merge_dicts(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge_dicts(d1[k], v)
                else:
                    d1[k] = v
            return d1
        
        return merge_dicts(config1.copy(), config2)
    
    def _create_backup(self, config_file: str):
        """Create backup of existing configuration file"""
        if Path(config_file).exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = Path(self.backup_dir) / f"{Path(config_file).stem}_{timestamp}{Path(config_file).suffix}"
            shutil.copy2(config_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to YAML file"""
        with open(self.output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        logger.info(f"Configuration saved to {self.output_file}")
    
    def _generate_migration_report(self):
        """Generate migration report"""
        report_file = Path(self.output_file).with_suffix('.migration_report.json')
        with open(report_file, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        logger.info(f"Migration report saved to {report_file}")
    
    def validate_migration(self) -> bool:
        """Validate the migrated configuration"""
        logger.info("Validating migrated configuration...")
        
        try:
            # Import the config loader and test loading
            from config_loader import Config
            config = Config(self.output_file)
            
            # Test legacy compatibility
            oak_config = config.get_legacy_oak_config()
            mvpa_config = config.get_legacy_mvpa_config()
            
            # Validate required attributes
            required_oak_attrs = ['DATA_ROOT', 'FMRIPREP_DIR', 'BEHAVIOR_DIR', 'OUTPUT_DIR', 'TR', 'HEMI_LAG', 'ROI_MASKS', 'CV_FOLDS', 'N_PERMUTATIONS']
            required_mvpa_attrs = ['CV_FOLDS', 'N_PERMUTATIONS', 'DEFAULT_CLASSIFIER', 'DEFAULT_REGRESSOR', 'SVM_C', 'RIDGE_ALPHA']
            
            missing_attrs = []
            for attr in required_oak_attrs:
                if not hasattr(oak_config, attr):
                    missing_attrs.append(f"oak_config.{attr}")
            
            for attr in required_mvpa_attrs:
                if not hasattr(mvpa_config, attr):
                    missing_attrs.append(f"mvpa_config.{attr}")
            
            if missing_attrs:
                logger.error(f"Validation failed: Missing attributes: {', '.join(missing_attrs)}")
                return False
            
            # Test basic functionality
            roi_masks = config.get_roi_mask_paths()
            output_paths = config.get_output_paths()
            
            if not roi_masks:
                logger.error("Validation failed: No ROI masks found")
                return False
            
            if not output_paths:
                logger.error("Validation failed: No output paths found")
                return False
            
            logger.info("✓ Migration validation passed!")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False
    
    def create_backup_all(self):
        """Create backup of all known configuration files"""
        logger.info("Creating backup of all configuration files...")
        
        config_files = [
            'oak_storage_config.py',
            'dd_geometry_config.json',
            'mvpa_utils.py',
            'config.yaml',
            'config.yml'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                self._create_backup(config_file)
        
        logger.info("Backup creation complete")


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description="Configuration Migration Utility")
    parser.add_argument('--migrate-all', action='store_true', help='Migrate all known configuration files')
    parser.add_argument('--migrate-from', type=str, help='Migrate from specific configuration file')
    parser.add_argument('--validate-migration', action='store_true', help='Validate migrated configuration')
    parser.add_argument('--create-backup', action='store_true', help='Create backup of all configuration files')
    parser.add_argument('--output-file', type=str, default='config.yaml', help='Output configuration file')
    parser.add_argument('--backup-dir', type=str, default='config_backup', help='Backup directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize migrator
    migrator = ConfigMigrator(args.output_file, args.backup_dir)
    
    try:
        if args.migrate_all:
            migrator.migrate_all()
        elif args.migrate_from:
            migrator.migrate_from_file(args.migrate_from)
        elif args.validate_migration:
            success = migrator.validate_migration()
            if not success:
                exit(1)
        elif args.create_backup:
            migrator.create_backup_all()
        else:
            print("Please specify an action. Use --help for options.")
            exit(1)
        
        print("Migration completed successfully!")
        
    except ConfigMigrationError as e:
        logger.error(f"Migration failed: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main() 