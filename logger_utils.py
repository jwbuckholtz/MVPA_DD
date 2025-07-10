#!/usr/bin/env python3
"""
Logger and Utilities Module for Delay Discounting Analysis Pipeline
===================================================================

This module provides standardized logging, argument parsing, and import management
for the delay discounting fMRI analysis pipeline. It eliminates code duplication
and ensures consistent behavior across all pipeline components.

Key Features:
- Standardized logging setup with configurable levels and formatting
- Common argument parsing patterns for pipeline scripts
- Centralized import management with optional dependency handling
- Configuration validation and environment setup
- Progress reporting and error handling utilities
- Integration with existing pipeline components

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import logging
import argparse
import warnings
import traceback
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Core imports that should be available in all scripts
CORE_IMPORTS = [
    'os', 'sys', 'numpy', 'pandas', 'pickle', 'pathlib', 'warnings'
]

# Optional imports with fallback behavior
OPTIONAL_IMPORTS = {
    'scientific': ['scipy', 'matplotlib', 'seaborn', 'sklearn', 'statsmodels'],
    'neuroimaging': ['nibabel', 'nilearn'],
    'pipeline': ['data_utils', 'mvpa_utils', 'oak_storage_config']
}

class LoggerConfig:
    """Configuration for logging setup"""
    
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # Default format
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    SIMPLE_FORMAT = '%(levelname)s: %(message)s'
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    # Default settings
    DEFAULT_LEVEL = INFO
    DEFAULT_FILE_LEVEL = DEBUG
    DEFAULT_CONSOLE_LEVEL = INFO


class PipelineLogger:
    """Standardized logger for pipeline components"""
    
    def __init__(self, name: str, config: LoggerConfig = None):
        self.name = name
        self.config = config or LoggerConfig()
        self.logger = logging.getLogger(name)
        self.console_handler = None
        self.file_handler = None
        self.setup_complete = False
    
    def setup_logging(self, 
                     console_level: int = None,
                     file_level: int = None,
                     log_file: str = None,
                     format_style: str = 'default',
                     suppress_warnings: bool = True) -> logging.Logger:
        """
        Setup standardized logging for pipeline components
        
        Parameters:
        -----------
        console_level : int, optional
            Console logging level (default: INFO)
        file_level : int, optional
            File logging level (default: DEBUG)
        log_file : str, optional
            Path to log file (default: None - no file logging)
        format_style : str
            Format style ('default', 'simple', 'detailed')
        suppress_warnings : bool
            Whether to suppress common warnings
        
        Returns:
        --------
        logging.Logger : Configured logger
        """
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set logger level to most permissive
        self.logger.setLevel(logging.DEBUG)
        
        # Choose format
        if format_style == 'simple':
            formatter = logging.Formatter(self.config.SIMPLE_FORMAT)
        elif format_style == 'detailed':
            formatter = logging.Formatter(self.config.DETAILED_FORMAT)
        else:
            formatter = logging.Formatter(self.config.DEFAULT_FORMAT)
        
        # Console handler
        console_level = console_level or self.config.DEFAULT_CONSOLE_LEVEL
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(console_level)
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.console_handler)
        
        # File handler (if specified)
        if log_file:
            file_level = file_level or self.config.DEFAULT_FILE_LEVEL
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.file_handler = logging.FileHandler(log_path)
            self.file_handler.setLevel(file_level)
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
        
        # Suppress common warnings
        if suppress_warnings:
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        self.setup_complete = True
        return self.logger
    
    def log_system_info(self):
        """Log system information for debugging"""
        if not self.setup_complete:
            return
        
        self.logger.info(f"System: {sys.platform}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"Script: {sys.argv[0]}")
        self.logger.info(f"Arguments: {sys.argv[1:]}")
    
    def log_pipeline_start(self, script_name: str, description: str = None):
        """Log pipeline component start"""
        if not self.setup_complete:
            return
        
        self.logger.info("=" * 60)
        self.logger.info(f"Starting {script_name}")
        if description:
            self.logger.info(f"Description: {description}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 60)
    
    def log_pipeline_end(self, script_name: str, success: bool = True):
        """Log pipeline component end"""
        if not self.setup_complete:
            return
        
        status = "COMPLETED" if success else "FAILED"
        self.logger.info("=" * 60)
        self.logger.info(f"{script_name} {status}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 60)
    
    def log_error_with_traceback(self, error: Exception, context: str = None):
        """Log error with full traceback"""
        if not self.setup_complete:
            return
        
        if context:
            self.logger.error(f"Error in {context}: {str(error)}")
        else:
            self.logger.error(f"Error: {str(error)}")
        
        # Log full traceback at debug level
        tb_str = traceback.format_exc()
        self.logger.debug(f"Full traceback:\n{tb_str}")
    
    def create_progress_logger(self, total_items: int, description: str = "Processing"):
        """Create a progress logger for long-running operations"""
        return ProgressLogger(self.logger, total_items, description)


class ProgressLogger:
    """Progress logging utility"""
    
    def __init__(self, logger: logging.Logger, total_items: int, description: str):
        self.logger = logger
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
        
        # Log intervals
        self.log_every_n = max(1, total_items // 10)  # Log every 10%
        self.log_every_seconds = 30  # Log every 30 seconds
    
    def update(self, item_description: str = None):
        """Update progress"""
        self.current_item += 1
        current_time = datetime.now()
        
        # Check if we should log
        should_log = (
            self.current_item % self.log_every_n == 0 or
            (current_time - self.last_log_time).total_seconds() >= self.log_every_seconds or
            self.current_item == self.total_items
        )
        
        if should_log:
            progress_pct = (self.current_item / self.total_items) * 100
            elapsed = current_time - self.start_time
            
            if self.current_item < self.total_items:
                eta = elapsed * (self.total_items - self.current_item) / self.current_item
                eta_str = f", ETA: {eta}"
            else:
                eta_str = ""
            
            msg = f"{self.description}: {self.current_item}/{self.total_items} ({progress_pct:.1f}%)"
            if item_description:
                msg += f" - {item_description}"
            msg += f" [Elapsed: {elapsed}{eta_str}]"
            
            self.logger.info(msg)
            self.last_log_time = current_time


class CommonArgumentParser:
    """Standardized argument parsing for pipeline scripts"""
    
    def __init__(self, description: str, script_type: str = 'analysis'):
        self.description = description
        self.script_type = script_type
        self.parser = argparse.ArgumentParser(description=description)
        self.added_groups = {}
    
    def add_data_arguments(self, require_neural: bool = True, require_behavioral: bool = True):
        """Add common data-related arguments"""
        data_group = self.parser.add_argument_group('Data Input')
        
        if require_neural:
            data_group.add_argument('--neural-data', required=True,
                                  help='Path to neural data file (.npy or .csv)')
        else:
            data_group.add_argument('--neural-data',
                                  help='Path to neural data file (.npy or .csv)')
        
        if require_behavioral:
            data_group.add_argument('--behavioral-data', required=True,
                                  help='Path to behavioral data CSV file')
        else:
            data_group.add_argument('--behavioral-data',
                                  help='Path to behavioral data CSV file')
        
        data_group.add_argument('--results-file',
                               help='Path to results pickle file')
        data_group.add_argument('--config',
                               help='Path to configuration file')
        
        self.added_groups['data'] = data_group
        return self
    
    def add_output_arguments(self, default_output_dir: str = './results'):
        """Add output-related arguments"""
        output_group = self.parser.add_argument_group('Output Control')
        
        output_group.add_argument('--output-dir', default=default_output_dir,
                                 help=f'Output directory (default: {default_output_dir})')
        output_group.add_argument('--save-intermediate', action='store_true',
                                 help='Save intermediate results')
        output_group.add_argument('--overwrite', action='store_true',
                                 help='Overwrite existing results')
        
        self.added_groups['output'] = output_group
        return self
    
    def add_analysis_arguments(self, analysis_type: str = 'mvpa'):
        """Add analysis-specific arguments"""
        analysis_group = self.parser.add_argument_group('Analysis Options')
        
        if analysis_type == 'mvpa':
            analysis_group.add_argument('--roi-name', default='unknown',
                                       help='Name of the ROI being analyzed')
            analysis_group.add_argument('--algorithms', nargs='+',
                                       default=['svm', 'logistic'],
                                       help='ML algorithms to use')
            analysis_group.add_argument('--cv-folds', type=int, default=5,
                                       help='Number of cross-validation folds')
            analysis_group.add_argument('--n-permutations', type=int, default=1000,
                                       help='Number of permutation tests')
        
        elif analysis_type == 'geometry':
            analysis_group.add_argument('--roi-name', default='ROI',
                                       help='Name of the ROI')
            analysis_group.add_argument('--methods', nargs='+',
                                       default=['pca', 'mds'],
                                       help='Dimensionality reduction methods')
            analysis_group.add_argument('--comparisons', nargs='+',
                                       help='Specific comparisons to run')
        
        elif analysis_type == 'behavioral':
            analysis_group.add_argument('--subject-id',
                                       help='Subject identifier')
            analysis_group.add_argument('--validate-data', action='store_true',
                                       help='Run data validation')
        
        self.added_groups['analysis'] = analysis_group
        return self
    
    def add_execution_arguments(self):
        """Add execution control arguments"""
        exec_group = self.parser.add_argument_group('Execution Control')
        
        exec_group.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose output')
        exec_group.add_argument('--debug', action='store_true',
                               help='Enable debug logging')
        exec_group.add_argument('--quiet', '-q', action='store_true',
                               help='Suppress non-error output')
        exec_group.add_argument('--log-file',
                               help='Path to log file')
        exec_group.add_argument('--n-jobs', type=int, default=1,
                               help='Number of parallel jobs')
        
        self.added_groups['execution'] = exec_group
        return self
    
    def add_testing_arguments(self):
        """Add testing and demo arguments"""
        test_group = self.parser.add_argument_group('Testing Options')
        
        test_group.add_argument('--example', action='store_true',
                               help='Run with example/synthetic data')
        test_group.add_argument('--demo-type', choices=['basic', 'advanced', 'all'],
                               default='basic', help='Type of demo to run')
        test_group.add_argument('--quick', action='store_true',
                               help='Run quick tests only')
        test_group.add_argument('--check-data', action='store_true',
                               help='Check data integrity')
        
        self.added_groups['testing'] = test_group
        return self
    
    def add_visualization_arguments(self):
        """Add visualization arguments"""
        viz_group = self.parser.add_argument_group('Visualization')
        
        viz_group.add_argument('--plot', action='store_true',
                              help='Generate plots')
        viz_group.add_argument('--plot-format', choices=['png', 'pdf', 'svg'],
                              default='png', help='Plot format')
        viz_group.add_argument('--dpi', type=int, default=300,
                              help='Plot resolution')
        viz_group.add_argument('--no-display', action='store_true',
                              help='Disable plot display (save only)')
        
        self.added_groups['visualization'] = viz_group
        return self
    
    def parse_args(self, args=None):
        """Parse arguments and return namespace"""
        parsed_args = self.parser.parse_args(args)
        
        # Validate argument combinations
        if hasattr(parsed_args, 'verbose') and hasattr(parsed_args, 'quiet'):
            if parsed_args.verbose and parsed_args.quiet:
                self.parser.error("Cannot specify both --verbose and --quiet")
        
        return parsed_args


class ImportManager:
    """Centralized import management with optional dependency handling"""
    
    def __init__(self):
        self.import_status = {}
        self.failed_imports = []
        self.warnings = []
    
    def import_core_modules(self) -> Dict[str, Any]:
        """Import core modules that should always be available"""
        modules = {}
        
        for module_name in CORE_IMPORTS:
            try:
                if module_name == 'numpy':
                    import numpy as np
                    modules['np'] = np
                    modules['numpy'] = np
                elif module_name == 'pandas':
                    import pandas as pd
                    modules['pd'] = pd
                    modules['pandas'] = pd
                elif module_name == 'pathlib':
                    from pathlib import Path
                    modules['Path'] = Path
                    modules['pathlib'] = Path
                elif module_name == 'pickle':
                    import pickle
                    modules['pickle'] = pickle
                elif module_name == 'warnings':
                    import warnings
                    modules['warnings'] = warnings
                else:
                    modules[module_name] = __import__(module_name)
                
                self.import_status[module_name] = True
                
            except ImportError as e:
                self.import_status[module_name] = False
                self.failed_imports.append(module_name)
                self.warnings.append(f"Failed to import core module {module_name}: {e}")
        
        return modules
    
    def import_optional_modules(self, category: str = 'all') -> Dict[str, Any]:
        """Import optional modules with fallback behavior"""
        modules = {}
        
        if category == 'all':
            categories = list(OPTIONAL_IMPORTS.keys())
        else:
            categories = [category] if category in OPTIONAL_IMPORTS else []
        
        for cat in categories:
            for module_name in OPTIONAL_IMPORTS[cat]:
                try:
                    # Handle special cases
                    if module_name == 'scipy':
                        import scipy
                        from scipy import stats
                        modules['scipy'] = scipy
                        modules['stats'] = stats
                    elif module_name == 'sklearn':
                        import sklearn
                        modules['sklearn'] = sklearn
                    elif module_name == 'nibabel':
                        import nibabel as nib
                        modules['nib'] = nib
                        modules['nibabel'] = nib
                    elif module_name == 'nilearn':
                        import nilearn
                        modules['nilearn'] = nilearn
                    elif module_name == 'matplotlib':
                        import matplotlib
                        import matplotlib.pyplot as plt
                        modules['matplotlib'] = matplotlib
                        modules['plt'] = plt
                    elif module_name == 'seaborn':
                        import seaborn as sns
                        modules['sns'] = sns
                        modules['seaborn'] = sns
                    else:
                        modules[module_name] = __import__(module_name)
                    
                    self.import_status[module_name] = True
                    
                except ImportError as e:
                    self.import_status[module_name] = False
                    self.failed_imports.append(module_name)
                    # Only warn for pipeline modules, not external ones
                    if cat == 'pipeline':
                        self.warnings.append(f"Failed to import {module_name}: {e}")
        
        return modules
    
    def check_requirements(self, required_modules: List[str]) -> bool:
        """Check if required modules are available"""
        missing = []
        for module in required_modules:
            if module not in self.import_status or not self.import_status[module]:
                missing.append(module)
        
        if missing:
            raise ImportError(f"Missing required modules: {missing}")
        
        return True
    
    def get_import_report(self) -> str:
        """Generate import status report"""
        report = ["Import Status Report", "=" * 30]
        
        if self.failed_imports:
            report.append(f"Failed imports: {len(self.failed_imports)}")
            for module in self.failed_imports:
                report.append(f"  ✗ {module}")
        
        successful = [m for m in self.import_status if self.import_status[m]]
        if successful:
            report.append(f"Successful imports: {len(successful)}")
            for module in successful:
                report.append(f"  ✓ {module}")
        
        if self.warnings:
            report.append("\nWarnings:")
            for warning in self.warnings:
                report.append(f"  ! {warning}")
        
        return "\n".join(report)


class ConfigurationManager:
    """Centralized configuration management"""
    
    def __init__(self):
        self.config_cache = {}
        self.config_validators = {}
    
    def load_config(self, config_path: str = None, config_type: str = 'oak') -> Any:
        """Load configuration with caching"""
        if config_path and config_path in self.config_cache:
            return self.config_cache[config_path]
        
        try:
            if config_type == 'oak':
                from oak_storage_config import OAKConfig
                config = OAKConfig()
            elif config_path and config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Default config
                config = self._create_default_config()
            
            if config_path:
                self.config_cache[config_path] = config
            
            return config
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'cv_folds': 5,
            'n_permutations': 1000,
            'n_jobs': 1,
            'random_state': 42,
            'alpha': 0.05,
            'output_dir': './results',
            'log_level': 'INFO'
        }
    
    def validate_config(self, config: Any, config_type: str = 'analysis') -> bool:
        """Validate configuration"""
        if config_type == 'analysis':
            required_keys = ['cv_folds', 'n_permutations', 'output_dir']
            if isinstance(config, dict):
                missing = [k for k in required_keys if k not in config]
                if missing:
                    raise ValueError(f"Missing required config keys: {missing}")
        
        return True
    
    def setup_directories(self, config: Any) -> None:
        """Setup output directories from configuration"""
        if hasattr(config, 'OUTPUT_DIR'):
            output_dir = Path(config.OUTPUT_DIR)
        elif isinstance(config, dict) and 'output_dir' in config:
            output_dir = Path(config['output_dir'])
        else:
            output_dir = Path('./results')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['logs', 'plots', 'intermediate']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)


# Convenience functions for common usage patterns
def setup_script_logging(script_name: str, 
                        log_level: str = 'INFO',
                        log_file: str = None,
                        verbose: bool = False,
                        debug: bool = False,
                        quiet: bool = False) -> PipelineLogger:
    """
    Setup logging for a pipeline script
    
    Parameters:
    -----------
    script_name : str
        Name of the script
    log_level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Path to log file
    verbose : bool
        Enable verbose logging
    debug : bool
        Enable debug logging
    quiet : bool
        Suppress non-error output
    
    Returns:
    --------
    PipelineLogger : Configured logger
    """
    logger = PipelineLogger(script_name)
    
    # Determine log levels
    if debug:
        console_level = logging.DEBUG
    elif verbose:
        console_level = logging.INFO
    elif quiet:
        console_level = logging.WARNING
    else:
        console_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup logging
    logger.setup_logging(
        console_level=console_level,
        log_file=log_file,
        format_style='detailed' if debug else 'default'
    )
    
    return logger


def create_analysis_parser(script_name: str, 
                          analysis_type: str = 'mvpa',
                          require_data: bool = True) -> CommonArgumentParser:
    """
    Create a standard argument parser for analysis scripts
    
    Parameters:
    -----------
    script_name : str
        Name of the script
    analysis_type : str
        Type of analysis ('mvpa', 'geometry', 'behavioral')
    require_data : bool
        Whether to require data inputs
    
    Returns:
    --------
    CommonArgumentParser : Configured parser
    """
    parser = CommonArgumentParser(
        description=f"{script_name} - Delay Discounting Analysis Pipeline",
        script_type=analysis_type
    )
    
    # Add standard argument groups
    if require_data:
        parser.add_data_arguments()
    
    parser.add_output_arguments()
    parser.add_analysis_arguments(analysis_type)
    parser.add_execution_arguments()
    parser.add_testing_arguments()
    parser.add_visualization_arguments()
    
    return parser


def validate_script_environment(required_modules: List[str] = None) -> ImportManager:
    """
    Validate script environment and import requirements
    
    Parameters:
    -----------
    required_modules : List[str], optional
        List of required module names
    
    Returns:
    --------
    ImportManager : Import manager with status
    """
    manager = ImportManager()
    
    # Import core modules
    manager.import_core_modules()
    
    # Import optional modules
    manager.import_optional_modules()
    
    # Check requirements
    if required_modules:
        manager.check_requirements(required_modules)
    
    return manager


def setup_pipeline_environment(script_name: str,
                              args: argparse.Namespace,
                              required_modules: List[str] = None) -> Dict[str, Any]:
    """
    Complete environment setup for pipeline scripts
    
    Parameters:
    -----------
    script_name : str
        Name of the script
    args : argparse.Namespace
        Parsed command line arguments
    required_modules : List[str], optional
        Required modules to check
    
    Returns:
    --------
    Dict[str, Any] : Environment components (logger, config, imports)
    """
    # Setup logging
    logger = setup_script_logging(
        script_name=script_name,
        log_level='DEBUG' if args.debug else 'INFO',
        log_file=getattr(args, 'log_file', None),
        verbose=getattr(args, 'verbose', False),
        debug=getattr(args, 'debug', False),
        quiet=getattr(args, 'quiet', False)
    )
    
    # Log script start
    logger.log_pipeline_start(script_name)
    logger.log_system_info()
    
    # Validate environment
    import_manager = validate_script_environment(required_modules)
    
    # Load configuration
    config_manager = ConfigurationManager()
    config = config_manager.load_config(getattr(args, 'config', None))
    
    # Setup directories
    config_manager.setup_directories(config)
    
    # Log environment status
    logger.logger.info("Environment setup complete")
    logger.logger.debug(import_manager.get_import_report())
    
    return {
        'logger': logger,
        'config': config,
        'import_manager': import_manager,
        'config_manager': config_manager
    }


# Integration with existing pipeline components
def get_pipeline_logger(script_name: str = None) -> PipelineLogger:
    """Get a logger for pipeline components"""
    if script_name is None:
        script_name = Path(sys.argv[0]).stem
    
    return PipelineLogger(script_name)


def log_analysis_results(logger: PipelineLogger, 
                        results: Dict[str, Any],
                        analysis_type: str = 'analysis') -> None:
    """Log analysis results summary"""
    logger.logger.info(f"Analysis Results Summary ({analysis_type})")
    logger.logger.info("-" * 40)
    
    if 'success' in results:
        logger.logger.info(f"Success: {results['success']}")
    
    if 'n_subjects' in results:
        logger.logger.info(f"Subjects analyzed: {results['n_subjects']}")
    
    if 'mean_accuracy' in results:
        logger.logger.info(f"Mean accuracy: {results['mean_accuracy']:.3f}")
    
    if 'p_value' in results:
        logger.logger.info(f"P-value: {results['p_value']:.4f}")
    
    if 'error' in results:
        logger.logger.error(f"Error: {results['error']}") 