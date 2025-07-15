#!/usr/bin/env python3
"""
Demo Script for Logger Utilities Module
=======================================

This script demonstrates how to use the logger_utils module for standardized
logging, argument parsing, and import management in the delay discounting
analysis pipeline.

Usage:
    python demo_logger_utils.py [options]

Author: Cognitive Neuroscience Lab, Stanford University
"""

import sys
import time
import numpy as np
from pathlib import Path

# Import logger utilities
from logger_utils import (
    setup_script_logging, create_analysis_parser, validate_script_environment,
    setup_pipeline_environment, get_pipeline_logger, log_analysis_results,
    PipelineLogger, CommonArgumentParser, ImportManager, ConfigurationManager
)


def demo_basic_logging():
    """Demonstrate basic logging functionality"""
    print("\n" + "="*50)
    print("DEMO 1: Basic Logging")
    print("="*50)
    
    # Create a logger
    logger = setup_script_logging(
        script_name='demo_basic_logging',
        log_level='INFO',
        verbose=True
    )
    
    # Log pipeline start
    logger.log_pipeline_start('Demo Basic Logging', 'Demonstrating basic logging features')
    
    # Log system info
    logger.log_system_info()
    
    # Different log levels
    logger.logger.debug("This is a debug message")
    logger.logger.info("This is an info message")
    logger.logger.warning("This is a warning message")
    logger.logger.error("This is an error message")
    
    # Log pipeline end
    logger.log_pipeline_end('Demo Basic Logging', success=True)
    
    return logger


def demo_advanced_logging():
    """Demonstrate advanced logging features"""
    print("\n" + "="*50)
    print("DEMO 2: Advanced Logging")
    print("="*50)
    
    # Create logger with file output
    log_file = Path("./logs/demo_advanced.log")
    logger = setup_script_logging(
        script_name='demo_advanced_logging',
        log_level='DEBUG',
        log_file=str(log_file),
        debug=True
    )
    
    logger.log_pipeline_start('Demo Advanced Logging', 'File logging and error handling')
    
    # Demonstrate error logging with traceback
    try:
        # Intentionally cause an error
        result = 1 / 0
    except Exception as e:
        logger.log_error_with_traceback(e, "demo division")
    
    # Demonstrate progress logging
    progress = logger.create_progress_logger(total_items=10, description="Processing items")
    
    for i in range(10):
        time.sleep(0.1)  # Simulate work
        progress.update(f"Item {i+1}")
    
    logger.log_pipeline_end('Demo Advanced Logging', success=True)
    
    print(f"Log file created: {log_file}")
    if log_file.exists():
        print(f"Log file size: {log_file.stat().st_size} bytes")
    
    return logger


def demo_argument_parsing():
    """Demonstrate argument parsing functionality"""
    print("\n" + "="*50)
    print("DEMO 3: Argument Parsing")
    print("="*50)
    
    # Create parser for MVPA analysis
    parser = create_analysis_parser(
        script_name='demo_mvpa_analysis',
        analysis_type='mvpa',
        require_data=True
    )
    
    # Demo parsing with synthetic arguments
    test_args = [
        '--neural-data', 'test_neural.npy',
        '--behavioral-data', 'test_behavioral.csv',
        '--output-dir', './demo_results',
        '--roi-name', 'test_roi',
        '--cv-folds', '10',
        '--n-permutations', '500',
        '--verbose',
        '--plot'
    ]
    
    print("Test arguments:", test_args)
    args = parser.parse_args(test_args)
    
    print("\nParsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Demo geometry analysis parser
    print("\n" + "-"*30)
    print("Geometry Analysis Parser:")
    
    geo_parser = create_analysis_parser(
        script_name='demo_geometry_analysis',
        analysis_type='geometry',
        require_data=False
    )
    
    geo_args = [
        '--roi-name', 'DLPFC',
        '--methods', 'pca', 'mds', 'tsne',
        '--example',
        '--quick'
    ]
    
    print("Geometry arguments:", geo_args)
    geo_parsed = geo_parser.parse_args(geo_args)
    
    print("\nParsed geometry arguments:")
    for key, value in vars(geo_parsed).items():
        print(f"  {key}: {value}")
    
    return args


def demo_import_management():
    """Demonstrate import management functionality"""
    print("\n" + "="*50)
    print("DEMO 4: Import Management")
    print("="*50)
    
    # Create import manager
    manager = ImportManager()
    
    # Import core modules
    print("Importing core modules...")
    core_modules = manager.import_core_modules()
    print(f"Core modules imported: {list(core_modules.keys())}")
    
    # Import optional modules
    print("\nImporting optional modules...")
    optional_modules = manager.import_optional_modules()
    print(f"Optional modules imported: {list(optional_modules.keys())}")
    
    # Check specific requirements
    print("\nChecking requirements...")
    try:
        manager.check_requirements(['numpy', 'pandas'])
        print("✓ Core requirements satisfied")
    except ImportError as e:
        print(f"✗ Requirements not met: {e}")
    
    # Generate import report
    print("\nImport Report:")
    print(manager.get_import_report())
    
    return manager


def demo_configuration_management():
    """Demonstrate configuration management"""
    print("\n" + "="*50)
    print("DEMO 5: Configuration Management")
    print("="*50)
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Load default configuration
    print("Loading default configuration...")
    default_config = config_manager.load_config()
    print("Default config:")
    if hasattr(default_config, 'items'):
        for key, value in default_config.items():
            print(f"  {key}: {value}")
    else:
        # Handle OAKConfig class
        config_attrs = [attr for attr in dir(default_config) if not attr.startswith('_')]
        for attr in config_attrs[:10]:  # Show first 10 attributes
            value = getattr(default_config, attr)
            print(f"  {attr}: {value}")
    
    # Try to load OAK config
    print("\nTrying to load OAK configuration...")
    try:
        oak_config = config_manager.load_config(config_type='oak')
        print("✓ OAK configuration loaded successfully")
        print(f"Output directory: {getattr(oak_config, 'OUTPUT_DIR', 'Not set')}")
    except Exception as e:
        print(f"✗ Failed to load OAK config: {e}")
    
    # Validate configuration
    print("\nValidating configuration...")
    try:
        # Create a dict config for validation demo
        dict_config = {
            'cv_folds': 5,
            'n_permutations': 1000,
            'output_dir': './results'
        }
        config_manager.validate_config(dict_config, 'analysis')
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # Setup directories
    print("\nSetting up directories...")
    try:
        config_manager.setup_directories(default_config)
        print("✓ Directories created")
    except (OSError, PermissionError) as e:
        print(f"✗ Directory creation failed (expected for OAK paths): {e}")
        print("  This is normal when OAK filesystem is not available locally")
    
    return config_manager


def demo_complete_environment_setup():
    """Demonstrate complete environment setup"""
    print("\n" + "="*50)
    print("DEMO 6: Complete Environment Setup")
    print("="*50)
    
    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.debug = False
            self.verbose = True
            self.quiet = False
            self.log_file = './logs/demo_complete.log'
            self.config = None
    
    args = MockArgs()
    
    # Setup complete environment
    print("Setting up complete environment...")
    env = setup_pipeline_environment(
        script_name='demo_complete_setup',
        args=args,
        required_modules=['numpy', 'pandas']
    )
    
    print("Environment components:")
    for key, value in env.items():
        print(f"  {key}: {type(value).__name__}")
    
    # Use the environment
    logger = env['logger']
    config = env['config']
    import_manager = env['import_manager']
    
    # Demo analysis with environment
    print("\nRunning mock analysis...")
    
    # Create some fake results
    fake_results = {
        'success': True,
        'n_subjects': 50,
        'mean_accuracy': 0.65,
        'p_value': 0.001,
        'roi_name': 'demo_roi'
    }
    
    # Log results
    log_analysis_results(logger, fake_results, 'classification')
    
    logger.log_pipeline_end('demo_complete_setup', success=True)
    
    return env


def demo_integration_examples():
    """Demonstrate integration with existing pipeline"""
    print("\n" + "="*50)
    print("DEMO 7: Integration Examples")
    print("="*50)
    
    # Example 1: Quick logger for existing script
    print("Example 1: Quick logger setup")
    logger = get_pipeline_logger('existing_script')
    logger.setup_logging(console_level=logger.config.INFO)
    logger.logger.info("Quick logger setup complete")
    
    # Example 2: Custom argument parser
    print("\nExample 2: Custom argument parser")
    parser = CommonArgumentParser("Custom Analysis Script")
    parser.add_data_arguments(require_neural=False, require_behavioral=True)
    parser.add_execution_arguments()
    parser.add_testing_arguments()
    
    # Example 3: Environment validation
    print("\nExample 3: Environment validation")
    try:
        env_manager = validate_script_environment(['numpy', 'pandas'])
        print("✓ Environment validation passed")
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")
    
    return parser


def demo_performance_comparison():
    """Demonstrate performance benefits of standardized logging"""
    print("\n" + "="*50)
    print("DEMO 8: Performance Comparison")
    print("="*50)
    
    # Old way - manual setup
    print("Old way - Manual logging setup:")
    start_time = time.time()
    
    import logging
    manual_logger = logging.getLogger('manual_demo')
    manual_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    manual_logger.addHandler(handler)
    
    old_time = time.time() - start_time
    print(f"Manual setup time: {old_time:.4f} seconds")
    
    # New way - standardized setup
    print("\nNew way - Standardized logging setup:")
    start_time = time.time()
    
    std_logger = setup_script_logging('standardized_demo', verbose=True)
    
    new_time = time.time() - start_time
    print(f"Standardized setup time: {new_time:.4f} seconds")
    
    # Compare functionality
    print("\nFunctionality comparison:")
    print("✓ Manual: Basic logging")
    print("✓ Standardized: Basic logging + system info + pipeline tracking + error handling + progress logging")
    
    return std_logger


def main():
    """Main demo function"""
    print("LOGGER UTILITIES MODULE DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates all features of the logger_utils module")
    
    # Create output directory
    Path("./logs").mkdir(exist_ok=True)
    
    try:
        # Run all demos
        demo_basic_logging()
        demo_advanced_logging()
        demo_argument_parsing()
        demo_import_management()
        demo_configuration_management()
        demo_complete_environment_setup()
        demo_integration_examples()
        demo_performance_comparison()
        
        print("\n" + "="*60)
        print("🎉 LOGGER UTILITIES DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Benefits Demonstrated:")
        print("• Standardized logging across all pipeline components")
        print("• Consistent argument parsing with common patterns")
        print("• Centralized import management with fallback behavior")
        print("• Automated environment setup and validation")
        print("• Integration with existing pipeline components")
        print("• Progress tracking and error handling")
        print("• Configuration management and validation")
        print("\nNext Steps:")
        print("1. Use setup_script_logging() in your scripts")
        print("2. Replace manual argparse with create_analysis_parser()")
        print("3. Use setup_pipeline_environment() for complete setup")
        print("4. Integrate with existing data_utils and mvpa_utils")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 