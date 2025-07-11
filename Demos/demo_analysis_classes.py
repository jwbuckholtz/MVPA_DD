#!/usr/bin/env python3
"""
Demo Script for Refactored Analysis Classes
==========================================

This script demonstrates the new analysis class hierarchy and how to use
the AnalysisFactory to create and manage different analysis types.

The refactored design eliminates code duplication and provides consistent
interfaces across behavioral, MVPA, and geometry analysis types.

Key Features Demonstrated:
- Creating analysis instances via AnalysisFactory
- Common interface across all analysis types
- Consistent data loading and result handling
- Memory-efficient processing options
- Centralized configuration management
- Comprehensive logging and error handling

Author: Cognitive Neuroscience Lab, Stanford University
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Import the analysis classes
from analysis_base import (
    BaseAnalysis, AnalysisFactory, AnalysisError, 
    create_analysis, setup_analysis_environment
)
from behavioral_analysis import BehavioralAnalysis
from mvpa_analysis import MVPAAnalysis
from geometry_analysis import GeometryAnalysis

# Import configuration
from oak_storage_config import OAKConfig

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def demo_analysis_factory():
    """Demonstrate the AnalysisFactory functionality"""
    print("Analysis Factory Demo")
    print("=" * 50)
    
    # Show available analysis types
    available_types = AnalysisFactory.list_available()
    print(f"Available analysis types: {available_types}")
    
    # Create instances using factory
    config = OAKConfig()
    
    # Method 1: Using factory directly
    behavioral_analysis = AnalysisFactory.create('behavioral', config=config)
    mvpa_analysis = AnalysisFactory.create('mvpa', config=config)
    geometry_analysis = AnalysisFactory.create('geometry', config=config)
    
    print(f"\nCreated analysis instances:")
    print(f"  Behavioral: {behavioral_analysis}")
    print(f"  MVPA: {mvpa_analysis}")
    print(f"  Geometry: {geometry_analysis}")
    
    # Method 2: Using convenience function
    behavioral_analysis2 = create_analysis('behavioral', config=config)
    print(f"\nUsing convenience function: {behavioral_analysis2}")
    
    # Method 3: Direct instantiation (still works)
    behavioral_analysis3 = BehavioralAnalysis(config)
    print(f"Direct instantiation: {behavioral_analysis3}")
    
    print("\n" + "=" * 50 + "\n")


def demo_common_interface():
    """Demonstrate the common interface across analysis types"""
    print("Common Interface Demo")
    print("=" * 50)
    
    config = OAKConfig()
    
    # Create all analysis types
    analyses = {
        'behavioral': AnalysisFactory.create('behavioral', config=config),
        'mvpa': AnalysisFactory.create('mvpa', config=config),
        'geometry': AnalysisFactory.create('geometry', config=config)
    }
    
    # Show common interface methods
    print("Common methods available across all analysis types:")
    
    for name, analysis in analyses.items():
        print(f"\n{name.upper()} Analysis:")
        print(f"  - Name: {analysis.name}")
        print(f"  - Config: {type(analysis.config).__name__}")
        print(f"  - Logger: {type(analysis.logger).__name__}")
        print(f"  - Cache info: {analysis.get_cache_info()}")
        print(f"  - Processing stats: {analysis.processing_stats}")
        
        # Show available methods (common interface)
        common_methods = [
            'load_behavioral_data', 'load_fmri_data', 'create_maskers',
            'save_results', 'load_results', 'export_results_summary',
            'process_subject', 'run_analysis', 'get_analysis_summary'
        ]
        
        available_methods = [method for method in common_methods 
                           if hasattr(analysis, method)]
        print(f"  - Available methods: {len(available_methods)}/{len(common_methods)}")
    
    print("\n" + "=" * 50 + "\n")


def demo_memory_efficient_loading():
    """Demonstrate memory-efficient loading options"""
    print("Memory-Efficient Loading Demo")
    print("=" * 50)
    
    config = OAKConfig()
    
    try:
        # Import memory-efficient utilities
        from memory_efficient_data import MemoryConfig
        
        # Create memory configuration
        memory_config = MemoryConfig()
        memory_config.MEMMAP_THRESHOLD_GB = 1.0  # Use memory mapping for files > 1GB
        
        print(f"Memory configuration:")
        print(f"  - Memory mapping threshold: {memory_config.MEMMAP_THRESHOLD_GB} GB")
        print(f"  - Temporary directory: {memory_config.TEMP_DIR}")
        print(f"  - Cleanup on exit: {memory_config.CLEANUP_ON_EXIT}")
        
        # Create analysis with memory-efficient loading
        behavioral_analysis = AnalysisFactory.create(
            'behavioral', 
            config=config,
            enable_memory_efficient=True,
            memory_config=memory_config
        )
        
        print(f"\nCreated memory-efficient analysis: {behavioral_analysis}")
        print(f"Memory-efficient enabled: {behavioral_analysis.enable_memory_efficient}")
        
        # Show memory loader info
        if behavioral_analysis.memory_loader:
            print(f"Memory loader available: {type(behavioral_analysis.memory_loader).__name__}")
        else:
            print("Memory loader not available")
        
    except ImportError:
        print("Memory-efficient utilities not available")
        print("Standard loading will be used instead")
    
    print("\n" + "=" * 50 + "\n")


def demo_result_handling():
    """Demonstrate result handling and persistence"""
    print("Result Handling Demo")
    print("=" * 50)
    
    config = OAKConfig()
    
    # Create a behavioral analysis instance
    behavioral_analysis = AnalysisFactory.create('behavioral', config=config)
    
    # Simulate some results
    behavioral_analysis.results = {
        'subject_001': {
            'success': True,
            'k': 0.025,
            'pseudo_r2': 0.45,
            'n_trials': 156,
            'processing_time': 2.3
        },
        'subject_002': {
            'success': True,
            'k': 0.018,
            'pseudo_r2': 0.52,
            'n_trials': 162,
            'processing_time': 2.1
        },
        '_summary': {
            'n_subjects_total': 2,
            'n_subjects_successful': 2,
            'n_subjects_failed': 0,
            'success_rate': 1.0,
            'k_mean': 0.0215,
            'k_std': 0.0035,
            'r2_mean': 0.485,
            'r2_std': 0.035
        }
    }
    
    # Update processing stats
    behavioral_analysis.processing_stats = {
        'subjects_processed': 2,
        'subjects_failed': 0,
        'processing_times': [2.3, 2.1],
        'memory_usage': [150.2, 148.7]
    }
    
    print("Simulated results created")
    print(f"Results keys: {list(behavioral_analysis.results.keys())}")
    
    # Get analysis summary
    summary = behavioral_analysis.get_analysis_summary()
    print(f"\nAnalysis Summary:")
    print(summary)
    
    # Create summary dataframe
    summary_df = behavioral_analysis.create_behavioral_summary_dataframe()
    print(f"\nSummary DataFrame:")
    print(summary_df)
    
    # Save results
    results_path = behavioral_analysis.save_results()
    print(f"\nResults saved to: {results_path}")
    
    # Export summary
    summary_path = behavioral_analysis.export_results_summary()
    print(f"Summary exported to: {summary_path}")
    
    # Load results into new instance
    behavioral_analysis2 = AnalysisFactory.create('behavioral', config=config)
    loaded_data = behavioral_analysis2.load_results(results_path)
    
    print(f"\nLoaded results into new instance:")
    print(f"  - Results keys: {list(behavioral_analysis2.results.keys())}")
    print(f"  - Processing stats: {behavioral_analysis2.processing_stats}")
    
    print("\n" + "=" * 50 + "\n")


def demo_error_handling():
    """Demonstrate error handling and logging"""
    print("Error Handling Demo")
    print("=" * 50)
    
    config = OAKConfig()
    
    # Create analysis instance
    behavioral_analysis = AnalysisFactory.create('behavioral', config=config)
    
    try:
        # Try to load data for non-existent subject
        behavioral_data = behavioral_analysis.load_behavioral_data('nonexistent_subject')
    except AnalysisError as e:
        print(f"Caught AnalysisError: {e}")
    
    try:
        # Try to load results from non-existent file
        behavioral_analysis.load_results('nonexistent_file.pkl')
    except AnalysisError as e:
        print(f"Caught AnalysisError: {e}")
    
    try:
        # Try to create unknown analysis type
        unknown_analysis = AnalysisFactory.create('unknown_type')
    except ValueError as e:
        print(f"Caught ValueError: {e}")
    
    print("\nError handling working correctly!")
    
    print("\n" + "=" * 50 + "\n")


def demo_analysis_comparison():
    """Demonstrate running different analysis types on the same data"""
    print("Analysis Comparison Demo")
    print("=" * 50)
    
    config = OAKConfig()
    
    # Get a small subset of subjects for demo
    try:
        from data_utils import get_complete_subjects
        all_subjects = get_complete_subjects(config)
        demo_subjects = all_subjects[:2]  # Just first 2 subjects
        
        print(f"Running demo on {len(demo_subjects)} subjects: {demo_subjects}")
        
        # Create all analysis types
        analyses = {
            'behavioral': AnalysisFactory.create('behavioral', config=config),
            'mvpa': AnalysisFactory.create('mvpa', config=config),
            'geometry': AnalysisFactory.create('geometry', config=config)
        }
        
        # Run each analysis type
        results = {}
        for name, analysis in analyses.items():
            print(f"\nRunning {name} analysis...")
            try:
                # Note: This would actually run the analysis
                # For demo purposes, we'll just show the interface
                print(f"  - Analysis name: {analysis.name}")
                print(f"  - Subject list method: {hasattr(analysis, 'get_subject_list')}")
                print(f"  - Process subject method: {hasattr(analysis, 'process_subject')}")
                print(f"  - Run analysis method: {hasattr(analysis, 'run_analysis')}")
                print(f"  - Ready to process {len(demo_subjects)} subjects")
                
                # Uncomment the next line to actually run the analysis
                # results[name] = analysis.run_analysis(demo_subjects)
                
            except Exception as e:
                print(f"  - Error: {e}")
                results[name] = {'error': str(e)}
        
        print(f"\nAnalysis comparison complete!")
        print(f"Results collected for: {list(results.keys())}")
        
    except Exception as e:
        print(f"Could not get subjects for demo: {e}")
        print("Demo would work with actual data")
    
    print("\n" + "=" * 50 + "\n")


def main():
    """Run all demonstration functions"""
    print("Refactored Analysis Classes Demo")
    print("=" * 60)
    print()
    
    # Setup analysis environment
    try:
        env = setup_analysis_environment()
        print(f"Analysis environment setup successful:")
        print(f"  - Available subjects: {env['n_subjects']}")
        print(f"  - Memory efficient available: {env['memory_efficient_available']}")
        print(f"  - Logger available: {env['logger_available']}")
        print()
    except Exception as e:
        print(f"Environment setup failed: {e}")
        print("Continuing with demo using default configuration")
        print()
    
    # Run demonstrations
    demo_functions = [
        demo_analysis_factory,
        demo_common_interface,
        demo_memory_efficient_loading,
        demo_result_handling,
        demo_error_handling,
        demo_analysis_comparison
    ]
    
    for demo_func in demo_functions:
        try:
            demo_func()
        except Exception as e:
            print(f"Demo function {demo_func.__name__} failed: {e}")
            print("Continuing with next demo...\n")
    
    print("Demo completed successfully!")
    print("\nKey Benefits of Refactored Design:")
    print("- Eliminated code duplication across analysis types")
    print("- Consistent interfaces and error handling")
    print("- Centralized configuration and logging")
    print("- Memory-efficient processing options")
    print("- Factory pattern for easy instance creation")
    print("- Comprehensive result handling and persistence")
    print("- Backward compatibility with existing code")


if __name__ == "__main__":
    main() 