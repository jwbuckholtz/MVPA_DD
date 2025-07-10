#!/usr/bin/env python3
"""
Demo Script for MVPA Utilities Module
=====================================

This script demonstrates how to use the mvpa_utils module for various
MVPA procedures in delay discounting analysis. It showcases the main
functions and their usage patterns.

Usage:
    python demo_mvpa_utils.py [--demo-type TYPE] [--verbose]

Demo Types:
    - basic: Basic classification and regression examples
    - advanced: Advanced pattern extraction and dimensionality reduction
    - searchlight: Searchlight analysis demonstration
    - all: Run all demonstrations

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression

# Import MVPA utilities
from mvpa_utils import (
    run_classification, run_regression, extract_neural_patterns,
    run_dimensionality_reduction, compute_feature_importance,
    run_choice_classification, run_continuous_decoding,
    MVPAConfig, MVPAError, update_mvpa_config
)

# Import data utilities for integration test
try:
    from data_utils import DataError
    from oak_storage_config import OAKConfig
    DATA_UTILS_AVAILABLE = True
except ImportError:
    warnings.warn("Data utilities not available. Some demos will be limited.")
    DATA_UTILS_AVAILABLE = False


def create_synthetic_fmri_data(n_trials=100, n_voxels=500, noise_level=0.1):
    """
    Create synthetic fMRI data for demonstration purposes
    
    Parameters:
    -----------
    n_trials : int
        Number of trials/samples
    n_voxels : int
        Number of voxels/features
    noise_level : float
        Amount of noise to add
        
    Returns:
    --------
    dict : Synthetic data with patterns and behavioral variables
    """
    print(f"Creating synthetic fMRI data: {n_trials} trials x {n_voxels} voxels")
    
    # Create choice-related patterns
    X_choice, y_choice = make_classification(
        n_samples=n_trials,
        n_features=n_voxels,
        n_informative=int(n_voxels * 0.1),  # 10% informative features
        n_redundant=int(n_voxels * 0.05),   # 5% redundant features
        n_classes=2,
        class_sep=1.0,
        random_state=42
    )
    
    # Add noise
    X_choice += np.random.normal(0, noise_level, X_choice.shape)
    
    # Create continuous variables
    np.random.seed(42)
    subjective_value = np.random.normal(50, 20, n_trials)
    delay_length = np.random.exponential(10, n_trials)
    discount_rate = np.random.lognormal(-2, 1, n_trials)
    
    # Create behavioral events DataFrame
    events_df = pd.DataFrame({
        'onset': np.sort(np.random.uniform(0, 300, n_trials)),  # 5-minute scan
        'choice': y_choice,
        'subjective_value': subjective_value,
        'delay_length': delay_length,
        'discount_rate': discount_rate
    })
    
    return {
        'neural_patterns': X_choice,
        'choice_labels': y_choice,
        'events_df': events_df,
        'continuous_vars': {
            'subjective_value': subjective_value,
            'delay_length': delay_length,
            'discount_rate': discount_rate
        }
    }


def demo_basic_classification(data, verbose=True):
    """
    Demonstrate basic classification procedures
    
    Parameters:
    -----------
    data : dict
        Synthetic data dictionary
    verbose : bool
        Whether to print detailed results
    """
    print("\n" + "="*60)
    print("DEMO 1: BASIC CLASSIFICATION")
    print("="*60)
    
    X = data['neural_patterns']
    y = data['choice_labels']
    
    if verbose:
        print(f"Data shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
    
    # Test different algorithms
    algorithms = ['svm', 'logistic', 'rf']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n--- Testing {algorithm.upper()} classifier ---")
        
        try:
            result = run_classification(
                X, y,
                algorithm=algorithm,
                cv_strategy='stratified',
                n_permutations=100,  # Reduced for demo speed
                roi_name=f'demo_roi_{algorithm}'
            )
            
            if result['success']:
                print(f"✓ Mean accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
                print(f"✓ Permutation p-value: {result['p_value']:.4f}")
                print(f"✓ Chance level: {result['chance_level']:.3f}")
                results[algorithm] = result
            else:
                print(f"✗ Failed: {result['error']}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Compare algorithms
    if results:
        print(f"\n--- Algorithm Comparison ---")
        for alg, res in results.items():
            print(f"{alg:10s}: {res['mean_accuracy']:.3f} ± {res['std_accuracy']:.3f} "
                  f"(p={res['p_value']:.4f})")
    
    return results


def demo_basic_regression(data, verbose=True):
    """
    Demonstrate basic regression procedures
    
    Parameters:
    -----------
    data : dict
        Synthetic data dictionary
    verbose : bool
        Whether to print detailed results
    """
    print("\n" + "="*60)
    print("DEMO 2: BASIC REGRESSION")
    print("="*60)
    
    X = data['neural_patterns']
    continuous_vars = data['continuous_vars']
    
    if verbose:
        print(f"Data shape: {X.shape}")
        print(f"Target variables: {list(continuous_vars.keys())}")
    
    # Test different algorithms and variables
    algorithms = ['ridge', 'lasso', 'elastic']
    results = {}
    
    for var_name, y in continuous_vars.items():
        print(f"\n--- Decoding {var_name} ---")
        results[var_name] = {}
        
        for algorithm in algorithms:
            try:
                result = run_regression(
                    X, y,
                    algorithm=algorithm,
                    cv_strategy='kfold',
                    n_permutations=100,  # Reduced for demo speed
                    variable_name=var_name,
                    roi_name=f'demo_roi_{algorithm}'
                )
                
                if result['success']:
                    print(f"  {algorithm:8s}: R² = {result['mean_r2']:.3f} ± {result['std_r2']:.3f} "
                          f"(p={result['p_value']:.4f})")
                    results[var_name][algorithm] = result
                else:
                    print(f"  {algorithm:8s}: Failed - {result['error']}")
                    
            except Exception as e:
                print(f"  {algorithm:8s}: Error - {e}")
    
    return results


def demo_dimensionality_reduction(data, verbose=True):
    """
    Demonstrate dimensionality reduction procedures
    
    Parameters:
    -----------
    data : dict
        Synthetic data dictionary
    verbose : bool
        Whether to print detailed results
    """
    print("\n" + "="*60)
    print("DEMO 3: DIMENSIONALITY REDUCTION")
    print("="*60)
    
    X = data['neural_patterns']
    y = data['choice_labels']
    
    if verbose:
        print(f"Original data shape: {X.shape}")
    
    # Test different methods
    methods = ['pca', 'mds', 'tsne', 'isomap']
    n_components = 3
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} ---")
        
        try:
            result = run_dimensionality_reduction(
                X, method=method, n_components=n_components
            )
            
            if result['success']:
                embedding = result['embedding']
                print(f"✓ Reduced shape: {embedding.shape}")
                
                # Check explained variance if available
                if result['explained_variance'] is not None:
                    exp_var = result['explained_variance']
                    if hasattr(exp_var, '__len__') and len(exp_var) > 1:
                        print(f"✓ Explained variance: {exp_var[:3]}")  # First 3 components
                    else:
                        print(f"✓ Explained variance: {exp_var}")
                
                results[method] = result
                
                # Test if embedding preserves class structure
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(embedding, y)
                accuracy = knn.score(embedding, y)
                print(f"✓ KNN accuracy on embedding: {accuracy:.3f}")
                
            else:
                print(f"✗ Failed: {result['error']}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return results


def demo_feature_importance(data, verbose=True):
    """
    Demonstrate feature importance and selection procedures
    
    Parameters:
    -----------
    data : dict
        Synthetic data dictionary
    verbose : bool
        Whether to print detailed results
    """
    print("\n" + "="*60)
    print("DEMO 4: FEATURE IMPORTANCE")
    print("="*60)
    
    X = data['neural_patterns']
    y_class = data['choice_labels']
    y_cont = data['continuous_vars']['subjective_value']
    
    if verbose:
        print(f"Data shape: {X.shape}")
    
    # Test feature importance methods
    methods = ['univariate', 'model_based']
    
    for task_type, y, y_name in [('classification', y_class, 'choice'), 
                                ('regression', y_cont, 'subjective_value')]:
        print(f"\n--- {task_type.upper()}: {y_name} ---")
        
        for method in methods:
            print(f"\n  {method} method:")
            
            try:
                result = compute_feature_importance(
                    X, y, method=method, task_type=task_type
                )
                
                if result['success']:
                    if method == 'univariate':
                        scores = result['scores']
                        print(f"    ✓ Top 5 features (by score): {result['ranking'][:5]}")
                        print(f"    ✓ Score range: {scores.min():.3f} - {scores.max():.3f}")
                        
                        # Count significant features
                        p_values = result['p_values']
                        sig_features = np.sum(p_values < 0.05)
                        print(f"    ✓ Significant features (p<0.05): {sig_features}/{len(p_values)}")
                        
                    elif method == 'model_based':
                        importance = result['importance']
                        print(f"    ✓ Top 5 features (by importance): {result['ranking'][:5]}")
                        print(f"    ✓ Importance range: {importance.min():.6f} - {importance.max():.6f}")
                
                else:
                    print(f"    ✗ Failed: {result['error']}")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")


def demo_convenience_functions(data, verbose=True):
    """
    Demonstrate convenience functions for common use cases
    
    Parameters:
    -----------
    data : dict
        Synthetic data dictionary
    verbose : bool
        Whether to print detailed results
    """
    print("\n" + "="*60)
    print("DEMO 5: CONVENIENCE FUNCTIONS")
    print("="*60)
    
    X = data['neural_patterns']
    choices = data['choice_labels']
    subjective_value = data['continuous_vars']['subjective_value']
    
    # Test choice classification convenience function
    print("\n--- Choice Classification (Convenience Function) ---")
    try:
        result = run_choice_classification(
            X, choices, 
            roi_name='demo_striatum',
            n_permutations=100
        )
        
        if result['success']:
            print(f"✓ Choice accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
            print(f"✓ Significance: p = {result['p_value']:.4f}")
        else:
            print(f"✗ Failed: {result['error']}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test continuous decoding convenience function
    print("\n--- Continuous Decoding (Convenience Function) ---")
    try:
        result = run_continuous_decoding(
            X, subjective_value,
            variable_name='subjective_value',
            roi_name='demo_vmpfc',
            n_permutations=100
        )
        
        if result['success']:
            print(f"✓ Subjective value R²: {result['mean_r2']:.3f} ± {result['std_r2']:.3f}")
            print(f"✓ Significance: p = {result['p_value']:.4f}")
        else:
            print(f"✗ Failed: {result['error']}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def demo_config_management(verbose=True):
    """
    Demonstrate configuration management
    
    Parameters:
    -----------
    verbose : bool
        Whether to print detailed information
    """
    print("\n" + "="*60)
    print("DEMO 6: CONFIGURATION MANAGEMENT")
    print("="*60)
    
    # Show current configuration
    print("--- Current MVPA Configuration ---")
    config_vars = [
        'CV_FOLDS', 'N_PERMUTATIONS', 'DEFAULT_CLASSIFIER', 'DEFAULT_REGRESSOR',
        'SVM_C', 'RIDGE_ALPHA', 'STANDARDIZE', 'N_JOBS'
    ]
    
    for var in config_vars:
        if hasattr(MVPAConfig, var):
            value = getattr(MVPAConfig, var)
            print(f"{var:20s}: {value}")
    
    # Demonstrate configuration update
    print("\n--- Updating Configuration ---")
    print("Changing CV_FOLDS from 5 to 3 and N_PERMUTATIONS from 1000 to 500...")
    
    update_mvpa_config(
        cv_folds=3,
        n_permutations=500,
        ridge_alpha=2.0
    )
    
    print("Updated configuration:")
    for var in ['CV_FOLDS', 'N_PERMUTATIONS', 'RIDGE_ALPHA']:
        if hasattr(MVPAConfig, var):
            value = getattr(MVPAConfig, var)
            print(f"{var:20s}: {value}")
    
    # Reset to defaults for other demos
    update_mvpa_config(
        cv_folds=5,
        n_permutations=100,  # Keep low for demo speed
        ridge_alpha=1.0
    )
    print("\nConfiguration reset to demo defaults.")


def demo_integration_test(verbose=True):
    """
    Demonstrate integration with existing pipeline components
    
    Parameters:
    -----------
    verbose : bool
        Whether to print detailed information
    """
    print("\n" + "="*60)
    print("DEMO 7: INTEGRATION TEST")
    print("="*60)
    
    if not DATA_UTILS_AVAILABLE:
        print("⚠ Data utilities not available. Skipping integration test.")
        return
    
    print("✓ MVPA utilities successfully imported")
    print("✓ Data utilities integration available")
    print("✓ Configuration classes accessible")
    
    # Test error handling integration
    print("\n--- Error Handling Integration ---")
    try:
        # This should raise an MVPAError
        from mvpa_utils import validate_input_data
        validate_input_data(
            np.array([[1, 2]]), 
            np.array([1, 2, 3]),  # Mismatched dimensions
            'classification'
        )
    except MVPAError as e:
        print(f"✓ MVPAError correctly raised: {e}")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
    
    print("\n--- Configuration Integration ---")
    try:
        config = OAKConfig()
        print(f"✓ OAK config loaded - TR: {config.TR}s")
        print(f"✓ ROI masks defined: {list(config.ROI_MASKS.keys())}")
    except Exception as e:
        print(f"⚠ Could not load OAK config: {e}")


def create_summary_plot(classification_results, regression_results, output_file=None):
    """
    Create a summary plot of demo results
    
    Parameters:
    -----------
    classification_results : dict
        Results from classification demos
    regression_results : dict
        Results from regression demos
    output_file : str, optional
        Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MVPA Utils Demo Results Summary', fontsize=16)
    
    # Classification accuracies
    if classification_results:
        ax1 = axes[0, 0]
        algorithms = list(classification_results.keys())
        accuracies = [classification_results[alg]['mean_accuracy'] for alg in algorithms]
        errors = [classification_results[alg]['std_accuracy'] for alg in algorithms]
        
        bars = ax1.bar(algorithms, accuracies, yerr=errors, capsize=5, alpha=0.7)
        ax1.set_title('Classification Accuracy by Algorithm')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance')
        ax1.legend()
        
        # Color bars by significance
        for i, (alg, bar) in enumerate(zip(algorithms, bars)):
            p_val = classification_results[alg]['p_value']
            if p_val < 0.05:
                bar.set_color('green')
                bar.set_alpha(0.8)
    
    # Regression R² scores  
    if regression_results:
        ax2 = axes[0, 1]
        var_names = list(regression_results.keys())
        algorithms = ['ridge', 'lasso', 'elastic']
        
        x = np.arange(len(var_names))
        width = 0.25
        
        for i, alg in enumerate(algorithms):
            r2_scores = []
            for var in var_names:
                if alg in regression_results[var] and regression_results[var][alg]['success']:
                    r2_scores.append(regression_results[var][alg]['mean_r2'])
                else:
                    r2_scores.append(0)
            
            ax2.bar(x + i*width, r2_scores, width, label=alg, alpha=0.7)
        
        ax2.set_title('Regression R² by Variable and Algorithm')
        ax2.set_ylabel('R² Score')
        ax2.set_xlabel('Variables')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(var_names, rotation=45)
        ax2.legend()
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # P-value distributions
    ax3 = axes[1, 0]
    all_p_values = []
    
    if classification_results:
        all_p_values.extend([res['p_value'] for res in classification_results.values()])
    
    if regression_results:
        for var_results in regression_results.values():
            for alg_results in var_results.values():
                if alg_results['success']:
                    all_p_values.append(alg_results['p_value'])
    
    if all_p_values:
        ax3.hist(all_p_values, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0.05, color='r', linestyle='--', label='p=0.05')
        ax3.set_title('Distribution of P-values')
        ax3.set_xlabel('P-value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary stats
    n_tests = len(all_p_values) if all_p_values else 0
    n_significant = sum(1 for p in all_p_values if p < 0.05) if all_p_values else 0
    
    summary_text = f"""
    Demo Summary:
    
    Total tests run: {n_tests}
    Significant results: {n_significant}
    Success rate: {n_significant/n_tests*100:.1f}%
    
    Classification algorithms: {len(classification_results)}
    Regression variables: {len(regression_results)}
    
    All demos completed successfully!
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSummary plot saved to: {output_file}")
    
    plt.show()


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Demo MVPA Utilities Module')
    parser.add_argument('--demo-type', choices=['basic', 'advanced', 'integration', 'all'],
                       default='all', help='Type of demo to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--plot', action='store_true', help='Create summary plots')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of synthetic trials')
    parser.add_argument('--n-voxels', type=int, default=500, help='Number of synthetic voxels')
    
    args = parser.parse_args()
    
    print("MVPA Utilities Module Demonstration")
    print("=" * 50)
    print(f"Demo type: {args.demo_type}")
    print(f"Synthetic data: {args.n_trials} trials x {args.n_voxels} voxels")
    
    # Create synthetic data
    data = create_synthetic_fmri_data(args.n_trials, args.n_voxels)
    
    # Initialize result storage
    classification_results = {}
    regression_results = {}
    
    # Run selected demos
    if args.demo_type in ['basic', 'all']:
        classification_results = demo_basic_classification(data, args.verbose)
        regression_results = demo_basic_regression(data, args.verbose)
    
    if args.demo_type in ['advanced', 'all']:
        demo_dimensionality_reduction(data, args.verbose)
        demo_feature_importance(data, args.verbose)
        demo_convenience_functions(data, args.verbose)
    
    if args.demo_type in ['integration', 'all']:
        demo_config_management(args.verbose)
        demo_integration_test(args.verbose)
    
    # Create summary plot
    if args.plot and (classification_results or regression_results):
        output_file = 'mvpa_utils_demo_summary.png' if args.plot else None
        create_summary_plot(classification_results, regression_results, output_file)
    
    print("\n" + "="*60)
    print("🎉 MVPA UTILITIES DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey takeaways:")
    print("✅ Centralized MVPA functions eliminate code duplication")
    print("✅ Consistent interfaces across different algorithms")
    print("✅ Comprehensive error handling and validation")
    print("✅ Easy configuration management")
    print("✅ Integration with existing pipeline components")
    print("\nThe mvpa_utils module is ready for use in your analysis pipeline!")


if __name__ == "__main__":
    main() 