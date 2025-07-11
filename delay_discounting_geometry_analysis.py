#!/usr/bin/env python3
"""
Delay Discounting Neural Geometry Analysis Script

Specialized tool for analyzing neural representational geometries in delay discounting fMRI data.
Supports multiple comparison types:
- Choice types (sooner-smaller vs larger-later)
- Delay lengths (short vs long delays)
- Subjective values (chosen vs unchosen options)
- Value differences (high vs low difference between options)
- Custom continuous and categorical splits

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import CCA
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

# Import shared geometry utilities
from geometry_utils import (
    compute_manifold_alignment, compute_procrustes_alignment, compute_cca_alignment,
    compute_information_geometry_metrics, compute_kl_divergence, compute_js_divergence,
    compute_wasserstein_approximation, compute_geodesic_distances, compute_manifold_curvature,
    compute_regression_alignment
)

# NOTE: All geometric transformation functions are now consolidated in geometry_utils.py
# This eliminates the need to import from geometric_transformation_analysis.py

# Import from main pipeline if available
try:
    from delay_discounting_mvpa_pipeline import GeometryAnalysis, Config, ROIAnalysis
except ImportError:
    warnings.warn("Could not import from delay_discounting_mvpa_pipeline. Some functionality may be limited.")
    GeometryAnalysis = None
    Config = None
    ROIAnalysis = None

class DelayDiscountingGeometryAnalyzer:
    """
    Specialized neural geometry analysis tool for delay discounting experiments
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the delay discounting geometry analyzer"""
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', './dd_geometry_analysis'))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize geometry analysis if available
        if GeometryAnalysis is not None:
            try:
                pipeline_config = Config() if Config is not None else None
                self.geometry_analysis = GeometryAnalysis(pipeline_config)
            except:
                self.geometry_analysis = None
                warnings.warn("Could not initialize GeometryAnalysis from pipeline")
        else:
            self.geometry_analysis = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'output_dir': './dd_geometry_analysis',
            'n_permutations': 1000,
            'random_state': 42,
            'alpha': 0.05,
            'n_components_pca': 10,
            'n_components_mds': 5,
            'n_components_tsne': 3,
            'standardize_data': True,
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300,
            # Delay discounting specific settings
            'value_percentile_split': 50,  # For median splits of continuous variables
            'delay_short_threshold': 7,    # Days - delays <= this are "short"
            'delay_long_threshold': 30,    # Days - delays >= this are "long"
            'value_diff_percentile': 67    # Top/bottom tercile for value differences
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def load_delay_discounting_data(self, 
                                  neural_data_path: str,
                                  behavioral_data_path: str,
                                  roi_name: str = "ROI") -> Dict:
        """
        Load neural data and behavioral data for delay discounting analysis
        
        Parameters:
        -----------
        neural_data_path : str
            Path to neural data (trials x voxels)
        behavioral_data_path : str  
            Path to behavioral data CSV with trial information
        roi_name : str
            Name of the ROI being analyzed
            
        Returns:
        --------
        dict : Complete dataset with neural and behavioral data
        """
        # Load neural data
        if neural_data_path.endswith('.npy'):
            neural_data = np.load(neural_data_path)
        elif neural_data_path.endswith('.csv'):
            neural_data = pd.read_csv(neural_data_path).values
        else:
            raise ValueError(f"Unsupported neural data format: {neural_data_path}")
        
        # Load behavioral data
        behavioral_data = pd.read_csv(behavioral_data_path)
        
        # Standardize neural data if requested
        if self.config['standardize_data']:
            scaler = StandardScaler()
            neural_data = scaler.fit_transform(neural_data)
        
        # Validate data alignment
        if len(neural_data) != len(behavioral_data):
            raise ValueError(f"Neural data ({len(neural_data)} trials) and behavioral data "
                           f"({len(behavioral_data)} trials) have different lengths")
        
        return {
            'neural_data': neural_data,
            'behavioral_data': behavioral_data,
            'roi_name': roi_name,
            'n_trials': len(neural_data),
            'n_voxels': neural_data.shape[1]
        }
    
    def create_choice_comparison(self, data: Dict) -> Dict:
        """
        Create comparison between sooner-smaller vs larger-later choices
        
        Expected behavioral columns: 'choice' (0=SS, 1=LL)
        """
        behavioral_data = data['behavioral_data']
        
        if 'choice' not in behavioral_data.columns:
            raise ValueError("Behavioral data must contain 'choice' column (0=SS, 1=LL)")
        
        choice_labels = behavioral_data['choice'].values
        condition_names = ['Sooner-Smaller', 'Larger-Later']
        
        return {
            'comparison_type': 'choice',
            'labels': choice_labels,
            'condition_names': condition_names,
            'description': 'Sooner-Smaller vs Larger-Later choices'
        }
    
    def create_delay_comparison(self, data: Dict, 
                              comparison_type: str = 'short_vs_long') -> Dict:
        """
        Create comparison based on delay lengths
        
        Parameters:
        -----------
        comparison_type : str
            'short_vs_long': Short vs long delays (based on thresholds)
            'immediate_vs_delayed': 0-day vs any delay
            'median_split': Median split of all delays
        
        Expected behavioral columns: 'delay_days'
        """
        behavioral_data = data['behavioral_data']
        
        if 'delay_days' not in behavioral_data.columns:
            raise ValueError("Behavioral data must contain 'delay_days' column")
        
        delays = behavioral_data['delay_days'].values
        
        if comparison_type == 'short_vs_long':
            short_threshold = self.config['delay_short_threshold']
            long_threshold = self.config['delay_long_threshold']
            
            # Create labels: 0=short, 1=long, exclude medium delays
            labels = np.full(len(delays), -1)  # -1 for excluded trials
            labels[delays <= short_threshold] = 0
            labels[delays >= long_threshold] = 1
            
            condition_names = [f'Short (≤{short_threshold}d)', f'Long (≥{long_threshold}d)']
            description = f'Short (≤{short_threshold} days) vs Long (≥{long_threshold} days) delays'
            
        elif comparison_type == 'immediate_vs_delayed':
            labels = (delays > 0).astype(int)
            condition_names = ['Immediate (0d)', 'Delayed (>0d)']
            description = 'Immediate (0 days) vs Delayed (>0 days)'
            
        elif comparison_type == 'median_split':
            median_delay = np.median(delays)
            labels = (delays > median_delay).astype(int)
            condition_names = [f'Short (≤{median_delay:.1f}d)', f'Long (>{median_delay:.1f}d)']
            description = f'Median split delays (threshold: {median_delay:.1f} days)'
            
        else:
            raise ValueError(f"Unknown delay comparison type: {comparison_type}")
        
        return {
            'comparison_type': f'delay_{comparison_type}',
            'labels': labels,
            'condition_names': condition_names,
            'description': description
        }
    
    def create_subjective_value_comparison(self, data: Dict, 
                                         value_type: str = 'chosen',
                                         split_method: str = 'median') -> Dict:
        """
        Create comparison based on subjective values
        
        Parameters:
        -----------
        value_type : str
            'chosen': Subjective value of chosen option
            'unchosen': Subjective value of unchosen option
            'difference': Difference in subjective values (chosen - unchosen)
        split_method : str
            'median': Median split
            'terciles': Top vs bottom tercile
        
        Expected behavioral columns: 'chosen_sv', 'unchosen_sv'
        """
        behavioral_data = data['behavioral_data']
        
        if value_type == 'chosen':
            if 'chosen_sv' not in behavioral_data.columns:
                raise ValueError("Behavioral data must contain 'chosen_sv' column")
            values = behavioral_data['chosen_sv'].values
            value_name = 'Chosen Option SV'
            
        elif value_type == 'unchosen':
            if 'unchosen_sv' not in behavioral_data.columns:
                raise ValueError("Behavioral data must contain 'unchosen_sv' column")
            values = behavioral_data['unchosen_sv'].values
            value_name = 'Unchosen Option SV'
            
        elif value_type == 'difference':
            required_cols = ['chosen_sv', 'unchosen_sv']
            missing_cols = [col for col in required_cols if col not in behavioral_data.columns]
            if missing_cols:
                raise ValueError(f"Behavioral data must contain columns: {missing_cols}")
            values = behavioral_data['chosen_sv'].values - behavioral_data['unchosen_sv'].values
            value_name = 'SV Difference (Chosen - Unchosen)'
            
        else:
            raise ValueError(f"Unknown value type: {value_type}")
        
        # Create labels based on split method
        if split_method == 'median':
            median_val = np.median(values)
            labels = (values > median_val).astype(int)
            condition_names = [f'Low {value_name}', f'High {value_name}']
            description = f'Median split of {value_name} (threshold: {median_val:.3f})'
            
        elif split_method == 'terciles':
            lower_tercile = np.percentile(values, 33.33)
            upper_tercile = np.percentile(values, 66.67)
            
            # Create labels: 0=bottom tercile, 1=top tercile, -1=middle (excluded)
            labels = np.full(len(values), -1)
            labels[values <= lower_tercile] = 0
            labels[values >= upper_tercile] = 1
            
            condition_names = [f'Low {value_name}', f'High {value_name}']
            description = f'Tercile split of {value_name} (bottom vs top tercile)'
            
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        return {
            'comparison_type': f'sv_{value_type}_{split_method}',
            'labels': labels,
            'condition_names': condition_names,
            'description': description
        }
    
    def create_value_difference_comparison(self, data: Dict,
                                         split_method: str = 'terciles') -> Dict:
        """
        Create comparison based on absolute difference in subjective values
        (how similar the two options are in value)
        
        Expected behavioral columns: 'chosen_sv', 'unchosen_sv'
        """
        behavioral_data = data['behavioral_data']
        
        required_cols = ['chosen_sv', 'unchosen_sv']
        missing_cols = [col for col in required_cols if col not in behavioral_data.columns]
        if missing_cols:
            raise ValueError(f"Behavioral data must contain columns: {missing_cols}")
        
        # Calculate absolute difference
        abs_diff = np.abs(behavioral_data['chosen_sv'].values - 
                         behavioral_data['unchosen_sv'].values)
        
        if split_method == 'median':
            median_diff = np.median(abs_diff)
            labels = (abs_diff > median_diff).astype(int)
            condition_names = ['Similar Values', 'Different Values']
            description = f'Median split of |SV difference| (threshold: {median_diff:.3f})'
            
        elif split_method == 'terciles':
            lower_tercile = np.percentile(abs_diff, 33.33)
            upper_tercile = np.percentile(abs_diff, 66.67)
            
            # Create labels: 0=small difference, 1=large difference, -1=medium (excluded)
            labels = np.full(len(abs_diff), -1)
            labels[abs_diff <= lower_tercile] = 0
            labels[abs_diff >= upper_tercile] = 1
            
            condition_names = ['Similar Values', 'Different Values']
            description = f'Tercile split of |SV difference| (small vs large differences)'
            
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        return {
            'comparison_type': f'value_diff_{split_method}',
            'labels': labels,
            'condition_names': condition_names,
            'description': description
        }
    
    def run_comprehensive_dd_analysis(self, data: Dict,
                                    comparisons: List[str] = None) -> Dict:
        """
        Run comprehensive delay discounting geometry analysis
        
        Parameters:
        -----------
        data : dict
            Data loaded from load_delay_discounting_data
        comparisons : list
            List of comparison types to run. If None, runs all available.
            Options: ['choice', 'delay_short_vs_long', 'delay_immediate_vs_delayed',
                     'sv_chosen_median', 'sv_unchosen_median', 'sv_difference_median',
                     'value_diff_terciles']
        
        Returns:
        --------
        dict : Results for all comparisons
        """
        if comparisons is None:
            comparisons = [
                'choice',
                'delay_short_vs_long',
                'delay_immediate_vs_delayed', 
                'sv_chosen_median',
                'sv_unchosen_median',
                'sv_difference_median',
                'value_diff_terciles'
            ]
        
        all_results = {}
        neural_data = data['neural_data']
        
        print(f"Running delay discounting geometry analysis for {data['roi_name']}")
        print(f"Neural data shape: {neural_data.shape}")
        print(f"Running {len(comparisons)} comparisons...")
        
        for comparison_name in comparisons:
            print(f"\n--- Running {comparison_name} comparison ---")
            
            try:
                # Create comparison
                if comparison_name == 'choice':
                    comparison = self.create_choice_comparison(data)
                elif comparison_name.startswith('delay_'):
                    delay_type = comparison_name.replace('delay_', '')
                    comparison = self.create_delay_comparison(data, delay_type)
                elif comparison_name.startswith('sv_'):
                    parts = comparison_name.split('_')
                    value_type = parts[1]
                    split_method = parts[2]
                    comparison = self.create_subjective_value_comparison(data, value_type, split_method)
                elif comparison_name.startswith('value_diff_'):
                    split_method = comparison_name.replace('value_diff_', '')
                    comparison = self.create_value_difference_comparison(data, split_method)
                else:
                    print(f"Unknown comparison type: {comparison_name}")
                    continue
                
                # Filter out excluded trials (label = -1)
                labels = comparison['labels']
                if np.any(labels == -1):
                    included_mask = labels != -1
                    filtered_neural_data = neural_data[included_mask]
                    filtered_labels = labels[included_mask]
                    print(f"Included {np.sum(included_mask)}/{len(labels)} trials "
                          f"(excluded {np.sum(~included_mask)} trials)")
                else:
                    filtered_neural_data = neural_data
                    filtered_labels = labels
                
                # Check if we have enough trials in each condition
                unique_labels, counts = np.unique(filtered_labels, return_counts=True)
                if len(unique_labels) < 2:
                    print(f"Skipping {comparison_name}: Only one condition found")
                    continue
                elif np.min(counts) < 10:
                    print(f"Warning for {comparison_name}: Small sample size in some conditions: {counts}")
                
                print(f"Description: {comparison['description']}")
                print(f"Conditions: {comparison['condition_names']}")
                print(f"Trial counts: {dict(zip(unique_labels, counts))}")
                
                # Run geometry analysis
                results = self.run_geometry_comparison(
                    filtered_neural_data,
                    filtered_labels,
                    comparison['condition_names'],
                    methods=['pca', 'mds']
                )
                
                # Run advanced geometry analysis
                print(f"  Running advanced geometry analysis for {comparison_name}...")
                advanced_results = self.run_advanced_geometry_analysis(
                    filtered_neural_data,
                    filtered_labels,
                    comparison['condition_names']
                )
                results['advanced_geometry'] = advanced_results
                
                # Add comparison metadata
                results['comparison_info'] = comparison
                all_results[comparison_name] = results
                
            except Exception as e:
                print(f"Error in {comparison_name}: {str(e)}")
                continue
        
        return all_results
    
    def run_geometry_comparison(self, neural_data: np.array,
                              condition_labels: np.array,
                              condition_names: List[str],
                              methods: List[str] = ['pca', 'mds']) -> Dict:
        """Run geometry comparison analysis (adapted from original script)"""
        results = {}
        
        # 1. RSA on original data  
        rsa_results = self.compute_representational_similarity(neural_data, condition_labels)
        results['rsa'] = rsa_results
        
        # 2. Dimensionality analysis
        dim_results = self.dimensionality_analysis(neural_data, condition_labels)
        results['dimensionality'] = dim_results
        
        # 3. Analysis for each reduction method
        for method in methods:
            # Dimensionality reduction
            embedding, reducer = self.dimensionality_reduction(neural_data, method=method)
            
            # Distance analysis
            distance_results = self.distance_analysis(embedding, condition_labels)
            
            # RSA on embedding
            embedding_rsa = self.compute_representational_similarity(embedding, condition_labels)
            
            results[method] = {
                'embedding': embedding,
                'reducer': reducer,
                'distance_analysis': distance_results,
                'rsa': embedding_rsa
            }
        
        return results
    
    def dimensionality_reduction(self, neural_data: np.array, method: str = 'pca',
                               n_components: Optional[int] = None) -> Tuple[np.array, Any]:
        """Perform dimensionality reduction"""
        if method == 'pca':
            n_comp = n_components or self.config['n_components_pca']
            reducer = PCA(n_components=n_comp, random_state=self.config['random_state'])
        elif method == 'mds':
            n_comp = n_components or self.config['n_components_mds']
            reducer = MDS(n_components=n_comp, random_state=self.config['random_state'])
        elif method == 'tsne':
            n_comp = n_components or self.config['n_components_tsne']
            reducer = TSNE(n_components=n_comp, random_state=self.config['random_state'])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced_data = reducer.fit_transform(neural_data)
        return reduced_data, reducer
    
    def compute_representational_similarity(self, neural_data: np.array,
                                          condition_labels: np.array,
                                          metric: str = 'correlation') -> Dict:
        """Compute RSA"""
        # Compute pairwise distances
        if metric == 'correlation':
            distances = 1 - np.corrcoef(neural_data)
        else:
            distances = pairwise_distances(neural_data, metric=metric)
        
        # Create condition model matrix
        condition_model = np.zeros((len(condition_labels), len(condition_labels)))
        for i, label_i in enumerate(condition_labels):
            for j, label_j in enumerate(condition_labels):
                condition_model[i, j] = 1 if label_i == label_j else 0
        
        # Compute RSA correlation
        triu_indices = np.triu_indices(distances.shape[0], k=1)
        distance_vector = distances[triu_indices]
        model_vector = condition_model[triu_indices]
        
        rsa_correlation, p_value = stats.pearsonr(distance_vector, model_vector)
        
        return {
            'rsa_correlation': rsa_correlation,
            'p_value': p_value,
            'distance_matrix': distances,
            'model_matrix': condition_model,
            'interpretation': self._interpret_rsa_correlation(rsa_correlation, p_value)
        }
    
    def distance_analysis(self, embedding: np.array, condition_labels: np.array) -> Dict:
        """Analyze distances within and between conditions"""
        distances = pairwise_distances(embedding)
        
        within_distances = []
        between_distances = []
        
        for i in range(len(condition_labels)):
            for j in range(i + 1, len(condition_labels)):
                if condition_labels[i] == condition_labels[j]:
                    within_distances.append(distances[i, j])
                else:
                    between_distances.append(distances[i, j])
        
        within_distances = np.array(within_distances)
        between_distances = np.array(between_distances)
        
        mean_within = np.mean(within_distances)
        mean_between = np.mean(between_distances)
        separation_ratio = mean_between / mean_within if mean_within > 0 else np.inf
        
        # Permutation test
        observed_diff = mean_between - mean_within
        n_perms = self.config['n_permutations']
        perm_diffs = []
        
        for _ in range(n_perms):
            perm_labels = np.random.permutation(condition_labels)
            perm_within = []
            perm_between = []
            
            for i in range(len(perm_labels)):
                for j in range(i + 1, len(perm_labels)):
                    if perm_labels[i] == perm_labels[j]:
                        perm_within.append(distances[i, j])
                    else:
                        perm_between.append(distances[i, j])
            
            perm_diff = np.mean(perm_between) - np.mean(perm_within)
            perm_diffs.append(perm_diff)
        
        p_value = np.mean(np.array(perm_diffs) >= observed_diff)
        
        return {
            'mean_within_condition_distance': mean_within,
            'mean_between_condition_distance': mean_between,
            'separation_ratio': separation_ratio,
            'separation_p_value': p_value,
            'within_distances': within_distances,
            'between_distances': between_distances,
            'interpretation': self._interpret_separation_ratio(separation_ratio, p_value)
        }
    
    def dimensionality_analysis(self, neural_data: np.array, condition_labels: np.array) -> Dict:
        """Analyze intrinsic dimensionality"""
        results = {}
        
        for condition in np.unique(condition_labels):
            condition_data = neural_data[condition_labels == condition]
            
            pca = PCA()
            pca.fit(condition_data)
            
            explained_var_ratio = pca.explained_variance_ratio_
            cumsum_var = np.cumsum(explained_var_ratio)
            
            dim_80 = np.argmax(cumsum_var >= 0.8) + 1
            dim_90 = np.argmax(cumsum_var >= 0.9) + 1
            
            participation_ratio = (np.sum(explained_var_ratio) ** 2) / np.sum(explained_var_ratio ** 2)
            
            results[f"condition_{condition}"] = {
                'explained_variance_ratio': explained_var_ratio,
                'cumulative_variance': cumsum_var,
                'intrinsic_dimensionality_80': dim_80,
                'intrinsic_dimensionality_90': dim_90,
                'participation_ratio': participation_ratio
            }
        
        return results
    
    # ===== ADVANCED GEOMETRY ANALYSIS METHODS =====
    # (From geometric_transformation_analysis.py)
    
    # NOTE: Geometry utility functions are now imported from geometry_utils module
    # to eliminate code duplication. The following functions are available:
    # - compute_manifold_alignment()
    # - compute_procrustes_alignment() 
    # - compute_cca_alignment()
    # - compute_geodesic_distances()
    # - compute_manifold_curvature()
    # - compute_information_geometry_metrics()
    # - compute_kl_divergence()
    # - compute_js_divergence()
    # - compute_wasserstein_approximation()
    
    def run_advanced_geometry_analysis(self, neural_data: np.array,
                                     condition_labels: np.array,
                                     condition_names: List[str]) -> Dict:
        """Run advanced geometric transformation analysis"""
        results = {}
        
        # Separate data by condition
        unique_labels = np.unique(condition_labels)
        condition_data = {}
        for label in unique_labels:
            condition_data[label] = neural_data[condition_labels == label]
        
        if len(unique_labels) >= 2:
            label1, label2 = unique_labels[:2]
            X1, X2 = condition_data[label1], condition_data[label2]
            
            print("  Computing manifold alignment...")
            # Manifold alignment
            results['manifold_alignment'] = {
                'procrustes': compute_manifold_alignment(X1, X2, method='procrustes'),
                'cca': compute_manifold_alignment(X1, X2, method='cca')
            }
            
            print("  Computing information geometry metrics...")
            # Information geometry metrics
            results['information_geometry'] = compute_information_geometry_metrics(X1, X2)
            
            print("  Computing geodesic distances...")
            # Geodesic distance analysis
            results['geodesic_analysis'] = {
                'condition_1': compute_geodesic_distances(X1),
                'condition_2': compute_geodesic_distances(X2)
            }
            
            print("  Computing manifold curvature...")
            # Manifold curvature
            curvature_1 = compute_manifold_curvature(X1)
            curvature_2 = compute_manifold_curvature(X2)
            
            results['curvature_analysis'] = {
                'condition_1': {
                    'curvatures': curvature_1,
                    'mean_curvature': np.mean(curvature_1),
                    'curvature_variance': np.var(curvature_1)
                },
                'condition_2': {
                    'curvatures': curvature_2,
                    'mean_curvature': np.mean(curvature_2),
                    'curvature_variance': np.var(curvature_2)
                }
            }
        
        return results
    
    def create_delay_trajectory_analysis(self, data: Dict, 
                                       min_trials_per_delay: int = 5) -> Dict:
        """
        Create delay_results structure for geometric transformation analysis
        
        Parameters:
        -----------
        data : dict
            Data loaded from load_delay_discounting_data
        min_trials_per_delay : int
            Minimum number of trials required per delay condition
            
        Returns:
        --------
        dict : delay_results structure for geometric transformation analysis
        """
        behavioral_data = data['behavioral_data']
        neural_data = data['neural_data']
        
        if 'delay_days' not in behavioral_data.columns:
            raise ValueError("Behavioral data must contain 'delay_days' column")
        
        delays = behavioral_data['delay_days'].values
        unique_delays = np.unique(delays)
        
        # Filter delays with sufficient trials
        valid_delays = []
        delay_patterns = {}
        
        for delay in unique_delays:
            delay_mask = delays == delay
            n_trials = np.sum(delay_mask)
            
            if n_trials >= min_trials_per_delay:
                valid_delays.append(delay)
                
                # Get neural patterns for this delay
                delay_neural_data = neural_data[delay_mask]
                
                # Apply PCA for embedding
                embedding, reducer = self.dimensionality_reduction(
                    delay_neural_data, method='pca', n_components=10
                )
                
                delay_patterns[delay] = {
                    'patterns': embedding,
                    'n_trials': n_trials,
                    'neural_data': delay_neural_data
                }
        
        if len(valid_delays) < 3:
            raise ValueError(f"Need at least 3 delay conditions, found {len(valid_delays)}")
        
        # Sort delays for trajectory analysis
        valid_delays.sort()
        
        return {
            'delays': valid_delays,
            'embeddings': delay_patterns,
            'roi_name': data['roi_name'],
            'total_trials': len(delays),
            'delay_distribution': {delay: np.sum(delays == delay) for delay in unique_delays}
        }
    
    def run_trajectory_analysis(self, data: Dict, 
                              save_visualizations: bool = True) -> Dict:
        """
        Run comprehensive geometric transformation trajectory analysis
        
        Parameters:
        -----------
        data : dict
            Data loaded from load_delay_discounting_data
        save_visualizations : bool
            Whether to create and save trajectory visualizations
            
        Returns:
        --------
        dict : Trajectory analysis results
        """
        print("Creating delay trajectory analysis structure...")
        
        try:
            delay_results = self.create_delay_trajectory_analysis(data)
        except ValueError as e:
            print(f"Cannot run trajectory analysis: {e}")
            return {'error': str(e)}
        
        delays = delay_results['delays']
        embeddings = delay_results['embeddings']
        roi_name = delay_results['roi_name']
        
        print(f"Trajectory analysis for {roi_name} with {len(delays)} delay conditions")
        print(f"Delays: {delays}")
        
        results = {
            'delay_results': delay_results,
            'trajectory_metrics': {}
        }
        
        # 1. Centroid trajectory analysis
        print("  Computing centroid trajectories...")
        centroids = []
        for delay in delays:
            patterns = embeddings[delay]['patterns']
            centroid = np.mean(patterns, axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Compute trajectory velocity (rate of change between consecutive delays)
        velocities = []
        for i in range(len(centroids) - 1):
            velocity = np.linalg.norm(centroids[i+1] - centroids[i])
            velocities.append(velocity)
        
        results['trajectory_metrics']['centroids'] = centroids
        results['trajectory_metrics']['velocities'] = velocities
        results['trajectory_metrics']['mean_velocity'] = np.mean(velocities)
        results['trajectory_metrics']['velocity_variability'] = np.std(velocities)
        
        # 2. Pairwise manifold alignment analysis
        print("  Computing pairwise manifold alignments...")
        alignment_matrix = np.zeros((len(delays), len(delays)))
        
        for i, delay1 in enumerate(delays):
            for j, delay2 in enumerate(delays):
                if i != j:
                    X1 = embeddings[delay1]['patterns']
                    X2 = embeddings[delay2]['patterns']
                    
                    # Use procrustes alignment
                    alignment = compute_manifold_alignment(X1, X2, method='procrustes')
                    alignment_matrix[i, j] = alignment['alignment_quality']
                    
                    # Also try regression alignment (unique to geometric_transformation_analysis)
                    if i == 0 and j == 1:  # Just for first pair as example
                        reg_alignment = compute_regression_alignment(X1, X2)
                        results['trajectory_metrics']['regression_alignment_example'] = reg_alignment
                else:
                    alignment_matrix[i, j] = 1.0
        
        results['trajectory_metrics']['alignment_matrix'] = alignment_matrix
        results['trajectory_metrics']['mean_alignment_quality'] = np.mean(alignment_matrix[alignment_matrix < 1.0])
        
        # 3. Information geometry across delays
        print("  Computing information geometry metrics...")
        js_divergences = []
        wasserstein_distances = []
        
        for i in range(len(delays) - 1):
            X1 = embeddings[delays[i]]['patterns']
            X2 = embeddings[delays[i+1]]['patterns']
            
            info_metrics = compute_information_geometry_metrics(X1, X2)
            js_divergences.append(info_metrics['js_divergence'])
            wasserstein_distances.append(info_metrics['wasserstein_distance'])
        
        results['trajectory_metrics']['js_divergences'] = js_divergences
        results['trajectory_metrics']['wasserstein_distances'] = wasserstein_distances
        results['trajectory_metrics']['mean_js_divergence'] = np.mean(js_divergences)
        results['trajectory_metrics']['mean_wasserstein_distance'] = np.mean(wasserstein_distances)
        
        # 4. Curvature evolution
        print("  Computing manifold curvature evolution...")
        mean_curvatures = []
        curvature_variances = []
        
        for delay in delays:
            patterns = embeddings[delay]['patterns']
            curvatures = compute_manifold_curvature(patterns)
            mean_curvatures.append(np.mean(curvatures))
            curvature_variances.append(np.var(curvatures))
        
        results['trajectory_metrics']['mean_curvatures'] = mean_curvatures
        results['trajectory_metrics']['curvature_variances'] = curvature_variances
        results['trajectory_metrics']['curvature_evolution_slope'] = np.polyfit(delays, mean_curvatures, 1)[0]
        
        # 5. Effective dimensionality evolution
        print("  Computing dimensionality evolution...")
        effective_dims = []
        
        for delay in delays:
            patterns = embeddings[delay]['patterns']
            pca = PCA()
            pca.fit(patterns)
            
            # Compute effective dimensionality (participation ratio)
            eigenvals = pca.explained_variance_
            effective_dim = (np.sum(eigenvals)**2) / np.sum(eigenvals**2)
            effective_dims.append(effective_dim)
        
        results['trajectory_metrics']['effective_dimensions'] = effective_dims
        results['trajectory_metrics']['dimensionality_evolution_slope'] = np.polyfit(delays, effective_dims, 1)[0]
        
        # 6. Create custom trajectory visualizations 
        if save_visualizations:
            print("  Creating trajectory visualizations...")
            save_path = self.output_dir / "trajectory_visualizations"
            save_path.mkdir(exist_ok=True)
            
            try:
                self._create_trajectory_visualizations(delay_results, results['trajectory_metrics'], str(save_path))
                results['visualization_path'] = str(save_path / f"{roi_name}_trajectory_analysis.png")
            except Exception as e:
                print(f"    Warning: Visualization failed: {e}")
                results['visualization_error'] = str(e)
        
        return results
    
    def run_comprehensive_trajectory_analysis(self, data: Dict) -> Dict:
        """
        Run both standard delay discounting analysis AND trajectory analysis
        
        Returns combined results from both analysis types
        """
        print("="*60)
        print("COMPREHENSIVE DELAY DISCOUNTING + TRAJECTORY ANALYSIS")
        print("="*60)
        
        # 1. Run standard delay discounting analysis
        print("\n1. Running standard delay discounting geometry analysis...")
        standard_results = self.run_comprehensive_dd_analysis(data)
        
        # 2. Run trajectory analysis
        print("\n2. Running trajectory analysis...")
        trajectory_results = self.run_trajectory_analysis(data)
        
        # 3. Combine results
        combined_results = {
            'roi_name': data['roi_name'],
            'analysis_type': 'comprehensive_dd_trajectory',
            'standard_comparisons': standard_results,
            'trajectory_analysis': trajectory_results,
            'summary': self._create_trajectory_summary(standard_results, trajectory_results)
        }
        
        return combined_results
    
    def _create_trajectory_summary(self, standard_results: Dict, trajectory_results: Dict) -> Dict:
        """Create a summary combining both analysis types"""
        summary = {
            'n_standard_comparisons': len(standard_results),
            'standard_comparisons_list': list(standard_results.keys())
        }
        
        if 'error' not in trajectory_results:
            traj_metrics = trajectory_results['trajectory_metrics']
            summary.update({
                'n_delay_conditions': len(trajectory_results['delay_results']['delays']),
                'delay_range': f"{min(trajectory_results['delay_results']['delays'])}-{max(trajectory_results['delay_results']['delays'])} days",
                'mean_trajectory_velocity': traj_metrics['mean_velocity'],
                'mean_alignment_quality': traj_metrics['mean_alignment_quality'],
                'curvature_evolution_trend': 'increasing' if traj_metrics['curvature_evolution_slope'] > 0 else 'decreasing',
                'dimensionality_evolution_trend': 'increasing' if traj_metrics['dimensionality_evolution_slope'] > 0 else 'decreasing'
            })
        else:
            summary['trajectory_analysis_error'] = trajectory_results['error']
        
        return summary
    
    def _create_trajectory_visualizations(self, delay_results: Dict, trajectory_metrics: Dict, save_path: str):
        """Create custom trajectory visualizations"""
        delays = delay_results['delays']
        embeddings = delay_results['embeddings']
        roi_name = delay_results['roi_name']
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 3D trajectory of centroids
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        centroids = trajectory_metrics['centroids']
        
        # Plot trajectory
        ax1.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                 'o-', linewidth=3, markersize=8)
        
        # Color code by delay
        colors = plt.cm.viridis(np.linspace(0, 1, len(delays)))
        for i, (delay, color) in enumerate(zip(delays, colors)):
            ax1.scatter(centroids[i, 0], centroids[i, 1], centroids[i, 2], 
                       c=[color], s=100, label=f'{delay}d')
        
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        ax1.set_title(f'{roi_name}: Centroid Trajectory')
        ax1.legend()
        
        # 2. Manifold curvature across delays
        ax2 = fig.add_subplot(2, 3, 2)
        
        mean_curvatures = trajectory_metrics['mean_curvatures']
        ax2.plot(delays, mean_curvatures, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Delay (days)')
        ax2.set_ylabel('Mean Curvature')
        ax2.set_title('Manifold Curvature vs Delay')
        ax2.grid(True, alpha=0.3)
        
        # 3. Pairwise alignment quality heatmap
        ax3 = fig.add_subplot(2, 3, 3)
        
        alignment_matrix = trajectory_metrics['alignment_matrix']
        im = ax3.imshow(alignment_matrix, cmap='viridis', vmin=0, vmax=1)
        ax3.set_xticks(range(len(delays)))
        ax3.set_yticks(range(len(delays)))
        ax3.set_xticklabels([f'{d}d' for d in delays])
        ax3.set_yticklabels([f'{d}d' for d in delays])
        ax3.set_title('Pairwise Alignment Quality')
        plt.colorbar(im, ax=ax3)
        
        # 4. Information geometry metrics
        ax4 = fig.add_subplot(2, 3, 4)
        
        js_divergences = trajectory_metrics['js_divergences']
        delay_pairs = [f'{delays[i]}-{delays[i+1]}' for i in range(len(delays)-1)]
        ax4.bar(range(len(js_divergences)), js_divergences)
        ax4.set_xticks(range(len(delay_pairs)))
        ax4.set_xticklabels(delay_pairs, rotation=45)
        ax4.set_ylabel('JS Divergence')
        ax4.set_title('Information Distance Between Consecutive Delays')
        
        # 5. Trajectory velocity
        ax5 = fig.add_subplot(2, 3, 5)
        
        velocities = trajectory_metrics['velocities']
        if len(velocities) > 0:
            ax5.plot(delays[1:], velocities, 'o-', linewidth=2, markersize=8, color='red')
            ax5.set_xlabel('Delay (days)')
            ax5.set_ylabel('Centroid Velocity')
            ax5.set_title('Rate of Geometric Change')
            ax5.grid(True, alpha=0.3)
        
        # 6. Dimensionality evolution
        ax6 = fig.add_subplot(2, 3, 6)
        
        effective_dims = trajectory_metrics['effective_dimensions']
        ax6.plot(delays, effective_dims, 'o-', linewidth=2, markersize=8, color='purple')
        ax6.set_xlabel('Delay (days)')
        ax6.set_ylabel('Effective Dimensionality')
        ax6.set_title('Representational Dimensionality vs Delay')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/{roi_name}_trajectory_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self, all_results: Dict, data: Dict) -> str:
        """Create a comprehensive summary report"""
        report = []
        report.append("=" * 80)
        report.append("DELAY DISCOUNTING NEURAL GEOMETRY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"ROI: {data['roi_name']}")
        report.append(f"Neural data shape: {data['neural_data'].shape}")
        report.append(f"Total trials: {data['n_trials']}")
        report.append("")
        
        for comparison_name, results in all_results.items():
            if 'comparison_info' not in results:
                continue
                
            comp_info = results['comparison_info']
            report.append(f"--- {comparison_name.upper()} COMPARISON ---")
            report.append(f"Description: {comp_info['description']}")
            report.append(f"Conditions: {comp_info['condition_names']}")
            report.append("")
            
            # RSA results
            if 'rsa' in results:
                rsa = results['rsa']
                report.append(f"Representational Similarity Analysis:")
                report.append(f"  Correlation: {rsa['rsa_correlation']:.3f}")
                report.append(f"  P-value: {rsa['p_value']:.3f}")
                report.append(f"  Interpretation: {rsa['interpretation']}")
                report.append("")
            
            # Distance analysis for each method
            for method in ['pca', 'mds']:
                if method in results and 'distance_analysis' in results[method]:
                    dist = results[method]['distance_analysis']
                    report.append(f"{method.upper()} Distance Analysis:")
                    report.append(f"  Separation ratio: {dist['separation_ratio']:.3f}")
                    report.append(f"  P-value: {dist['separation_p_value']:.3f}")
                    report.append(f"  Interpretation: {dist['interpretation']}")
                    report.append("")
            
            report.append("-" * 60)
            report.append("")
        
        return "\n".join(report)
    
    def save_all_results(self, all_results: Dict, data: Dict):
        """Save all results and create summary report"""
        
        # Handle different result structures (standard vs comprehensive)
        if 'analysis_type' in all_results and all_results['analysis_type'] == 'comprehensive_dd_trajectory':
            # Comprehensive analysis results structure
            
            # Save standard comparisons
            if 'standard_comparisons' in all_results:
                standard_results = all_results['standard_comparisons']
                for comparison_name, results in standard_results.items():
                    if isinstance(results, dict):  # Ensure it's a dict
                        self.save_results(results, f"{data['roi_name']}_{comparison_name}_results.json")
                
                # Create advanced geometry visualizations for standard results
                print("\nCreating advanced geometry visualizations...")
                self.visualize_advanced_geometry_results(standard_results, data['roi_name'])
                
                # Create summary report for standard results
                summary_report = self.create_summary_report(standard_results, data)
                
            # Save trajectory analysis results
            if 'trajectory_analysis' in all_results:
                trajectory_results = all_results['trajectory_analysis']
                if isinstance(trajectory_results, dict) and 'error' not in trajectory_results:
                    self.save_results(trajectory_results, f"{data['roi_name']}_trajectory_analysis_results.json")
                    
                    # Add trajectory summary to report
                    summary_report += "\n\n" + "="*60 + "\n"
                    summary_report += "TRAJECTORY ANALYSIS SUMMARY\n"
                    summary_report += "="*60 + "\n"
                    
                    if 'summary' in all_results:
                        traj_summary = all_results['summary']
                        if 'n_delay_conditions' in traj_summary:
                            summary_report += f"Delay conditions: {traj_summary['n_delay_conditions']}\n"
                            summary_report += f"Delay range: {traj_summary['delay_range']}\n"
                            summary_report += f"Mean trajectory velocity: {traj_summary['mean_trajectory_velocity']:.3f}\n"
                            summary_report += f"Mean alignment quality: {traj_summary['mean_alignment_quality']:.3f}\n"
                        else:
                            summary_report += f"Trajectory analysis error: {traj_summary.get('trajectory_analysis_error', 'Unknown error')}\n"
            
            # Save comprehensive results
            self.save_results(all_results, f"{data['roi_name']}_comprehensive_results.json")
            
        else:
            # Standard analysis results structure
            for comparison_name, results in all_results.items():
                if isinstance(results, dict):  # Ensure it's a dict
                    self.save_results(results, f"{data['roi_name']}_{comparison_name}_results.json")
            
            # Create advanced geometry visualizations
            print("\nCreating advanced geometry visualizations...")
            self.visualize_advanced_geometry_results(all_results, data['roi_name'])
            
            # Create and save summary report
            summary_report = self.create_summary_report(all_results, data)
        
        # Save summary report
        report_file = self.output_dir / f"{data['roi_name']}_summary_report.txt"
        with open(report_file, 'w') as f:
            f.write(summary_report)
        
        print(f"\nSummary report saved to: {report_file}")
        print("\nSUMMARY REPORT:")
        print(summary_report)
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file with improved numpy array handling"""
        
        def make_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable objects"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__') and not isinstance(obj, (int, float, str, bool, type(None))):
                # Skip objects with __dict__ (like sklearn objects)
                return f"<{obj.__class__.__name__} object - not serializable>"
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()  # Convert numpy scalars to Python scalars
            else:
                return obj
        
        try:
            serializable_results = make_serializable(results)
            output_file = self.output_dir / filename
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save {filename}: {e}")
            # Save a simplified version
            simplified_results = {}
            for key, value in results.items():
                try:
                    simplified_results[key] = make_serializable(value)
                except:
                    simplified_results[key] = f"<Could not serialize {type(value).__name__}>"
            
            try:
                output_file = self.output_dir / filename
                with open(output_file, 'w') as f:
                    json.dump(simplified_results, f, indent=2)
            except Exception as e2:
                print(f"Error: Could not save even simplified results for {filename}: {e2}")
    
    def visualize_advanced_geometry_results(self, all_results: Dict, roi_name: str = "ROI"):
        """Create advanced visualizations of geometric transformation results"""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for comparison_name, comparison_results in all_results.items():
            if 'advanced_geometry' not in comparison_results:
                continue
                
            advanced_results = comparison_results['advanced_geometry']
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(16, 12))
            
            # 1. Manifold alignment visualization
            if 'manifold_alignment' in advanced_results:
                ax1 = fig.add_subplot(2, 3, 1)
                
                procrustes_results = advanced_results['manifold_alignment']['procrustes']
                cca_results = advanced_results['manifold_alignment']['cca']
                
                alignment_methods = ['Procrustes', 'CCA']
                alignment_values = [
                    procrustes_results.get('alignment_quality', 0),
                    cca_results.get('mean_correlation', 0)
                ]
                
                bars = ax1.bar(alignment_methods, alignment_values, color=['skyblue', 'lightcoral'])
                ax1.set_ylabel('Alignment Quality')
                ax1.set_title('Manifold Alignment Quality')
                ax1.set_ylim([0, 1])
                
                # Add value labels
                for bar, value in zip(bars, alignment_values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Information geometry metrics
            if 'information_geometry' in advanced_results:
                ax2 = fig.add_subplot(2, 3, 2)
                
                info_geom = advanced_results['information_geometry']
                metrics = ['KL\nDivergence', 'JS\nDivergence', 'Wasserstein\nDistance']
                values = [
                    info_geom.get('kl_divergence', 0),
                    info_geom.get('js_divergence', 0),
                    info_geom.get('wasserstein_distance', 0)
                ]
                
                colors = ['red', 'blue', 'green']
                bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
                ax2.set_ylabel('Distance/Divergence')
                ax2.set_title('Information Geometry Metrics')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 3. Curvature analysis
            if 'curvature_analysis' in advanced_results:
                ax3 = fig.add_subplot(2, 3, 3)
                
                curv_analysis = advanced_results['curvature_analysis']
                mean_curvatures = []
                condition_names = []
                
                for key, curv_data in curv_analysis.items():
                    if isinstance(curv_data, dict) and 'mean_curvature' in curv_data:
                        mean_curvatures.append(curv_data['mean_curvature'])
                        condition_names.append(key.replace('condition_', 'Cond '))
                
                if mean_curvatures:
                    bars = ax3.bar(condition_names, mean_curvatures, color=['orange', 'purple'])
                    ax3.set_ylabel('Mean Curvature')
                    ax3.set_title('Manifold Curvature by Condition')
                    
                    # Add value labels
                    for bar, value in zip(bars, mean_curvatures):
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height + max(mean_curvatures)*0.02,
                                f'{value:.3f}', ha='center', va='bottom')
            
            # 4. Geodesic distance comparison
            if 'geodesic_analysis' in advanced_results:
                ax4 = fig.add_subplot(2, 3, 4)
                
                geodesic_analysis = advanced_results['geodesic_analysis']
                mean_geodesics = []
                condition_names = []
                
                for key, geod_data in geodesic_analysis.items():
                    if isinstance(geod_data, dict) and 'mean_geodesic_distance' in geod_data:
                        mean_geodesics.append(geod_data['mean_geodesic_distance'])
                        condition_names.append(key.replace('condition_', 'Cond '))
                
                if mean_geodesics:
                    bars = ax4.bar(condition_names, mean_geodesics, color=['cyan', 'magenta'])
                    ax4.set_ylabel('Mean Geodesic Distance')
                    ax4.set_title('Geodesic Distances by Condition')
                    
                    # Add value labels
                    for bar, value in zip(bars, mean_geodesics):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + max(mean_geodesics)*0.02,
                                f'{value:.3f}', ha='center', va='bottom')
            
            # 5. PCA embedding comparison
            if 'pca' in comparison_results:
                ax5 = fig.add_subplot(2, 3, 5)
                
                pca_results = comparison_results['pca']
                embedding = pca_results['embedding']
                
                # Get condition labels for coloring
                if 'comparison_info' in comparison_results:
                    labels = comparison_results['comparison_info']['labels']
                    
                    # Handle filtered data - only use included trials
                    if np.any(labels == -1):
                        # Some trials were excluded - filter both embedding and labels
                        included_mask = labels != -1
                        filtered_labels = labels[included_mask]
                        # embedding should already be filtered, so use it as is
                        if len(filtered_labels) != len(embedding):
                            print(f"Warning: Label/embedding mismatch for {comparison_name}")
                            continue
                        labels_to_use = filtered_labels
                    else:
                        # No trials excluded
                        labels_to_use = labels
                    
                    unique_labels = np.unique(labels_to_use[labels_to_use != -1])
                    
                    colors = ['red', 'blue', 'green', 'purple']
                    for i, label in enumerate(unique_labels):
                        mask = labels_to_use == label
                        if np.any(mask) and np.sum(mask) <= len(embedding):
                            ax5.scatter(embedding[mask, 0], embedding[mask, 1], 
                                      c=colors[i % len(colors)], alpha=0.6, 
                                      label=f'Condition {label}', s=30)
                
                ax5.set_xlabel('PC1')
                ax5.set_ylabel('PC2')
                ax5.set_title('PCA Embedding')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            
            # 6. Distance analysis summary
            if 'pca' in comparison_results and 'distance_analysis' in comparison_results['pca']:
                ax6 = fig.add_subplot(2, 3, 6)
                
                dist_analysis = comparison_results['pca']['distance_analysis']
                within_dist = dist_analysis.get('mean_within_condition_distance', 0)
                between_dist = dist_analysis.get('mean_between_condition_distance', 0)
                
                bars = ax6.bar(['Within\nCondition', 'Between\nCondition'], 
                             [within_dist, between_dist], 
                             color=['lightblue', 'lightcoral'])
                ax6.set_ylabel('Mean Distance')
                ax6.set_title('Within vs Between Condition Distances')
                
                # Add value labels
                for bar, value in zip(bars, [within_dist, between_dist]):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + max(within_dist, between_dist)*0.02,
                            f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save figure
            filename = viz_dir / f'{roi_name}_{comparison_name}_advanced_geometry.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Advanced geometry visualization saved: {filename}")
    
    def _interpret_rsa_correlation(self, correlation: float, p_value: float) -> str:
        significance = "significant" if p_value < self.config['alpha'] else "not significant"
        
        if abs(correlation) < 0.1:
            strength = "very weak"
        elif abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.5:
            strength = "moderate"
        elif abs(correlation) < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if correlation > 0 else "negative"
        return f"{strength} {direction} correlation ({significance})"
    
    def _interpret_separation_ratio(self, ratio: float, p_value: float) -> str:
        significance = "significant" if p_value < self.config['alpha'] else "not significant"
        
        if ratio < 1.1:
            strength = "no"
        elif ratio < 1.5:
            strength = "weak"
        elif ratio < 2.0:
            strength = "moderate"
        elif ratio < 3.0:
            strength = "strong"
        else:
            strength = "very strong"
        
        return f"{strength} separation between conditions ({significance})"

def generate_example_dd_data():
    """Generate example delay discounting data for testing"""
    np.random.seed(42)
    
    n_trials = 200
    n_voxels = 100
    
    # Generate behavioral data
    behavioral_data = pd.DataFrame({
        'choice': np.random.choice([0, 1], n_trials),  # 0=SS, 1=LL
        'delay_days': np.random.choice([0, 1, 7, 30, 90, 180], n_trials),
        'chosen_sv': np.random.uniform(0, 1, n_trials),
        'unchosen_sv': np.random.uniform(0, 1, n_trials)
    })
    
    # Generate neural data with some structure based on choice
    neural_data = np.random.randn(n_trials, n_voxels)
    
    # Add signal for SS vs LL choices
    choice_signal = np.random.randn(n_voxels) * 0.5
    for i in range(n_trials):
        if behavioral_data.loc[i, 'choice'] == 1:  # LL choice
            neural_data[i] += choice_signal
    
    return neural_data, behavioral_data

def main():
    """Main function for delay discounting geometry analysis with trajectory analysis support"""
    parser = argparse.ArgumentParser(description="Delay Discounting Neural Geometry Analysis with Trajectory Support")
    parser.add_argument("--neural-data", help="Path to neural data file")
    parser.add_argument("--behavioral-data", help="Path to behavioral data CSV")
    parser.add_argument("--roi-name", default="ROI", help="Name of the ROI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--comparisons", nargs="+", 
                       help="Specific comparisons to run", 
                       choices=['choice', 'delay_short_vs_long', 'delay_immediate_vs_delayed',
                               'sv_chosen_median', 'sv_unchosen_median', 'sv_difference_median',
                               'value_diff_terciles'])
    parser.add_argument("--example", action="store_true", help="Run with example data")
    
    # NEW: Trajectory analysis options
    parser.add_argument("--trajectory", action="store_true", 
                       help="Run trajectory analysis (requires delay_days column)")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run both standard and trajectory analysis")
    parser.add_argument("--min-trials-per-delay", type=int, default=5,
                       help="Minimum trials per delay for trajectory analysis")
    parser.add_argument("--trajectory-only", action="store_true",
                       help="Run only trajectory analysis, skip standard comparisons")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DelayDiscountingGeometryAnalyzer(args.config)
    
    if args.output_dir:
        analyzer.output_dir = Path(args.output_dir)
        analyzer.output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.example:
        # Generate example data
        print("Generating example delay discounting data...")
        neural_data, behavioral_data = generate_example_dd_data()
        
        # Save temporary files
        temp_neural_file = analyzer.output_dir / "temp_neural_data.npy"
        temp_behavioral_file = analyzer.output_dir / "temp_behavioral_data.csv"
        
        np.save(temp_neural_file, neural_data)
        behavioral_data.to_csv(temp_behavioral_file, index=False)
        
        # Load data
        data = analyzer.load_delay_discounting_data(
            str(temp_neural_file),
            str(temp_behavioral_file),
            "Example_ROI"
        )
        
        # Clean up temp files
        temp_neural_file.unlink()
        temp_behavioral_file.unlink()
        
    else:
        if not args.neural_data or not args.behavioral_data:
            print("Error: --neural-data and --behavioral-data are required when not using --example")
            return
        
        data = analyzer.load_delay_discounting_data(
            args.neural_data,
            args.behavioral_data,
            args.roi_name
        )
    
    # Determine analysis type
    if args.comprehensive:
        # Run both standard and trajectory analysis
        print("🚀 Running COMPREHENSIVE analysis (standard + trajectory)...")
        all_results = analyzer.run_comprehensive_trajectory_analysis(data)
        
    elif args.trajectory_only:
        # Run only trajectory analysis
        print("🎯 Running TRAJECTORY-ONLY analysis...")
        all_results = analyzer.run_trajectory_analysis(data)
        
    elif args.trajectory:
        # Run standard analysis + trajectory analysis separately
        print("📊 Running STANDARD analysis...")
        standard_results = analyzer.run_comprehensive_dd_analysis(data, args.comparisons)
        
        print("\n🎯 Running TRAJECTORY analysis...")
        trajectory_results = analyzer.run_trajectory_analysis(data)
        
        all_results = {
            'standard_analysis': standard_results,
            'trajectory_analysis': trajectory_results
        }
        
    else:
        # Run standard analysis only
        print("📊 Running STANDARD delay discounting analysis...")
        all_results = analyzer.run_comprehensive_dd_analysis(data, args.comparisons)
    
    # Save results and create report
    if args.trajectory_only:
        # Special handling for trajectory-only results
        if 'error' not in all_results:
            traj_file = analyzer.output_dir / f"{data['roi_name']}_trajectory_results.json"
            analyzer.save_results(all_results, traj_file.name)
            print(f"Trajectory results saved to: {traj_file}")
        else:
            print(f"Trajectory analysis failed: {all_results['error']}")
    else:
        analyzer.save_all_results(all_results, data)
    
    print(f"\n✅ Analysis complete! Results saved to: {analyzer.output_dir}")
    
    # Print summary of what was run
    if args.comprehensive:
        print("\n📋 ANALYSIS SUMMARY:")
        if 'summary' in all_results:
            summary = all_results['summary']
            print(f"   Standard comparisons: {summary['n_standard_comparisons']}")
            if 'n_delay_conditions' in summary:
                print(f"   Trajectory conditions: {summary['n_delay_conditions']}")
                print(f"   Delay range: {summary['delay_range']}")
        
    elif args.trajectory or args.trajectory_only:
        if 'error' not in all_results and 'trajectory_metrics' in all_results:
            traj_metrics = all_results['trajectory_metrics']
            print("\n🎯 TRAJECTORY ANALYSIS SUMMARY:")
            print(f"   Mean velocity: {traj_metrics['mean_velocity']:.3f}")
            print(f"   Mean alignment quality: {traj_metrics['mean_alignment_quality']:.3f}")
            print(f"   Curvature trend: {traj_metrics['curvature_evolution_slope']:.3f}")
    
    print(f"\n🎉 Use --comprehensive for the most complete analysis!")
    print(f"💡 Trajectory analysis provides 3D visualizations and geometric dynamics!")

if __name__ == "__main__":
    main() 