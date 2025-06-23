#!/usr/bin/env python3
"""
Neural Geometry Analysis Script

A comprehensive tool for analyzing neural representational geometries in fMRI data.
Performs formal geometry comparisons between experimental conditions using:
- Representational Similarity Analysis (RSA)
- Procrustes analysis
- Dimensionality analysis
- Permutation testing
- Manifold learning

Based on geometry_comparison_example.py but enhanced for production use.

Author: Joshua Buckholtz Lab
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
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes

# Import from main pipeline if available
try:
    from delay_discounting_mvpa_pipeline import GeometryAnalysis, Config, ROIAnalysis
except ImportError:
    warnings.warn("Could not import from delay_discounting_mvpa_pipeline. Some functionality may be limited.")
    GeometryAnalysis = None
    Config = None
    ROIAnalysis = None

class NeuralGeometryAnalyzer:
    """
    Comprehensive neural geometry analysis tool based on geometry_comparison_example.py
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the neural geometry analyzer
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', './neural_geometry_analysis'))
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
            'output_dir': './neural_geometry_analysis',
            'n_permutations': 1000,
            'random_state': 42,
            'alpha': 0.05,
            'n_components_pca': 10,
            'n_components_mds': 5,
            'n_components_tsne': 3,
            'standardize_data': True,
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def load_neural_data(self, 
                        data_path: str, 
                        condition_labels: Optional[np.array] = None,
                        condition_names: Optional[List[str]] = None) -> Dict:
        """
        Load neural data from various formats
        
        Parameters:
        -----------
        data_path : str
            Path to neural data (numpy or csv)
        condition_labels : array-like, optional
            Condition labels for each trial
        condition_names : list, optional
            Names for each condition
            
        Returns:
        --------
        dict : Contains neural data and metadata
        """
        data_path = Path(data_path)
        
        if data_path.suffix == '.npy':
            # Load numpy array
            neural_data = np.load(data_path)
            
        elif data_path.suffix == '.csv':
            # Load CSV file
            df = pd.read_csv(data_path)
            neural_data = df.values
            
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Standardize data if requested
        if self.config['standardize_data']:
            scaler = StandardScaler()
            neural_data = scaler.fit_transform(neural_data)
        
        # Handle condition labels
        if condition_labels is None:
            condition_labels = np.zeros(neural_data.shape[0])
            
        if condition_names is None:
            condition_names = [f"Condition_{i}" for i in np.unique(condition_labels)]
        
        return {
            'neural_data': neural_data,
            'condition_labels': np.array(condition_labels),
            'condition_names': condition_names,
            'n_trials': neural_data.shape[0],
            'n_features': neural_data.shape[1]
        }
    
    def dimensionality_reduction(self, 
                               neural_data: np.array,
                               method: str = 'pca',
                               n_components: Optional[int] = None) -> Tuple[np.array, Any]:
        """
        Perform dimensionality reduction on neural data
        
        Parameters:
        -----------
        neural_data : array
            Neural data (trials x features)
        method : str
            Reduction method ('pca', 'mds', 'tsne')
        n_components : int, optional
            Number of components to keep
            
        Returns:
        --------
        tuple : (reduced_data, fitted_reducer)
        """
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
    
    def compute_representational_similarity(self, 
                                          neural_data: np.array,
                                          condition_labels: np.array,
                                          metric: str = 'correlation') -> Dict:
        """
        Compute representational similarity analysis (RSA)
        
        Parameters:
        -----------
        neural_data : array
            Neural data (trials x features)
        condition_labels : array
            Condition labels for each trial
        metric : str
            Distance metric to use
            
        Returns:
        --------
        dict : RSA results
        """
        # Compute pairwise distances
        if metric == 'correlation':
            # Use 1 - correlation as distance
            distances = 1 - np.corrcoef(neural_data)
        else:
            distances = pairwise_distances(neural_data, metric=metric)
        
        # Create condition model matrix
        n_conditions = len(np.unique(condition_labels))
        condition_model = np.zeros((len(condition_labels), len(condition_labels)))
        
        for i, label_i in enumerate(condition_labels):
            for j, label_j in enumerate(condition_labels):
                condition_model[i, j] = 1 if label_i == label_j else 0
        
        # Compute RSA correlation
        # Get upper triangular parts
        triu_indices = np.triu_indices(distances.shape[0], k=1)
        distance_vector = distances[triu_indices]
        model_vector = condition_model[triu_indices]
        
        # Compute correlation
        rsa_correlation, p_value = stats.pearsonr(distance_vector, model_vector)
        
        return {
            'rsa_correlation': rsa_correlation,
            'p_value': p_value,
            'distance_matrix': distances,
            'model_matrix': condition_model,
            'interpretation': self._interpret_rsa_correlation(rsa_correlation, p_value)
        }
    
    def procrustes_analysis(self, 
                          embedding1: np.array,
                          embedding2: np.array) -> Dict:
        """
        Perform Procrustes analysis to compare two embeddings
        
        Parameters:
        -----------
        embedding1, embedding2 : array
            Two embeddings to compare
            
        Returns:
        --------
        dict : Procrustes analysis results
        """
        # Perform Procrustes analysis
        mtx1, mtx2, disparity = procrustes(embedding1, embedding2)
        
        return {
            'disparity': disparity,
            'aligned_embedding1': mtx1,
            'aligned_embedding2': mtx2,
            'interpretation': self._interpret_procrustes_disparity(disparity)
        }
    
    def distance_analysis(self, 
                         embedding: np.array,
                         condition_labels: np.array) -> Dict:
        """
        Analyze distances within and between conditions
        
        Parameters:
        -----------
        embedding : array
            Embedded neural data
        condition_labels : array
            Condition labels
            
        Returns:
        --------
        dict : Distance analysis results
        """
        # Compute pairwise distances
        distances = pairwise_distances(embedding)
        
        # Separate within and between condition distances
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
        
        # Compute summary statistics
        mean_within = np.mean(within_distances)
        mean_between = np.mean(between_distances)
        separation_ratio = mean_between / mean_within if mean_within > 0 else np.inf
        
        # Permutation test for separation
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
    
    def dimensionality_analysis(self, 
                              neural_data: np.array,
                              condition_labels: np.array) -> Dict:
        """
        Analyze intrinsic dimensionality of neural representations
        
        Parameters:
        -----------
        neural_data : array
            Neural data
        condition_labels : array
            Condition labels
            
        Returns:
        --------
        dict : Dimensionality analysis results
        """
        results = {}
        
        for condition in np.unique(condition_labels):
            condition_data = neural_data[condition_labels == condition]
            
            # PCA to get explained variance
            pca = PCA()
            pca.fit(condition_data)
            
            # Compute intrinsic dimensionality
            explained_var_ratio = pca.explained_variance_ratio_
            cumsum_var = np.cumsum(explained_var_ratio)
            
            # Find dimensions needed for 80% and 90% variance
            dim_80 = np.argmax(cumsum_var >= 0.8) + 1
            dim_90 = np.argmax(cumsum_var >= 0.9) + 1
            
            # Participation ratio
            participation_ratio = (np.sum(explained_var_ratio) ** 2) / np.sum(explained_var_ratio ** 2)
            
            results[f"condition_{condition}"] = {
                'explained_variance_ratio': explained_var_ratio,
                'cumulative_variance': cumsum_var,
                'intrinsic_dimensionality_80': dim_80,
                'intrinsic_dimensionality_90': dim_90,
                'participation_ratio': participation_ratio
            }
        
        return results
    
    def comprehensive_geometry_comparison(self, 
                                        neural_data: np.array,
                                        condition_labels: np.array,
                                        condition_names: List[str],
                                        methods: List[str] = ['pca', 'mds']) -> Dict:
        """
        Perform comprehensive geometry comparison analysis
        
        Parameters:
        -----------
        neural_data : array
            Neural data
        condition_labels : array
            Condition labels
        condition_names : list
            Names for conditions
        methods : list
            Dimensionality reduction methods to use
            
        Returns:
        --------
        dict : Comprehensive analysis results
        """
        results = {}
        
        print(f"Running comprehensive geometry comparison...")
        print(f"Data shape: {neural_data.shape}")
        print(f"Conditions: {condition_names}")
        
        # 1. RSA on original data  
        print("Computing RSA...")
        rsa_results = self.compute_representational_similarity(neural_data, condition_labels)
        results['rsa'] = rsa_results
        
        # 2. Dimensionality analysis
        print("Computing dimensionality analysis...")
        dim_results = self.dimensionality_analysis(neural_data, condition_labels)
        results['dimensionality'] = dim_results
        
        # 3. Analysis for each reduction method
        for method in methods:
            print(f"Running {method.upper()} analysis...")
            
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
        
        # 4. Cross-method comparison (if multiple methods)
        if len(methods) > 1:
            print("Comparing across methods...")
            cross_comparisons = {}
            
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    emb1 = results[method1]['embedding']
                    emb2 = results[method2]['embedding']
                    
                    # Align dimensionalities
                    min_dim = min(emb1.shape[1], emb2.shape[1])
                    emb1_aligned = emb1[:, :min_dim]
                    emb2_aligned = emb2[:, :min_dim]
                    
                    # Procrustes analysis
                    procrustes_results = self.procrustes_analysis(emb1_aligned, emb2_aligned)
                    
                    cross_comparisons[f"{method1}_vs_{method2}"] = procrustes_results
            
            results['cross_method_comparison'] = cross_comparisons
        
        return results
    
    def create_visualizations(self, 
                            results: Dict,
                            condition_labels: np.array,
                            condition_names: List[str],
                            roi_name: str = "ROI") -> Dict:
        """
        Create comprehensive visualizations of geometry analysis results
        """
        plot_files = {}
        
        # Set up colors for conditions
        colors = plt.cm.Set1(np.linspace(0, 1, len(condition_names)))
        color_map = {i: colors[i] for i in range(len(condition_names))}
        
        # 1. Embedding plots
        methods_with_embeddings = [k for k in results.keys() if k in ['pca', 'mds', 'tsne']]
        if methods_with_embeddings:
            fig, axes = plt.subplots(1, len(methods_with_embeddings), 
                                    figsize=(6 * len(methods_with_embeddings), 5))
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            for plot_idx, method in enumerate(methods_with_embeddings):
                embedding = results[method]['embedding']
                
                ax = axes[plot_idx]
                for i, condition in enumerate(np.unique(condition_labels)):
                    mask = condition_labels == condition
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                             c=[color_map[i]], label=condition_names[i], alpha=0.7)
                
                ax.set_xlabel(f'{method.upper()} Component 1')
                ax.set_ylabel(f'{method.upper()} Component 2')
                ax.set_title(f'{method.upper()} Embedding - {roi_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            embedding_file = self.output_dir / f"{roi_name}_embeddings.{self.config['plot_format']}"
            plt.savefig(embedding_file, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            plot_files['embeddings'] = str(embedding_file)
        
        # 2. Distance distributions
        if len(methods_with_embeddings) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for idx, method in enumerate(methods_with_embeddings[:2]):
                if method in results and 'distance_analysis' in results[method]:
                    dist_results = results[method]['distance_analysis']
                    
                    within_dists = dist_results['within_distances']
                    between_dists = dist_results['between_distances']
                    
                    ax = axes[idx]
                    ax.hist(within_dists, alpha=0.7, label='Within condition', bins=20)
                    ax.hist(between_dists, alpha=0.7, label='Between condition', bins=20)
                    ax.set_xlabel('Distance')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{method.upper()} Distance Distribution - {roi_name}')
                    ax.legend()
                    
                    # Add separation ratio as text
                    sep_ratio = dist_results['separation_ratio']
                    p_val = dist_results['separation_p_value']
                    ax.text(0.7, 0.9, f'Separation ratio: {sep_ratio:.2f}\np-value: {p_val:.3f}',
                           transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            
            plt.tight_layout()
            distance_file = self.output_dir / f"{roi_name}_distances.{self.config['plot_format']}"
            plt.savefig(distance_file, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            plot_files['distances'] = str(distance_file)
        
        return plot_files
    
    def print_results_summary(self, results: Dict, condition_names: List[str]):
        """Print a comprehensive summary of results"""
        print("\n" + "="*60)
        print("NEURAL GEOMETRY ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        # RSA results
        if 'rsa' in results:
            rsa = results['rsa']
            print(f"\nRepresentational Similarity Analysis:")
            print(f"  Correlation: {rsa['rsa_correlation']:.3f}")
            print(f"  P-value: {rsa['p_value']:.3f}")
            print(f"  Interpretation: {rsa['interpretation']}")
        
        # Dimensionality results
        if 'dimensionality' in results:
            print(f"\nDimensionality Analysis:")
            for condition_key, dim_results in results['dimensionality'].items():
                cond_idx = int(condition_key.split('_')[1])
                cond_name = condition_names[cond_idx]
                print(f"  {cond_name}:")
                print(f"    Participation ratio: {dim_results['participation_ratio']:.3f}")
                print(f"    Intrinsic dim (80%): {dim_results['intrinsic_dimensionality_80']}")
                print(f"    Intrinsic dim (90%): {dim_results['intrinsic_dimensionality_90']}")
        
        # Method-specific results
        for method in ['pca', 'mds']:
            if method in results and 'distance_analysis' in results[method]:
                dist = results[method]['distance_analysis']
                print(f"\n{method.upper()} Distance Analysis:")
                print(f"  Mean within-condition distance: {dist['mean_within_condition_distance']:.3f}")
                print(f"  Mean between-condition distance: {dist['mean_between_condition_distance']:.3f}")
                print(f"  Separation ratio: {dist['separation_ratio']:.3f}")
                print(f"  P-value: {dist['separation_p_value']:.3f}")
                print(f"  Interpretation: {dist['interpretation']}")
                
                if 'rsa' in results[method]:
                    method_rsa = results[method]['rsa']
                    print(f"  RSA Correlation: {method_rsa['rsa_correlation']:.3f}")
                    print(f"  RSA P-value: {method_rsa['p_value']:.3f}")
        
        # Cross-method comparisons
        if 'cross_method_comparison' in results:
            print(f"\nCross-Method Comparisons:")
            for comparison_name, proc_results in results['cross_method_comparison'].items():
                print(f"  {comparison_name}:")
                print(f"    Procrustes disparity: {proc_results['disparity']:.3f}")
                print(f"    Interpretation: {proc_results['interpretation']}")
    
    def save_results(self, results: Dict, filename: str):
        """Save analysis results to file"""
        # Create a serializable version of results
        serializable_results = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_results[key][subkey] = subvalue.tolist()
                    elif hasattr(subvalue, '__dict__'):
                        # Skip non-serializable objects like fitted models
                        continue
                    else:
                        serializable_results[key][subkey] = subvalue
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif hasattr(value, '__dict__'):
                # Skip non-serializable objects
                continue
            else:
                serializable_results[key] = value
        
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def _interpret_rsa_correlation(self, correlation: float, p_value: float) -> str:
        """Interpret RSA correlation results"""
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
    
    def _interpret_procrustes_disparity(self, disparity: float) -> str:
        """Interpret Procrustes disparity results"""
        if disparity < 0.1:
            return "very similar geometries"
        elif disparity < 0.3:
            return "similar geometries"
        elif disparity < 0.5:
            return "moderately different geometries"
        elif disparity < 0.7:
            return "different geometries"
        else:
            return "very different geometries"
    
    def _interpret_separation_ratio(self, ratio: float, p_value: float) -> str:
        """Interpret separation ratio results"""
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

def generate_example_data():
    """Generate example data for testing (similar to geometry_comparison_example.py)"""
    np.random.seed(42)
    
    # Generate two conditions with different geometric properties
    n_trials_per_condition = 100
    n_features = 50
    
    # Condition 1: More compact, lower dimensional structure
    n_latent_1 = 3
    latent_1 = np.random.randn(n_trials_per_condition, n_latent_1)
    mixing_matrix_1 = np.random.randn(n_latent_1, n_features) * 0.5
    condition_1 = latent_1 @ mixing_matrix_1 + np.random.randn(n_trials_per_condition, n_features) * 0.1
    
    # Condition 2: More spread out, higher dimensional structure
    n_latent_2 = 8
    latent_2 = np.random.randn(n_trials_per_condition, n_latent_2) * 1.5
    mixing_matrix_2 = np.random.randn(n_latent_2, n_features) * 0.3
    condition_2 = latent_2 @ mixing_matrix_2 + np.random.randn(n_trials_per_condition, n_features) * 0.2
    
    # Add systematic separation between conditions
    condition_2 += 1.5
    
    # Combine data
    neural_data = np.vstack([condition_1, condition_2])
    condition_labels = np.hstack([np.zeros(n_trials_per_condition), 
                                 np.ones(n_trials_per_condition)])
    condition_names = ['Compact Condition', 'Spread Condition']
    
    return neural_data, condition_labels, condition_names

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Neural Geometry Analysis")
    parser.add_argument("--data-path", help="Path to neural data file")
    parser.add_argument("--condition-labels", help="Path to condition labels file")
    parser.add_argument("--condition-names", nargs="+", help="Names for conditions")
    parser.add_argument("--roi-name", default="ROI", help="Name of the ROI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--methods", nargs="+", default=["pca", "mds"], 
                       help="Dimensionality reduction methods to use")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--example", action="store_true", help="Run with example data")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = NeuralGeometryAnalyzer(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        analyzer.output_dir = Path(args.output_dir)
        analyzer.output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.example:
        # Use example data
        print("Generating example data...")
        neural_data, condition_labels, condition_names = generate_example_data()
        
        data = {
            'neural_data': neural_data,
            'condition_labels': condition_labels,
            'condition_names': condition_names,
            'n_trials': neural_data.shape[0],
            'n_features': neural_data.shape[1]
        }
        
    else:
        if not args.data_path:
            print("Error: --data-path is required when not using --example")
            return
        
        # Load condition labels if provided
        condition_labels = None
        if args.condition_labels:
            if args.condition_labels.endswith('.npy'):
                condition_labels = np.load(args.condition_labels)
            elif args.condition_labels.endswith('.csv'):
                condition_labels = pd.read_csv(args.condition_labels).values.flatten()
        
        # Load neural data
        print("Loading neural data...")
        data = analyzer.load_neural_data(
            args.data_path,
            condition_labels=condition_labels,
            condition_names=args.condition_names
        )
    
    print(f"Loaded data: {data['n_trials']} trials, {data['n_features']} features")
    print(f"Conditions: {data['condition_names']}")
    
    # Run comprehensive analysis
    print("Running comprehensive geometry analysis...")
    results = analyzer.comprehensive_geometry_comparison(
        data['neural_data'],
        data['condition_labels'],
        data['condition_names'],
        methods=args.methods
    )
    
    # Create visualizations
    print("Creating visualizations...")
    plot_files = analyzer.create_visualizations(
        results,
        data['condition_labels'],
        data['condition_names'],
        roi_name=args.roi_name
    )
    
    # Print comprehensive results summary
    analyzer.print_results_summary(results, data['condition_names'])
    
    # Save results
    analyzer.save_results(results, f"{args.roi_name}_geometry_analysis_results.json")
    
    print(f"\nResults saved to: {analyzer.output_dir}")
    print(f"Visualizations: {list(plot_files.keys())}")

if __name__ == "__main__":
    main() 