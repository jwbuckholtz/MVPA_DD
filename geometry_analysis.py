#!/usr/bin/env python3
"""
Geometry Analysis Module
=======================

Refactored geometry analysis class that inherits from BaseAnalysis.
Focuses on neural geometry analysis including dimensionality reduction,
representational similarity analysis, and manifold comparisons.

Key Features:
- Representational dissimilarity matrix (RDM) computation
- Dimensionality reduction (PCA, MDS, t-SNE, Isomap)
- Behavioral-neural geometry correlations
- Embedding visualizations
- Condition-based geometry comparisons
- Permutation testing for geometric properties

Author: Cognitive Neuroscience Lab, Stanford University
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Import base analysis class
from analysis_base import BaseAnalysis, AnalysisError, AnalysisFactory
from oak_storage_config import OAKConfig
from data_utils import DataError

# Import MVPA utilities for dimensionality reduction
from mvpa_utils import run_dimensionality_reduction, MVPAError


class GeometryAnalysis(BaseAnalysis):
    """
    Geometry analysis class for neural representational geometry
    
    This class inherits from BaseAnalysis and implements geometry-specific
    functionality including RDM computation, dimensionality reduction,
    and behavioral-neural geometry correlations.
    """
    
    def __init__(self, config: OAKConfig = None, **kwargs):
        """
        Initialize geometry analysis
        
        Parameters:
        -----------
        config : OAKConfig, optional
            Configuration object
        **kwargs : dict
            Additional arguments for base class
        """
        super().__init__(
            config=config,
            name='GeometryAnalysis',
            **kwargs
        )
        
        # Geometry analysis specific settings
        self.geometry_params = {
            'dimensionality_reduction': {
                'methods': ['pca', 'mds', 'tsne', 'isomap'],
                'n_components': 10,
                'default_method': 'pca'
            },
            'rdm': {
                'metric': 'correlation',
                'standardize': True
            },
            'visualization': {
                'figsize': (15, 12),
                'dpi': 300,
                'alpha': 0.6
            },
            'permutation_testing': {
                'n_permutations': 1000,
                'two_tailed': True
            }
        }
        
        # Initialize maskers dictionary
        self.maskers = {}
        
        self.logger.info("Geometry analysis initialized")
    
    def _initialize_analysis_components(self):
        """Initialize geometry analysis specific components"""
        # Create maskers for ROIs
        self.maskers = self.create_maskers()
        self.logger.info(f"Initialized {len(self.maskers)} maskers")
    
    def compute_neural_rdm(self, X: np.ndarray, metric: str = None) -> np.ndarray:
        """
        Compute representational dissimilarity matrix
        
        Parameters:
        -----------
        X : np.ndarray
            Neural data (n_trials x n_voxels)
        metric : str, optional
            Distance metric to use
            
        Returns:
        --------
        np.ndarray : RDM (n_trials x n_trials)
        """
        if metric is None:
            metric = self.geometry_params['rdm']['metric']
        
        # Standardize if requested
        if self.geometry_params['rdm']['standardize']:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Compute pairwise distances
        distances = pdist(X, metric=metric)
        rdm = squareform(distances)
        
        return rdm
    
    def dimensionality_reduction(self, X: np.ndarray, method: str = None, 
                               n_components: int = None) -> Tuple[np.ndarray, Any]:
        """
        Perform dimensionality reduction on neural data
        
        Parameters:
        -----------
        X : np.ndarray
            Neural data (n_trials x n_voxels)
        method : str, optional
            Method for dimensionality reduction
        n_components : int, optional
            Number of components
            
        Returns:
        --------
        Tuple[np.ndarray, Any] : Low-dimensional embedding and reducer object
        """
        if method is None:
            method = self.geometry_params['dimensionality_reduction']['default_method']
        
        if n_components is None:
            n_components = self.geometry_params['dimensionality_reduction']['n_components']
        
        result = run_dimensionality_reduction(
            X, method=method, n_components=n_components
        )
        
        if result['success']:
            return result['embedding'], result['reducer']
        else:
            raise AnalysisError(f"Dimensionality reduction failed: {result['error']}")
    
    def behavioral_geometry_correlation(self, embedding: np.ndarray, 
                                      behavioral_vars: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Correlate behavioral variables with neural geometry
        
        Parameters:
        -----------
        embedding : np.ndarray
            Low-dimensional neural embedding (n_trials x n_components)
        behavioral_vars : Dict[str, np.ndarray]
            Dictionary of behavioral variables (each n_trials length)
            
        Returns:
        --------
        Dict[str, Any] : Correlation results
        """
        results = {}
        
        for var_name, var_values in behavioral_vars.items():
            correlations = []
            p_values = []
            
            for dim in range(embedding.shape[1]):
                r, p = stats.pearsonr(embedding[:, dim], var_values)
                correlations.append(r)
                p_values.append(p)
            
            results[var_name] = {
                'correlations': np.array(correlations),
                'p_values': np.array(p_values),
                'max_correlation': np.max(np.abs(correlations)),
                'best_dimension': np.argmax(np.abs(correlations))
            }
        
        return results
    
    def compare_embeddings_by_condition(self, embedding: np.ndarray, 
                                      condition_labels: np.ndarray,
                                      condition_names: List[str] = None,
                                      n_permutations: int = None) -> Dict[str, Any]:
        """
        Compare neural embeddings between conditions using permutation testing
        
        Parameters:
        -----------
        embedding : np.ndarray
            Low-dimensional neural embedding (n_trials x n_components)
        condition_labels : np.ndarray
            Condition labels for each trial
        condition_names : List[str], optional
            Names for conditions
        n_permutations : int, optional
            Number of permutations for testing
            
        Returns:
        --------
        Dict[str, Any] : Comparison results
        """
        if n_permutations is None:
            n_permutations = self.geometry_params['permutation_testing']['n_permutations']
        
        unique_conditions = np.unique(condition_labels)
        
        if condition_names is None:
            condition_names = [f"Condition_{i}" for i in unique_conditions]
        
        results = {
            'condition_names': condition_names,
            'unique_conditions': unique_conditions,
            'n_permutations': n_permutations,
            'properties': {}
        }
        
        # Define geometric properties to test
        def compute_centroid_distance(labels):
            """Compute distance between condition centroids"""
            centroids = []
            for condition in unique_conditions:
                mask = labels == condition
                if np.sum(mask) > 0:
                    centroid = np.mean(embedding[mask], axis=0)
                    centroids.append(centroid)
            
            if len(centroids) >= 2:
                return np.linalg.norm(centroids[0] - centroids[1])
            else:
                return 0.0
        
        def compute_within_condition_variance(labels):
            """Compute within-condition variance"""
            total_variance = 0
            n_conditions = 0
            
            for condition in unique_conditions:
                mask = labels == condition
                if np.sum(mask) > 1:
                    condition_data = embedding[mask]
                    condition_variance = np.mean(np.var(condition_data, axis=0))
                    total_variance += condition_variance
                    n_conditions += 1
            
            return total_variance / n_conditions if n_conditions > 0 else 0.0
        
        def compute_between_condition_variance(labels):
            """Compute between-condition variance"""
            centroids = []
            for condition in unique_conditions:
                mask = labels == condition
                if np.sum(mask) > 0:
                    centroid = np.mean(embedding[mask], axis=0)
                    centroids.append(centroid)
            
            if len(centroids) >= 2:
                return np.mean(np.var(centroids, axis=0))
            else:
                return 0.0
        
        # Test each property
        properties = {
            'centroid_distance': compute_centroid_distance,
            'within_condition_variance': compute_within_condition_variance,
            'between_condition_variance': compute_between_condition_variance
        }
        
        for prop_name, prop_func in properties.items():
            # Observed statistic
            observed_stat = prop_func(condition_labels)
            
            # Permutation test
            permuted_stats = []
            for _ in range(n_permutations):
                shuffled_labels = np.random.permutation(condition_labels)
                permuted_stat = prop_func(shuffled_labels)
                permuted_stats.append(permuted_stat)
            
            permuted_stats = np.array(permuted_stats)
            
            # Calculate p-value
            if self.geometry_params['permutation_testing']['two_tailed']:
                p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
            else:
                p_value = np.mean(permuted_stats >= observed_stat)
            
            results['properties'][prop_name] = {
                'observed': observed_stat,
                'permuted_mean': np.mean(permuted_stats),
                'permuted_std': np.std(permuted_stats),
                'p_value': p_value,
                'z_score': (observed_stat - np.mean(permuted_stats)) / np.std(permuted_stats)
            }
        
        return results
    
    def visualize_embeddings(self, embedding: np.ndarray, behavioral_vars: Dict[str, np.ndarray],
                           roi_name: str, output_dir: str = None, 
                           figsize: Tuple[int, int] = None) -> Dict[str, str]:
        """
        Create comprehensive visualizations of neural embeddings
        
        Parameters:
        -----------
        embedding : np.ndarray
            Low-dimensional neural embedding (n_trials x n_components)
        behavioral_vars : Dict[str, np.ndarray]
            Dictionary of behavioral variables
        roi_name : str
            Name of ROI for plot titles
        output_dir : str, optional
            Directory to save plots
        figsize : Tuple[int, int], optional
            Figure size for plots
            
        Returns:
        --------
        Dict[str, str] : Dictionary of created figure paths
        """
        if figsize is None:
            figsize = self.geometry_params['visualization']['figsize']
        
        if output_dir is None:
            output_dir = Path(self.config.OUTPUT_DIR) / "geometry_plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        figures = {}
        n_dims = embedding.shape[1]
        
        # 2D scatter plots for first few dimensions
        if n_dims >= 2:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.ravel()
            
            # Plot different combinations of dimensions
            dim_pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
            
            for i, (dim1, dim2) in enumerate(dim_pairs[:6]):
                if i < len(axes) and dim1 < n_dims and dim2 < n_dims:
                    ax = axes[i]
                    
                    # Color by choice if available
                    if 'choice' in behavioral_vars:
                        choices = behavioral_vars['choice']
                        colors = ['red' if c == 0 else 'blue' for c in choices]
                        scatter = ax.scatter(embedding[:, dim1], embedding[:, dim2], 
                                           c=colors, alpha=self.geometry_params['visualization']['alpha'], s=50)
                        # Create legend
                        red_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor='red', markersize=8, label='Smaller-Sooner')
                        blue_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor='blue', markersize=8, label='Larger-Later')
                        ax.legend(handles=[red_patch, blue_patch])
                    else:
                        ax.scatter(embedding[:, dim1], embedding[:, dim2], 
                                 alpha=self.geometry_params['visualization']['alpha'], s=50)
                    
                    ax.set_xlabel(f'Dimension {dim1 + 1}')
                    ax.set_ylabel(f'Dimension {dim2 + 1}')
                    ax.set_title(f'{roi_name}: Dims {dim1+1} vs {dim2+1}')
                    ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(dim_pairs), len(axes)):
                if i < len(axes):
                    fig.delaxes(axes[i])
            
            plt.tight_layout()
            fig.suptitle(f'{roi_name} - Neural Embedding 2D Projections', 
                        fontsize=16, y=1.02)
            
            filename = output_dir / f'{roi_name}_embedding_2d.png'
            plt.savefig(filename, dpi=self.geometry_params['visualization']['dpi'], 
                       bbox_inches='tight')
            figures['2d_projections'] = str(filename)
            plt.close()
        
        # 3D visualization
        if n_dims >= 3:
            fig = plt.figure(figsize=(12, 8))
            
            # Create 2x2 subplot for different 3D views
            for i, view_angle in enumerate([(30, 45), (45, 30), (60, 60), (15, 75)]):
                ax = fig.add_subplot(2, 2, i+1, projection='3d')
                
                if 'choice' in behavioral_vars:
                    choices = behavioral_vars['choice']
                    colors = ['red' if c == 0 else 'blue' for c in choices]
                    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                             c=colors, alpha=self.geometry_params['visualization']['alpha'], s=50)
                else:
                    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                             alpha=self.geometry_params['visualization']['alpha'], s=50)
                
                ax.set_xlabel('Dimension 1')
                ax.set_ylabel('Dimension 2')
                ax.set_zlabel('Dimension 3')
                ax.set_title(f'View {i+1}')
                ax.view_init(elev=view_angle[0], azim=view_angle[1])
            
            fig.suptitle(f'{roi_name} - 3D Neural Embedding', fontsize=16)
            plt.tight_layout()
            
            filename = output_dir / f'{roi_name}_embedding_3d.png'
            plt.savefig(filename, dpi=self.geometry_params['visualization']['dpi'], 
                       bbox_inches='tight')
            figures['3d_embedding'] = str(filename)
            plt.close()
        
        return figures
    
    def process_subject(self, subject_id: str, **kwargs) -> Dict[str, Any]:
        """
        Process geometry analysis for a single subject
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        **kwargs : dict
            Additional parameters for processing
            
        Returns:
        --------
        Dict[str, Any] : Processing results
        """
        start_time = time.time()
        
        try:
            # Load behavioral data
            behavioral_data = self.load_behavioral_data(subject_id, **kwargs)
            
            # Load fMRI data
            img, confounds = self.load_fmri_data(subject_id, **kwargs)
            
            # Validate data
            if len(behavioral_data) == 0:
                return {
                    'subject_id': subject_id,
                    'success': False,
                    'error': 'Empty behavioral data'
                }
            
            # Import MVPAAnalysis to extract neural patterns
            from mvpa_analysis import MVPAAnalysis
            
            # Create temporary MVPA analysis instance for pattern extraction
            temp_mvpa = MVPAAnalysis(self.config)
            
            # Process each ROI
            roi_results = {}
            available_rois = list(self.maskers.keys())
            
            for roi_name in available_rois:
                roi_start_time = time.time()
                
                try:
                    # Extract neural patterns using MVPA analysis
                    X = temp_mvpa.extract_trial_data(img, behavioral_data, roi_name, confounds)
                    
                    # Validate neural data
                    if X.shape[0] != len(behavioral_data):
                        self.logger.warning(f"Trial count mismatch for {roi_name}: "
                                          f"neural={X.shape[0]}, behavioral={len(behavioral_data)}")
                        continue
                    
                    # Compute RDM
                    rdm = self.compute_neural_rdm(X)
                    
                    # Dimensionality reduction
                    embedding, reducer = self.dimensionality_reduction(X)
                    
                    # Behavioral correlations
                    behavioral_vars = {}
                    for var_name in ['sv_diff', 'sv_sum', 'sv_chosen', 'sv_unchosen', 'svchosen_unchosen', 'choice']:
                        if var_name in behavioral_data.columns:
                            behavioral_vars[var_name] = behavioral_data[var_name].values
                    
                    correlations = self.behavioral_geometry_correlation(embedding, behavioral_vars)
                    
                    # Condition comparisons (if choice data available)
                    condition_comparisons = {}
                    if 'choice' in behavioral_vars:
                        choices = behavioral_vars['choice']
                        valid_mask = ~np.isnan(choices)
                        
                        if np.sum(valid_mask) > 10:  # Minimum trials threshold
                            try:
                                condition_comparisons = self.compare_embeddings_by_condition(
                                    embedding[valid_mask], 
                                    choices[valid_mask].astype(int),
                                    condition_names=['Smaller-Sooner', 'Larger-Later']
                                )
                            except Exception as e:
                                self.logger.warning(f"Condition comparison failed for {roi_name}: {e}")
                                condition_comparisons = {'error': str(e)}
                    
                    # Create visualizations
                    try:
                        figures = self.visualize_embeddings(
                            embedding, behavioral_vars, roi_name,
                            output_dir=Path(self.config.OUTPUT_DIR) / 'geometry_plots' / subject_id
                        )
                    except Exception as e:
                        self.logger.warning(f"Visualization failed for {roi_name}: {e}")
                        figures = {'error': str(e)}
                    
                    # Store ROI results
                    roi_processing_time = time.time() - roi_start_time
                    roi_results[roi_name] = {
                        'success': True,
                        'rdm': rdm,
                        'embedding': embedding,
                        'behavioral_correlations': correlations,
                        'condition_comparisons': condition_comparisons,
                        'figures': figures,
                        'n_voxels': X.shape[1],
                        'n_trials': X.shape[0],
                        'n_components': embedding.shape[1],
                        'processing_time': roi_processing_time
                    }
                    
                    self.logger.info(f"Processed {roi_name} for {subject_id}: "
                                   f"{X.shape[1]} voxels, {X.shape[0]} trials, "
                                   f"{embedding.shape[1]} components")
                    
                except Exception as e:
                    roi_processing_time = time.time() - roi_start_time
                    roi_results[roi_name] = {
                        'success': False,
                        'error': str(e),
                        'processing_time': roi_processing_time
                    }
                    self.logger.error(f"ROI processing failed for {roi_name}: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update processing stats
            self.update_processing_stats(subject_id, processing_time, success=True)
            
            # Create summary
            successful_rois = [roi for roi, result in roi_results.items() if result['success']]
            
            # Store results
            result = {
                'subject_id': subject_id,
                'success': True,
                'roi_results': roi_results,
                'successful_rois': successful_rois,
                'n_successful_rois': len(successful_rois),
                'n_total_rois': len(available_rois),
                'processing_time': processing_time,
                'behavioral_trials': len(behavioral_data)
            }
            
            self.results[subject_id] = result
            
            self.logger.info(f"Processed {subject_id}: {len(successful_rois)}/{len(available_rois)} ROIs successful")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_processing_stats(subject_id, processing_time, success=False)
            
            error_msg = f"Processing failed for {subject_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'subject_id': subject_id,
                'success': False,
                'error': error_msg,
                'processing_time': processing_time
            }
    
    def run_analysis(self, subjects: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run geometry analysis for multiple subjects
        
        Parameters:
        -----------
        subjects : List[str], optional
            List of subject IDs to process
        **kwargs : dict
            Additional parameters for processing
            
        Returns:
        --------
        Dict[str, Any] : Complete analysis results
        """
        # Get subject list
        subjects = self.get_subject_list(subjects)
        
        self.logger.info(f"Starting geometry analysis for {len(subjects)} subjects")
        
        # Process subjects sequentially
        all_results = {}
        successful_subjects = []
        failed_subjects = []
        
        for subject_id in subjects:
            result = self.process_subject(subject_id, **kwargs)
            all_results[subject_id] = result
            
            if result['success']:
                successful_subjects.append(subject_id)
            else:
                failed_subjects.append(subject_id)
                self.logger.warning(f"Subject {subject_id} failed: {result['error']}")
        
        # Create summary statistics
        if successful_subjects:
            # Collect correlation strengths
            correlation_summaries = {}
            condition_summaries = {}
            
            for subject_id in successful_subjects:
                subject_result = all_results[subject_id]
                
                for roi_name, roi_result in subject_result['roi_results'].items():
                    if roi_result['success']:
                        # Behavioral correlations
                        correlations = roi_result['behavioral_correlations']
                        for var_name, corr_result in correlations.items():
                            key = f"{roi_name}_{var_name}"
                            if key not in correlation_summaries:
                                correlation_summaries[key] = []
                            correlation_summaries[key].append(corr_result['max_correlation'])
                        
                        # Condition comparisons
                        comparisons = roi_result['condition_comparisons']
                        if 'properties' in comparisons:
                            for prop_name, prop_result in comparisons['properties'].items():
                                key = f"{roi_name}_{prop_name}"
                                if key not in condition_summaries:
                                    condition_summaries[key] = []
                                condition_summaries[key].append(prop_result['p_value'])
            
            # Calculate summary statistics
            correlation_summary = {}
            for key, values in correlation_summaries.items():
                correlation_summary[key] = {
                    'mean_correlation': np.mean(values),
                    'std_correlation': np.std(values),
                    'n_subjects': len(values)
                }
            
            condition_summary = {}
            for key, p_values in condition_summaries.items():
                condition_summary[key] = {
                    'mean_p_value': np.mean(p_values),
                    'significant_subjects': np.sum(np.array(p_values) < 0.05),
                    'n_subjects': len(p_values)
                }
            
            summary_stats = {
                'n_subjects_total': len(subjects),
                'n_subjects_successful': len(successful_subjects),
                'n_subjects_failed': len(failed_subjects),
                'success_rate': len(successful_subjects) / len(subjects),
                'correlation_summary': correlation_summary,
                'condition_summary': condition_summary
            }
        else:
            summary_stats = {
                'n_subjects_total': len(subjects),
                'n_subjects_successful': 0,
                'n_subjects_failed': len(failed_subjects),
                'success_rate': 0.0
            }
        
        # Store summary in results
        self.results['_summary'] = summary_stats
        
        self.logger.info(f"Geometry analysis complete: "
                        f"{len(successful_subjects)}/{len(subjects)} subjects successful")
        
        return {
            'results': all_results,
            'summary': summary_stats,
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects
        }
    
    def get_analysis_summary(self) -> str:
        """
        Get geometry analysis specific summary
        
        Returns:
        --------
        str : Analysis summary
        """
        if '_summary' not in self.results:
            return "No analysis results available"
        
        summary = self.results['_summary']
        
        # Format correlation summary
        correlation_summary = summary.get('correlation_summary', {})
        correlation_lines = []
        for key, stats in correlation_summary.items():
            correlation = stats['mean_correlation']
            n_subjects = stats['n_subjects']
            correlation_lines.append(f"  {key}: {correlation:.3f} ± {stats['std_correlation']:.3f} (n={n_subjects})")
        
        # Format condition comparison summary
        condition_summary = summary.get('condition_summary', {})
        condition_lines = []
        for key, stats in condition_summary.items():
            significant = stats['significant_subjects']
            n_subjects = stats['n_subjects']
            condition_lines.append(f"  {key}: {significant}/{n_subjects} significant (p<0.05)")
        
        summary_text = f"""Geometry Analysis Summary:
- Subjects processed: {summary.get('n_subjects_successful', 0)}/{summary.get('n_subjects_total', 0)}
- Success rate: {summary.get('success_rate', 0.0):.1%}

Behavioral-Neural Correlations:
{chr(10).join(correlation_lines) if correlation_lines else "  No correlation results"}

Condition Comparisons:
{chr(10).join(condition_lines) if condition_lines else "  No condition comparison results"}"""
        
        return summary_text
    
    def create_geometry_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of geometry results
        
        Returns:
        --------
        pd.DataFrame : Summary of geometry results
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for subject_id, result in self.results.items():
            if subject_id.startswith('_'):  # Skip summary entries
                continue
                
            if result['success']:
                for roi_name, roi_result in result['roi_results'].items():
                    if roi_result['success']:
                        # Behavioral correlation rows
                        correlations = roi_result['behavioral_correlations']
                        for var_name, corr_result in correlations.items():
                            summary_data.append({
                                'subject_id': subject_id,
                                'roi_name': roi_name,
                                'analysis_type': 'behavioral_correlation',
                                'variable_name': var_name,
                                'max_correlation': corr_result['max_correlation'],
                                'best_dimension': corr_result['best_dimension'],
                                'n_trials': roi_result['n_trials'],
                                'n_voxels': roi_result['n_voxels'],
                                'n_components': roi_result['n_components']
                            })
                        
                        # Condition comparison rows
                        comparisons = roi_result['condition_comparisons']
                        if 'properties' in comparisons:
                            for prop_name, prop_result in comparisons['properties'].items():
                                summary_data.append({
                                    'subject_id': subject_id,
                                    'roi_name': roi_name,
                                    'analysis_type': 'condition_comparison',
                                    'variable_name': prop_name,
                                    'observed_value': prop_result['observed'],
                                    'p_value': prop_result['p_value'],
                                    'z_score': prop_result['z_score'],
                                    'n_trials': roi_result['n_trials'],
                                    'n_voxels': roi_result['n_voxels'],
                                    'n_components': roi_result['n_components']
                                })
        
        return pd.DataFrame(summary_data)


# Register the class with the factory
AnalysisFactory.register('geometry', GeometryAnalysis)


if __name__ == "__main__":
    # Example usage
    from oak_storage_config import OAKConfig
    
    # Create geometry analysis instance
    config = OAKConfig()
    geometry_analysis = GeometryAnalysis(config)
    
    # Run analysis on a few subjects
    subjects = geometry_analysis.get_subject_list()[:2]  # Just first 2 subjects
    results = geometry_analysis.run_analysis(subjects)
    
    print("Analysis Results:")
    print(results['summary'])
    
    # Create summary dataframe
    summary_df = geometry_analysis.create_geometry_summary_dataframe()
    print("\nSummary DataFrame:")
    print(summary_df) 