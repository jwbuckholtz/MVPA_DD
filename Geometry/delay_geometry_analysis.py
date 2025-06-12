#!/usr/bin/env python3
"""
Delay-Specific Geometry Analysis for MVPA
=========================================

This script analyzes how the geometric properties of neural representations
change across different delay conditions in the delay discounting task.

Key Questions:
1. How does representational geometry change with delay length?
2. Are there systematic transformations in neural space across delays?
3. Which brain regions show delay-dependent geometric changes?
4. How do value representations evolve across delay conditions?

Author: Stanford Cognitive Neuroscience Lab
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def load_delay_specific_data(subject_id, roi_name):
    """
    Load fMRI data organized by delay condition.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    roi_name : str
        ROI name (striatum, dlpfc, vmpfc)
    
    Returns:
    --------
    dict
        Dictionary with delay conditions as keys, neural patterns as values
    """
    # Replace with actual paths to your data
    data_path = f'/oak/stanford/groups/your_group/data/sub-{subject_id}'
    
    # Load behavioral data to get delay information
    behavior_file = f'{data_path}/behavior/task-delaydiscount_events.tsv'
    behavior_data = pd.read_csv(behavior_file, sep='\t')
    
    # Load neural data
    neural_file = f'{data_path}/func/roi_{roi_name}_patterns.npy'
    neural_patterns = np.load(neural_file)  # Shape: (n_trials, n_voxels)
    
    # Organize by delay condition
    delay_conditions = {}
    unique_delays = sorted(behavior_data['delay'].unique())
    
    for delay in unique_delays:
        delay_mask = behavior_data['delay'] == delay
        delay_conditions[delay] = {
            'patterns': neural_patterns[delay_mask],
            'choices': behavior_data.loc[delay_mask, 'choice'].values,
            'values': behavior_data.loc[delay_mask, 'chosen_value'].values,
            'value_diff': (behavior_data.loc[delay_mask, 'chosen_value'] - 
                          behavior_data.loc[delay_mask, 'unchosen_value']).values
        }
    
    return delay_conditions, unique_delays

def compute_geometric_properties(X_reduced, labels, values):
    """
    Compute comprehensive geometric properties of neural representations.
    
    Parameters:
    -----------
    X_reduced : np.ndarray
        Low-dimensional neural patterns (n_trials, n_components)
    labels : np.ndarray
        Choice labels (0=SS, 1=LL)
    values : np.ndarray
        Value differences
    
    Returns:
    --------
    dict
        Dictionary of geometric properties
    """
    properties = {}
    
    # 1. Centroid locations
    centroid_ss = np.mean(X_reduced[labels == 0], axis=0)
    centroid_ll = np.mean(X_reduced[labels == 1], axis=0)
    properties['centroid_distance'] = np.linalg.norm(centroid_ss - centroid_ll)
    properties['centroid_ss'] = centroid_ss
    properties['centroid_ll'] = centroid_ll
    
    # 2. Cluster spread (within-condition variance)
    spread_ss = np.mean([np.linalg.norm(x - centroid_ss) for x in X_reduced[labels == 0]])
    spread_ll = np.mean([np.linalg.norm(x - centroid_ll) for x in X_reduced[labels == 1]])
    properties['spread_ss'] = spread_ss
    properties['spread_ll'] = spread_ll
    properties['spread_ratio'] = spread_ss / spread_ll if spread_ll > 0 else np.inf
    
    # 3. Separability index
    between_dist = np.mean(euclidean_distances(X_reduced[labels == 0], 
                                              X_reduced[labels == 1]))
    within_dist_ss = np.mean(pdist(X_reduced[labels == 0]))
    within_dist_ll = np.mean(pdist(X_reduced[labels == 1]))
    within_dist_avg = (within_dist_ss + within_dist_ll) / 2
    properties['separability'] = between_dist / within_dist_avg if within_dist_avg > 0 else np.inf
    
    # 4. Value gradient strength
    # Compute correlation between position in space and value
    if len(values) > 3:  # Need minimum samples for correlation
        value_corr_pc1, _ = pearsonr(X_reduced[:, 0], values)
        value_corr_pc2, _ = pearsonr(X_reduced[:, 1], values)
        properties['value_gradient_pc1'] = value_corr_pc1
        properties['value_gradient_pc2'] = value_corr_pc2
        properties['value_gradient_magnitude'] = np.sqrt(value_corr_pc1**2 + value_corr_pc2**2)
    else:
        properties['value_gradient_pc1'] = 0
        properties['value_gradient_pc2'] = 0
        properties['value_gradient_magnitude'] = 0
    
    # 5. Dimensionality (effective rank)
    # Compute effective dimensionality using participation ratio
    eigenvals = np.linalg.eigvals(np.cov(X_reduced.T))
    eigenvals = eigenvals[eigenvals > 0]  # Remove zero eigenvalues
    if len(eigenvals) > 0:
        properties['dimensionality'] = (np.sum(eigenvals)**2) / np.sum(eigenvals**2)
    else:
        properties['dimensionality'] = 0
    
    # 6. Anisotropy (how stretched vs spherical the distribution is)
    if len(eigenvals) >= 2:
        properties['anisotropy'] = eigenvals[0] / eigenvals[-1]
    else:
        properties['anisotropy'] = 1
    
    return properties

def analyze_delay_progression(delay_conditions, unique_delays):
    """
    Analyze how geometric properties change across delay conditions.
    
    Parameters:
    -----------
    delay_conditions : dict
        Neural data organized by delay
    unique_delays : list
        Sorted list of delay values
    
    Returns:
    --------
    dict
        Analysis results across delays
    """
    results = {
        'delays': unique_delays,
        'properties': {},
        'embeddings': {},
        'transformations': {}
    }
    
    # Apply PCA to each delay condition
    for delay in unique_delays:
        patterns = delay_conditions[delay]['patterns']
        choices = delay_conditions[delay]['choices']
        values = delay_conditions[delay]['value_diff']
        
        # Dimensionality reduction
        pca = PCA(n_components=min(10, patterns.shape[1]))
        X_reduced = pca.fit_transform(patterns)
        
        # Store embeddings
        results['embeddings'][delay] = {
            'patterns': X_reduced,
            'choices': choices,
            'values': values,
            'pca': pca
        }
        
        # Compute geometric properties
        properties = compute_geometric_properties(X_reduced, choices, values)
        results['properties'][delay] = properties
    
    # Analyze transformations between consecutive delays
    for i in range(len(unique_delays) - 1):
        delay1, delay2 = unique_delays[i], unique_delays[i + 1]
        
        # Procrustes analysis to find optimal transformation
        transformation = compute_procrustes_transformation(
            results['embeddings'][delay1]['patterns'],
            results['embeddings'][delay2]['patterns']
        )
        
        results['transformations'][f'{delay1}_to_{delay2}'] = transformation
    
    return results

def compute_procrustes_transformation(X1, X2):
    """
    Compute Procrustes transformation between two sets of points.
    
    This finds the optimal rotation, scaling, and translation to align
    two point clouds, which tells us about systematic geometric changes.
    """
    from scipy.spatial.distance import procrustes
    
    # Ensure same number of points (use minimum)
    n_points = min(X1.shape[0], X2.shape[0])
    X1_sub = X1[:n_points]
    X2_sub = X2[:n_points]
    
    # Perform Procrustes analysis
    mtx1, mtx2, disparity = procrustes(X1_sub, X2_sub)
    
    return {
        'disparity': disparity,  # How well the shapes can be aligned
        'aligned_X1': mtx1,
        'aligned_X2': mtx2,
        'transformation_quality': 1 - disparity  # Higher = better alignment
    }

def compute_delay_trajectory_metrics(results):
    """
    Compute metrics that describe how representations change across delays.
    """
    delays = results['delays']
    properties = results['properties']
    
    trajectory_metrics = {}
    
    # 1. Monotonicity of changes
    property_names = ['centroid_distance', 'separability', 'value_gradient_magnitude', 
                     'dimensionality', 'anisotropy']
    
    for prop_name in property_names:
        values = [properties[delay][prop_name] for delay in delays]
        
        # Compute trend (correlation with delay)
        trend_corr, trend_p = spearmanr(delays, values)
        trajectory_metrics[f'{prop_name}_trend'] = trend_corr
        trajectory_metrics[f'{prop_name}_trend_p'] = trend_p
        
        # Compute variability
        trajectory_metrics[f'{prop_name}_variability'] = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
    
    # 2. Transformation quality across delays
    transformation_qualities = []
    for key, trans in results['transformations'].items():
        transformation_qualities.append(trans['transformation_quality'])
    
    trajectory_metrics['mean_transformation_quality'] = np.mean(transformation_qualities)
    trajectory_metrics['transformation_consistency'] = 1 - np.std(transformation_qualities)
    
    return trajectory_metrics

def visualize_delay_geometry(results, roi_name, save_path):
    """
    Create comprehensive visualizations of delay-dependent geometry changes.
    """
    delays = results['delays']
    properties = results['properties']
    embeddings = results['embeddings']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. PCA embeddings for each delay
    n_delays = len(delays)
    n_cols = min(4, n_delays)
    n_rows = int(np.ceil(n_delays / n_cols))
    
    for i, delay in enumerate(delays):
        ax = plt.subplot(n_rows + 2, n_cols, i + 1)
        
        X = embeddings[delay]['patterns']
        choices = embeddings[delay]['choices']
        values = embeddings[delay]['values']
        
        # Plot colored by choice
        scatter = ax.scatter(X[:, 0], X[:, 1], c=choices, cmap='coolwarm', 
                           s=50, alpha=0.7)
        ax.set_title(f'Delay: {delay}d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        
        if i == 0:
            plt.colorbar(scatter, ax=ax, label='Choice (SS=0, LL=1)')
    
    # 2. Property trajectories across delays
    ax_traj = plt.subplot(n_rows + 2, 1, n_rows + 1)
    
    property_names = ['centroid_distance', 'separability', 'value_gradient_magnitude']
    colors = ['blue', 'red', 'green']
    
    for prop_name, color in zip(property_names, colors):
        values = [properties[delay][prop_name] for delay in delays]
        ax_traj.plot(delays, values, 'o-', color=color, label=prop_name, linewidth=2)
    
    ax_traj.set_xlabel('Delay (days)')
    ax_traj.set_ylabel('Property Value')
    ax_traj.set_title(f'{roi_name}: Geometric Properties Across Delays')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    
    # 3. Transformation quality heatmap
    ax_trans = plt.subplot(n_rows + 2, 1, n_rows + 2)
    
    # Create transformation matrix
    trans_matrix = np.zeros((len(delays), len(delays)))
    for i, delay1 in enumerate(delays):
        for j, delay2 in enumerate(delays):
            if i != j and f'{delay1}_to_{delay2}' in results['transformations']:
                trans_matrix[i, j] = results['transformations'][f'{delay1}_to_{delay2}']['transformation_quality']
            elif i == j:
                trans_matrix[i, j] = 1.0  # Perfect self-alignment
    
    im = ax_trans.imshow(trans_matrix, cmap='viridis', vmin=0, vmax=1)
    ax_trans.set_xticks(range(len(delays)))
    ax_trans.set_yticks(range(len(delays)))
    ax_trans.set_xticklabels([f'{d}d' for d in delays])
    ax_trans.set_yticklabels([f'{d}d' for d in delays])
    ax_trans.set_xlabel('Target Delay')
    ax_trans.set_ylabel('Source Delay')
    ax_trans.set_title('Transformation Quality Between Delays')
    plt.colorbar(im, ax=ax_trans, label='Alignment Quality')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{roi_name}_delay_geometry.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_roi_delay_effects(roi_results):
    """
    Compare delay effects across different brain regions.
    
    Parameters:
    -----------
    roi_results : dict
        Results for each ROI
    
    Returns:
    --------
    dict
        Cross-ROI comparison results
    """
    comparison = {
        'roi_names': list(roi_results.keys()),
        'delay_sensitivity': {},
        'property_correlations': {}
    }
    
    # Compute delay sensitivity for each ROI
    for roi_name, results in roi_results.items():
        trajectory_metrics = compute_delay_trajectory_metrics(results)
        
        # Overall delay sensitivity (average absolute trend correlations)
        trend_correlations = [abs(v) for k, v in trajectory_metrics.items() 
                            if k.endswith('_trend')]
        comparison['delay_sensitivity'][roi_name] = np.mean(trend_correlations)
    
    # Compare property trajectories across ROIs
    property_names = ['centroid_distance', 'separability', 'value_gradient_magnitude']
    
    for prop_name in property_names:
        roi_trajectories = {}
        
        for roi_name, results in roi_results.items():
            delays = results['delays']
            values = [results['properties'][delay][prop_name] for delay in delays]
            roi_trajectories[roi_name] = values
        
        # Compute correlations between ROI trajectories
        roi_names = list(roi_trajectories.keys())
        corr_matrix = np.zeros((len(roi_names), len(roi_names)))
        
        for i, roi1 in enumerate(roi_names):
            for j, roi2 in enumerate(roi_names):
                if i != j:
                    corr, _ = pearsonr(roi_trajectories[roi1], roi_trajectories[roi2])
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = 1.0
        
        comparison['property_correlations'][prop_name] = {
            'matrix': corr_matrix,
            'roi_names': roi_names
        }
    
    return comparison

def main_delay_geometry_analysis(subject_ids, roi_names=['striatum', 'dlpfc', 'vmpfc']):
    """
    Main function to run delay geometry analysis across subjects and ROIs.
    """
    print("Starting Delay Geometry Analysis")
    print("=" * 50)
    
    os.makedirs('results/delay_geometry', exist_ok=True)
    
    all_results = {}
    
    for subject_id in subject_ids:
        print(f"\nProcessing Subject {subject_id}")
        subject_results = {}
        
        for roi_name in roi_names:
            print(f"  Analyzing {roi_name}...")
            
            # Load delay-specific data
            delay_conditions, unique_delays = load_delay_specific_data(subject_id, roi_name)
            
            # Analyze delay progression
            results = analyze_delay_progression(delay_conditions, unique_delays)
            
            # Compute trajectory metrics
            trajectory_metrics = compute_delay_trajectory_metrics(results)
            results['trajectory_metrics'] = trajectory_metrics
            
            # Store results
            subject_results[roi_name] = results
            
            # Create visualizations
            visualize_delay_geometry(results, f'{subject_id}_{roi_name}', 
                                   'results/delay_geometry')
        
        # Compare across ROIs for this subject
        roi_comparison = compare_roi_delay_effects(subject_results)
        
        all_results[subject_id] = {
            'roi_results': subject_results,
            'roi_comparison': roi_comparison
        }
    
    # Save comprehensive results
    np.save('results/delay_geometry/all_delay_results.npy', all_results)
    
    # Create summary report
    create_summary_report(all_results)
    
    print("\nDelay Geometry Analysis Complete!")
    print("Results saved in 'results/delay_geometry/'")

def create_summary_report(all_results):
    """
    Create a summary report of delay geometry findings.
    """
    print("\n" + "=" * 60)
    print("DELAY GEOMETRY ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Aggregate across subjects
    roi_names = list(next(iter(all_results.values()))['roi_results'].keys())
    
    for roi_name in roi_names:
        print(f"\n{roi_name.upper()} Results:")
        print("-" * 30)
        
        # Collect delay sensitivity across subjects
        sensitivities = []
        for subject_id, results in all_results.items():
            sensitivity = results['roi_comparison']['delay_sensitivity'][roi_name]
            sensitivities.append(sensitivity)
        
        mean_sensitivity = np.mean(sensitivities)
        std_sensitivity = np.std(sensitivities)
        
        print(f"Delay Sensitivity: {mean_sensitivity:.3f} ± {std_sensitivity:.3f}")
        
        if mean_sensitivity > 0.3:
            print("→ High delay sensitivity - geometry changes significantly with delay")
        elif mean_sensitivity > 0.15:
            print("→ Moderate delay sensitivity - some geometric changes")
        else:
            print("→ Low delay sensitivity - stable geometry across delays")
    
    print(f"\nAnalyzed {len(all_results)} subjects across {len(roi_names)} ROIs")
    print("Check individual visualizations for detailed patterns!")

if __name__ == "__main__":
    # Example usage
    subject_ids = [f"{i:03d}" for i in range(1, 11)]  # First 10 subjects
    main_delay_geometry_analysis(subject_ids) 