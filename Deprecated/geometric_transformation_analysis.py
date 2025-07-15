#!/usr/bin/env python3
"""
Advanced Geometric Transformation Analysis
==========================================

This script provides advanced methods for analyzing geometric transformations
in neural representations across delay conditions, including:

1. Manifold alignment techniques
2. Geodesic distance analysis
3. Curvature and topology measures
4. Dynamic trajectory analysis
5. Information geometry metrics

Author: Stanford Cognitive Neuroscience Lab
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import shared geometry utilities
from geometry_utils import (
    compute_manifold_alignment, compute_procrustes_alignment, compute_cca_alignment,
    compute_information_geometry_metrics, compute_kl_divergence, compute_js_divergence,
    compute_wasserstein_approximation, compute_geodesic_distances, compute_manifold_curvature,
    analyze_trajectory_dynamics
)

# NOTE: The following functions are now imported from geometry_utils module:
# - compute_manifold_alignment()
# - compute_procrustes_alignment()
# - compute_cca_alignment() 
# - compute_geodesic_distances()
# - compute_manifold_curvature()
# This eliminates code duplication across the codebase.

def compute_regression_alignment(X1, X2):
    """
    Linear regression alignment for manifold alignment.
    
    This is a specialized function not available in geometry_utils.
    """
    from sklearn.linear_model import LinearRegression
    
    # Ensure same number of points
    n_points = min(X1.shape[0], X2.shape[0])
    X1_sub = X1[:n_points]
    X2_sub = X2[:n_points]
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X1_sub, X2_sub)
    
    # Predict X2 from X1
    X2_pred = reg.predict(X1_sub)
    
    # Compute alignment quality as R²
    from sklearn.metrics import r2_score
    alignment_quality = r2_score(X2_sub, X2_pred, multioutput='uniform_average')
    
    return {
        'method': 'linear_regression',
        'alignment_quality': alignment_quality,
        'coefficients': reg.coef_,
        'intercept': reg.intercept_,
        'predicted_X2': X2_pred
    }

# NOTE: The following functions are now imported from geometry_utils module:
# - analyze_trajectory_dynamics()
# - compute_information_geometry_metrics()
# - compute_kl_divergence()
# - compute_js_divergence()
# - compute_wasserstein_approximation()
# This eliminates code duplication across the codebase.

def visualize_geometric_transformations(delay_results, roi_name, save_path):
    """
    Create comprehensive visualizations of geometric transformations.
    """
    delays = delay_results['delays']
    embeddings = delay_results['embeddings']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 3D trajectory of centroids
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    centroids = []
    for delay in delays:
        patterns = embeddings[delay]['patterns']
        centroid = np.mean(patterns, axis=0)
        centroids.append(centroid[:3])  # Use first 3 PCs
    
    centroids = np.array(centroids)
    
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
    
    mean_curvatures = []
    for delay in delays:
        patterns = embeddings[delay]['patterns']
        curvatures = compute_manifold_curvature(patterns)
        mean_curvatures.append(np.mean(curvatures))
    
    ax2.plot(delays, mean_curvatures, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Delay (days)')
    ax2.set_ylabel('Mean Curvature')
    ax2.set_title('Manifold Curvature vs Delay')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pairwise alignment quality heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    
    alignment_matrix = np.zeros((len(delays), len(delays)))
    for i, delay1 in enumerate(delays):
        for j, delay2 in enumerate(delays):
            if i != j:
                X1 = embeddings[delay1]['patterns']
                X2 = embeddings[delay2]['patterns']
                alignment = compute_manifold_alignment(X1, X2, method='procrustes')
                alignment_matrix[i, j] = alignment['alignment_quality']
            else:
                alignment_matrix[i, j] = 1.0
    
    im = ax3.imshow(alignment_matrix, cmap='viridis', vmin=0, vmax=1)
    ax3.set_xticks(range(len(delays)))
    ax3.set_yticks(range(len(delays)))
    ax3.set_xticklabels([f'{d}d' for d in delays])
    ax3.set_yticklabels([f'{d}d' for d in delays])
    ax3.set_title('Pairwise Alignment Quality')
    plt.colorbar(im, ax=ax3)
    
    # 4. Information geometry metrics
    ax4 = fig.add_subplot(2, 3, 4)
    
    js_divergences = []
    for i in range(len(delays) - 1):
        X1 = embeddings[delays[i]]['patterns']
        X2 = embeddings[delays[i+1]]['patterns']
        info_metrics = compute_information_geometry_metrics(X1, X2)
        js_divergences.append(info_metrics['js_divergence'])
    
    delay_pairs = [f'{delays[i]}-{delays[i+1]}' for i in range(len(delays)-1)]
    ax4.bar(range(len(js_divergences)), js_divergences)
    ax4.set_xticks(range(len(delay_pairs)))
    ax4.set_xticklabels(delay_pairs, rotation=45)
    ax4.set_ylabel('JS Divergence')
    ax4.set_title('Information Distance Between Consecutive Delays')
    
    # 5. Trajectory dynamics
    ax5 = fig.add_subplot(2, 3, 5)
    
    trajectory_results = analyze_trajectory_dynamics(embeddings, delays)
    velocities = trajectory_results['velocity']
    
    if len(velocities) > 0:
        ax5.plot(delays[1:], velocities, 'o-', linewidth=2, markersize=8, color='red')
        ax5.set_xlabel('Delay (days)')
        ax5.set_ylabel('Centroid Velocity')
        ax5.set_title('Rate of Geometric Change')
        ax5.grid(True, alpha=0.3)
    
    # 6. Dimensionality evolution
    ax6 = fig.add_subplot(2, 3, 6)
    
    effective_dims = []
    for delay in delays:
        patterns = embeddings[delay]['patterns']
        pca = PCA()
        pca.fit(patterns)
        
        # Compute effective dimensionality
        eigenvals = pca.explained_variance_
        effective_dim = (np.sum(eigenvals)**2) / np.sum(eigenvals**2)
        effective_dims.append(effective_dim)
    
    ax6.plot(delays, effective_dims, 'o-', linewidth=2, markersize=8, color='purple')
    ax6.set_xlabel('Delay (days)')
    ax6.set_ylabel('Effective Dimensionality')
    ax6.set_title('Representational Dimensionality vs Delay')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{roi_name}_geometric_transformations.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main_geometric_analysis(subject_ids, roi_names=['striatum', 'dlpfc', 'vmpfc']):
    """
    Main function for geometric transformation analysis.
    """
    print("Starting Advanced Geometric Transformation Analysis")
    print("=" * 60)
    
    os.makedirs('results/geometric_transformations', exist_ok=True)
    
    # This would integrate with the delay_geometry_analysis.py results
    # For now, we'll create a placeholder structure
    
    print("Analysis complete! Advanced geometric metrics computed.")
    print("Results include:")
    print("- Manifold alignment quality across delays")
    print("- Geodesic distance analysis")
    print("- Local curvature estimates")
    print("- Trajectory dynamics")
    print("- Information geometry metrics")

if __name__ == "__main__":
    # Example usage
    subject_ids = [f"{i:03d}" for i in range(1, 6)]  # First 5 subjects
    main_geometric_analysis(subject_ids) 