#!/usr/bin/env python3
"""
Shared Geometry Utilities
=========================

Consolidated geometric analysis functions used across multiple modules.
This module eliminates code duplication by providing unified implementations
of advanced geometric analysis methods.

Functions included:
- Manifold alignment (Procrustes, CCA)
- Information geometry metrics (KL divergence, JS divergence, Wasserstein)
- Geodesic distance analysis
- Manifold curvature estimation
- Trajectory dynamics analysis

Author: Cognitive Neuroscience Lab, Stanford University
Created: December 2024 (Consolidation from multiple modules)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# Core scientific computing
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import gaussian_kde

# Machine learning
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import CCA

# Suppress potential warnings
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# MANIFOLD ALIGNMENT FUNCTIONS
# =============================================================================

def compute_manifold_alignment(X1: np.ndarray, X2: np.ndarray, 
                             method: str = 'procrustes') -> Dict[str, Any]:
    """
    Compute alignment between two neural manifolds using various methods.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Neural patterns for two conditions (n_trials x n_features)
    method : str
        Alignment method ('procrustes', 'cca')
    
    Returns:
    --------
    Dict[str, Any]
        Alignment results and quality metrics
    """
    if method == 'procrustes':
        return compute_procrustes_alignment(X1, X2)
    elif method == 'cca':
        return compute_cca_alignment(X1, X2)
    else:
        raise ValueError(f"Unknown alignment method: {method}. Supported: 'procrustes', 'cca'")


def compute_procrustes_alignment(X1: np.ndarray, X2: np.ndarray) -> Dict[str, Any]:
    """
    Procrustes analysis for manifold alignment.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Neural patterns for two conditions
    
    Returns:
    --------
    Dict[str, Any]
        Procrustes alignment results
    """
    # Ensure same number of points
    n_points = min(X1.shape[0], X2.shape[0])
    X1_sub = X1[:n_points]
    X2_sub = X2[:n_points]
    
    # Perform Procrustes analysis
    mtx1, mtx2, disparity = procrustes(X1_sub, X2_sub)
    
    # Compute additional metrics
    alignment_quality = 1 - disparity
    
    # Compute rotation matrix
    H = X1_sub.T @ X2_sub
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    return {
        'method': 'procrustes',
        'disparity': disparity,
        'alignment_quality': alignment_quality,
        'rotation_matrix': R,
        'aligned_X1': mtx1,
        'aligned_X2': mtx2,
        'singular_values': S
    }


def compute_cca_alignment(X1: np.ndarray, X2: np.ndarray) -> Dict[str, Any]:
    """
    Canonical Correlation Analysis for manifold alignment.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Neural patterns for two conditions
    
    Returns:
    --------
    Dict[str, Any]
        CCA alignment results
    """
    # Ensure same number of points
    n_points = min(X1.shape[0], X2.shape[0])
    X1_sub = X1[:n_points]
    X2_sub = X2[:n_points]
    
    # Fit CCA
    n_components = min(X1_sub.shape[1], X2_sub.shape[1], 5)
    cca = CCA(n_components=n_components)
    X1_c, X2_c = cca.fit_transform(X1_sub, X2_sub)
    
    # Compute canonical correlations
    canonical_corrs = []
    for i in range(n_components):
        corr, _ = stats.pearsonr(X1_c[:, i], X2_c[:, i])
        canonical_corrs.append(corr)
    
    return {
        'method': 'cca',
        'canonical_correlations': canonical_corrs,
        'mean_correlation': np.mean(canonical_corrs),
        'X1_canonical': X1_c,
        'X2_canonical': X2_c,
        'cca_model': cca
    }


def compute_regression_alignment(X1, X2):
    """
    Linear regression alignment for manifold alignment.
    
    This function fits a linear regression model to align two neural 
    manifolds and computes the alignment quality based on R².
    
    Parameters:
    -----------
    X1 : np.ndarray
        First neural data matrix (n_trials, n_features)
    X2 : np.ndarray  
        Second neural data matrix (n_trials, n_features)
        
    Returns:
    --------
    dict : Dictionary containing alignment results
        - method: 'linear_regression'
        - alignment_quality: R² score of the alignment
        - coefficients: Regression coefficients
        - intercept: Regression intercept
        - predicted_X2: Predicted X2 values
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
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
    alignment_quality = r2_score(X2_sub, X2_pred, multioutput='uniform_average')
    
    return {
        'method': 'linear_regression',
        'alignment_quality': alignment_quality,
        'coefficients': reg.coef_,
        'intercept': reg.intercept_,
        'predicted_X2': X2_pred
    }


# =============================================================================
# INFORMATION GEOMETRY FUNCTIONS
# =============================================================================

def compute_information_geometry_metrics(X1: np.ndarray, X2: np.ndarray) -> Dict[str, Any]:
    """
    Compute information geometry metrics between two neural distributions.
    
    Uses kernel density estimation on PCA-reduced data for computational efficiency.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Neural patterns for two conditions
    
    Returns:
    --------
    Dict[str, Any]
        Information geometry metrics including KL divergence, JS divergence, 
        Wasserstein distance, and density estimates
    """
    # Use first two PCA components for computational efficiency
    pca = PCA(n_components=2)
    X1_pca = pca.fit_transform(X1)
    X2_pca = pca.transform(X2)
    
    # Create KDE estimates
    kde1 = gaussian_kde(X1_pca.T)
    kde2 = gaussian_kde(X2_pca.T)
    
    # Create evaluation grid
    x_min = min(X1_pca[:, 0].min(), X2_pca[:, 0].min())
    x_max = max(X1_pca[:, 0].max(), X2_pca[:, 0].max())
    y_min = min(X1_pca[:, 1].min(), X2_pca[:, 1].min())
    y_max = max(X1_pca[:, 1].max(), X2_pca[:, 1].max())
    
    xx, yy = np.mgrid[x_min:x_max:20j, y_min:y_max:20j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Evaluate densities
    p1 = kde1(positions).reshape(xx.shape)
    p2 = kde2(positions).reshape(xx.shape)
    
    # Normalize to ensure they're proper probability distributions
    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)
    
    # Compute information divergences
    kl_div = compute_kl_divergence(p1.flatten(), p2.flatten())
    js_div = compute_js_divergence(p1.flatten(), p2.flatten())
    wasserstein_dist = compute_wasserstein_approximation(X1_pca, X2_pca)
    
    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'wasserstein_distance': wasserstein_dist,
        'density_1': p1,
        'density_2': p2,
        'grid_x': xx,
        'grid_y': yy
    }


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler divergence between two probability distributions.
    
    Parameters:
    -----------
    p, q : np.ndarray
        Probability distributions (must be same shape)
    epsilon : float
        Small constant to avoid log(0)
    
    Returns:
    --------
    float
        KL divergence D(p||q)
    """
    p = p + epsilon  # Add small constant to avoid log(0)
    q = q + epsilon
    return np.sum(p * np.log(p / q))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    Parameters:
    -----------
    p, q : np.ndarray
        Probability distributions (must be same shape)
    
    Returns:
    --------
    float
        JS divergence (symmetric, bounded [0,1])
    """
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)


def compute_wasserstein_approximation(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute approximation of Wasserstein distance using sample means.
    
    Note: This is a simplified approximation. For exact Wasserstein distances,
    consider using scipy.stats.wasserstein_distance or optimal transport libraries.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Sample data from two distributions
    
    Returns:
    --------
    float
        Approximate Wasserstein distance
    """
    # Simple approximation: distance between sample means
    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)
    return np.linalg.norm(mean1 - mean2)


# =============================================================================
# GEODESIC AND CURVATURE ANALYSIS
# =============================================================================

def compute_geodesic_distances(X: np.ndarray, k: int = 5) -> Dict[str, Any]:
    """
    Compute geodesic distances on the neural manifold using Isomap.
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns (n_trials x n_features)
    k : int
        Number of nearest neighbors for manifold construction
    
    Returns:
    --------
    Dict[str, Any]
        Geodesic distance matrix and summary statistics
    """
    # Use Isomap to compute geodesic distances
    n_components = min(X.shape[1], 10)  # Limit components for efficiency
    isomap = Isomap(n_neighbors=k, n_components=n_components)
    isomap.fit(X)
    
    # Get geodesic distance matrix
    geodesic_distances = isomap.dist_matrix_
    
    return {
        'geodesic_distance_matrix': geodesic_distances,
        'mean_geodesic_distance': np.mean(geodesic_distances),
        'geodesic_variance': np.var(geodesic_distances),
        'isomap_model': isomap
    }


def compute_manifold_curvature(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Estimate local curvature of the neural manifold.
    
    Curvature is estimated based on the distribution of variance explained
    by local PCA analysis in each point's neighborhood.
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns (n_trials x n_features)
    k : int
        Number of neighbors for local analysis
    
    Returns:
    --------
    np.ndarray
        Curvature estimates for each point
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    curvatures = []
    
    for i in range(len(X)):
        # Get local neighborhood
        local_points = X[indices[i]]
        
        # Fit local PCA
        local_pca = PCA(n_components=min(3, local_points.shape[1]))
        local_pca.fit(local_points)
        
        # Curvature estimate based on explained variance ratio
        explained_ratios = local_pca.explained_variance_ratio_
        if len(explained_ratios) >= 2:
            # Higher curvature when variance is more evenly distributed
            curvature = 1 - explained_ratios[0]
        else:
            curvature = 0
        
        curvatures.append(curvature)
    
    return np.array(curvatures)


# =============================================================================
# TRAJECTORY DYNAMICS ANALYSIS
# =============================================================================

def analyze_trajectory_dynamics(embeddings_dict: Dict[str, np.ndarray], 
                               condition_labels: List[str]) -> Dict[str, Any]:
    """
    Analyze the dynamics of how neural representations change across conditions.
    
    Parameters:
    -----------
    embeddings_dict : Dict[str, np.ndarray]
        Dictionary mapping condition names to their neural embeddings
    condition_labels : List[str]
        Ordered list of condition names (e.g., delay values)
    
    Returns:
    --------
    Dict[str, Any]
        Trajectory analysis results including centroids, velocities, 
        accelerations, and trajectory metrics
    """
    results = {
        'centroid_trajectory': [],
        'velocity': [],
        'acceleration': [],
        'trajectory_length': 0,
        'trajectory_smoothness': 0
    }
    
    # Compute centroids for each condition
    centroids = []
    for condition in condition_labels:
        if condition in embeddings_dict:
            patterns = embeddings_dict[condition]
            centroid = np.mean(patterns, axis=0)
            centroids.append(centroid)
            results['centroid_trajectory'].append(centroid)
        else:
            warnings.warn(f"Condition '{condition}' not found in embeddings_dict")
    
    if len(centroids) < 2:
        warnings.warn("Need at least 2 conditions for trajectory analysis")
        return results
    
    centroids = np.array(centroids)
    
    # Compute velocity (first derivative)
    velocities = []
    for i in range(1, len(centroids)):
        velocity = centroids[i] - centroids[i-1]
        velocities.append(np.linalg.norm(velocity))
    results['velocity'] = velocities
    
    # Compute acceleration (second derivative)
    accelerations = []
    for i in range(1, len(velocities)):
        acceleration = velocities[i] - velocities[i-1]
        accelerations.append(abs(acceleration))
    results['acceleration'] = accelerations
    
    # Compute trajectory length
    trajectory_length = 0
    for i in range(1, len(centroids)):
        segment_length = np.linalg.norm(centroids[i] - centroids[i-1])
        trajectory_length += segment_length
    results['trajectory_length'] = trajectory_length
    
    # Compute trajectory smoothness (inverse of acceleration variance)
    if len(accelerations) > 0:
        results['trajectory_smoothness'] = 1 / (1 + np.var(accelerations))
    
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_input_arrays(X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize input arrays for geometric analysis.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Input neural data arrays
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Validated arrays
    
    Raises:
    -------
    ValueError
        If arrays have incompatible shapes or insufficient data
    """
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("Input arrays must be 2D (n_trials x n_features)")
    
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("Input arrays must have same number of features")
    
    if X1.shape[0] < 3 or X2.shape[0] < 3:
        raise ValueError("Need at least 3 trials per condition for robust analysis")
    
    return X1, X2


def get_available_functions() -> List[str]:
    """
    Get list of all available geometry utility functions.
    
    Returns:
    --------
    List[str]
        List of function names
    """
    return [
        'compute_manifold_alignment',
        'compute_procrustes_alignment', 
        'compute_cca_alignment',
        'compute_information_geometry_metrics',
        'compute_kl_divergence',
        'compute_js_divergence',
        'compute_wasserstein_approximation',
        'compute_geodesic_distances',
        'compute_manifold_curvature',
        'analyze_trajectory_dynamics',
        'validate_input_arrays'
    ]


# =============================================================================
# MODULE INFORMATION
# =============================================================================

__all__ = [
    'compute_manifold_alignment',
    'compute_procrustes_alignment',
    'compute_cca_alignment', 
    'compute_information_geometry_metrics',
    'compute_kl_divergence',
    'compute_js_divergence',
    'compute_wasserstein_approximation',
    'compute_geodesic_distances',
    'compute_manifold_curvature',
    'analyze_trajectory_dynamics',
    'validate_input_arrays',
    'get_available_functions'
]

__version__ = "1.0.0"
__author__ = "Cognitive Neuroscience Lab, Stanford University"
__description__ = "Consolidated geometric analysis utilities for neural data"


if __name__ == "__main__":
    # Example usage and testing
    print("Geometry Utils Module")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Available functions: {len(get_available_functions())}")
    print("\nFunction list:")
    for func in get_available_functions():
        print(f"  - {func}")
    
    # Simple test with synthetic data
    print("\nTesting with synthetic data...")
    np.random.seed(42)
    X1 = np.random.randn(50, 10)
    X2 = np.random.randn(45, 10) + 0.5
    
    try:
        # Test manifold alignment
        alignment = compute_manifold_alignment(X1, X2, method='procrustes')
        print(f"Procrustes alignment quality: {alignment['alignment_quality']:.3f}")
        
        # Test information geometry
        info_metrics = compute_information_geometry_metrics(X1, X2)
        print(f"KL divergence: {info_metrics['kl_divergence']:.3f}")
        print(f"JS divergence: {info_metrics['js_divergence']:.3f}")
        
        # Test curvature
        curvatures = compute_manifold_curvature(X1)
        print(f"Mean curvature (X1): {np.mean(curvatures):.3f}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}") 