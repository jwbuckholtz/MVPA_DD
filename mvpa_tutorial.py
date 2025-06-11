#!/usr/bin/env python3
"""
MVPA and Representational Geometry Tutorial
==========================================

This tutorial walks through Multi-Voxel Pattern Analysis (MVPA) and representational 
geometry concepts using delay discounting fMRI data.

Learning Objectives:
1. Understand what MVPA is and why it's useful
2. Learn about representational geometry and neural coding
3. Understand dimensionality reduction and its applications
4. Learn statistical validation techniques
5. Practice Python programming for neuroscience

Author: Tutorial for Stanford Cognitive Neuroscience Lab
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr, ttest_ind
from scipy.spatial.distance import pdist, squareform

print("MVPA Tutorial Starting...")

# Basic demonstration
np.random.seed(42)
condition1 = np.array([1, -1, 1, -1, 1]) + np.random.normal(0, 0.5, 5)
condition2 = np.array([-1, 1, -1, 1, -1]) + np.random.normal(0, 0.5, 5)

print("Condition 1 pattern:", condition1.round(2))
print("Condition 2 pattern:", condition2.round(2))

pattern_corr = np.corrcoef(condition1, condition2)[0, 1]
print(f"Pattern correlation: r={pattern_corr:.3f}")
print("Patterns are clearly different despite individual voxel similarities!")

# =============================================================================
# SECTION 1: WHAT IS MVPA AND WHY DO WE USE IT?
# =============================================================================

"""
MVPA (Multi-Voxel Pattern Analysis) Concepts:

Traditional fMRI analysis looks at each voxel independently:
- "Is this voxel more active in condition A vs B?"
- Univariate: one voxel at a time

MVPA looks at patterns across multiple voxels:
- "What information is encoded in the pattern of activity across voxels?"
- Multivariate: many voxels together

Key Insight: The brain might encode information in patterns, not just 
individual voxel activations. For example:
- Individual voxels might not show differences between conditions
- But the pattern across voxels might be very different
"""

def demonstrate_mvpa_concept():
    """
    Simple demonstration of why MVPA is powerful.
    """
    print("=== MVPA Concept Demonstration ===")
    
    # Create synthetic data for two conditions
    np.random.seed(42)  # For reproducible results
    
    # Condition 1: Pattern [1, -1, 1, -1, 1]
    condition1 = np.array([1, -1, 1, -1, 1]) + np.random.normal(0, 0.5, 5)
    
    # Condition 2: Pattern [-1, 1, -1, 1, -1] 
    condition2 = np.array([-1, 1, -1, 1, -1]) + np.random.normal(0, 0.5, 5)
    
    print("Condition 1 pattern:", condition1.round(2))
    print("Condition 2 pattern:", condition2.round(2))
    
    # Univariate analysis: t-test on each voxel
    print("\nUnivariate analysis (each voxel separately):")
    for i in range(5):
        t_stat, p_val = ttest_ind([condition1[i]], [condition2[i]])
        print(f"Voxel {i+1}: t={t_stat:.2f}, p={p_val:.3f}")
    
    # MVPA analysis: pattern correlation
    pattern_corr = np.corrcoef(condition1, condition2)[0, 1]
    print(f"\nMVPA analysis (pattern correlation): r={pattern_corr:.3f}")
    print("Notice: Individual voxels might not be significant,")
    print("but the overall patterns are clearly different!")

# =============================================================================
# SECTION 2: REPRESENTATIONAL GEOMETRY CONCEPTS
# =============================================================================

"""
Representational Geometry Concepts:

The brain represents information in high-dimensional spaces. Each neural 
population creates a "representational space" where:

1. Each stimulus/condition is a point in this space
2. Similar stimuli are close together
3. Different stimuli are far apart
4. The geometry of this space tells us about neural coding

Key Questions:
- How are different conditions represented?
- Do similar conditions have similar neural patterns?
- How does the geometry change across brain regions?
"""

def explain_representational_space():
    """
    Demonstrate representational space concepts with simple examples.
    """
    print("\n=== Representational Space Demonstration ===")
    
    # Create example neural patterns for different stimuli
    np.random.seed(42)
    
    # Imagine we have 3 stimuli and 100 neurons
    n_neurons = 100
    
    # Stimulus A: High activity in first 50 neurons
    stim_A = np.concatenate([np.random.normal(2, 0.5, 50), 
                            np.random.normal(0, 0.5, 50)])
    
    # Stimulus B: High activity in last 50 neurons  
    stim_B = np.concatenate([np.random.normal(0, 0.5, 50),
                            np.random.normal(2, 0.5, 50)])
    
    # Stimulus C: Similar to A but with some differences
    stim_C = stim_A + np.random.normal(0, 0.3, 100)
    
    # Calculate distances between stimuli
    dist_AB = np.linalg.norm(stim_A - stim_B)
    dist_AC = np.linalg.norm(stim_A - stim_C)
    dist_BC = np.linalg.norm(stim_B - stim_C)
    
    print(f"Distance A-B: {dist_AB:.2f}")
    print(f"Distance A-C: {dist_AC:.2f}")  # Should be smaller
    print(f"Distance B-C: {dist_BC:.2f}")
    
    print("\nInterpretation:")
    print("- A and C are more similar (smaller distance)")
    print("- This suggests they might be processed similarly")
    print("- The geometry tells us about neural coding!")

# =============================================================================
# SECTION 3: DIMENSIONALITY REDUCTION - WHY AND HOW
# =============================================================================

"""
Dimensionality Reduction Concepts:

Problem: fMRI data is high-dimensional
- Each voxel is a dimension
- ROIs might have 1000+ voxels
- Hard to visualize and analyze

Solution: Reduce to fewer dimensions while preserving important information

Common Methods:
1. PCA (Principal Component Analysis): Find directions of maximum variance
2. LDA (Linear Discriminant Analysis): Find directions that separate conditions
3. t-SNE: Preserve local neighborhood structure
"""

def demonstrate_pca():
    """
    Show how PCA works with a simple example.
    """
    print("\n=== PCA Demonstration ===")
    
    # Create correlated 2D data
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 0.5, 100)  # y is correlated with x
    
    data = np.column_stack([x, y])
    
    # Apply PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    print("Original data shape:", data.shape)
    print("PCA data shape:", data_pca.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
    
    # Plot to visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(data[:, 0], data[:, 1], alpha=0.6)
    ax1.set_title('Original Data')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    
    ax2.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6)
    ax2.set_title('PCA-transformed Data')
    ax2.set_xlabel('PC1 (explains {:.1%})'.format(pca.explained_variance_ratio_[0]))
    ax2.set_ylabel('PC2 (explains {:.1%})'.format(pca.explained_variance_ratio_[1]))
    
    plt.tight_layout()
    plt.savefig('pca_demonstration.png')
    plt.close()
    
    print("Key insight: PC1 captures most of the variance!")
    print("We could use just PC1 and lose little information.")

# =============================================================================
# SECTION 4: PATTERN DISTINCTNESS - MEASURING SEPARABILITY
# =============================================================================

def explain_pattern_distinctness():
    """
    Explain what pattern distinctness means and how to measure it.
    """
    print("\n=== Pattern Distinctness Concepts ===")
    
    print("""
    Pattern Distinctness asks: "How different are neural patterns between conditions?"
    
    Method:
    1. Calculate distances within each condition (how variable is each condition?)
    2. Calculate distances between conditions (how different are the conditions?)
    3. Compare: Are between-condition distances larger than within-condition distances?
    
    Interpretation:
    - High distinctness: Conditions have very different neural patterns
    - Low distinctness: Conditions have similar neural patterns
    """)

def compute_pattern_distinctness_tutorial(X, labels):
    """
    Tutorial version of pattern distinctness with detailed explanations.
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns (n_trials x n_features)
    labels : np.ndarray  
        Condition labels (0 or 1)
    """
    print("\n=== Computing Pattern Distinctness ===")
    
    # Step 1: Split data by condition
    X_condition0 = X[labels == 0]  # e.g., sooner-smaller choices
    X_condition1 = X[labels == 1]  # e.g., larger-later choices
    
    print(f"Condition 0 has {len(X_condition0)} trials")
    print(f"Condition 1 has {len(X_condition1)} trials")
    
    # Step 2: Calculate within-condition distances
    # pdist calculates pairwise distances between all points
    distances_within_0 = pdist(X_condition0, metric='euclidean')
    distances_within_1 = pdist(X_condition1, metric='euclidean')
    
    print(f"Within-condition 0 distances: {len(distances_within_0)} pairs")
    print(f"Within-condition 1 distances: {len(distances_within_1)} pairs")
    
    # Step 3: Calculate between-condition distances
    # This compares every trial in condition 0 to every trial in condition 1
    from sklearn.metrics.pairwise import euclidean_distances
    distances_between = euclidean_distances(X_condition0, X_condition1)
    distances_between = distances_between.flatten()  # Make it 1D
    
    print(f"Between-condition distances: {len(distances_between)} pairs")
    
    # Step 4: Compare the distributions
    mean_within_0 = np.mean(distances_within_0)
    mean_within_1 = np.mean(distances_within_1)
    mean_between = np.mean(distances_between)
    
    print(f"\nMean distance within condition 0: {mean_within_0:.3f}")
    print(f"Mean distance within condition 1: {mean_within_1:.3f}")
    print(f"Mean distance between conditions: {mean_between:.3f}")
    
    # Step 5: Calculate effect size (Cohen's d)
    pooled_within = np.concatenate([distances_within_0, distances_within_1])
    all_distances = np.concatenate([distances_between, pooled_within])
    
    cohens_d = (mean_between - np.mean(pooled_within)) / np.std(all_distances)
    
    print(f"Cohen's d (effect size): {cohens_d:.3f}")
    print("Interpretation:")
    if cohens_d > 0.8:
        print("- Large effect: Conditions are very distinct")
    elif cohens_d > 0.5:
        print("- Medium effect: Conditions are moderately distinct")
    elif cohens_d > 0.2:
        print("- Small effect: Conditions are somewhat distinct")
    else:
        print("- No effect: Conditions are not distinct")
    
    return {
        'within_0_mean': mean_within_0,
        'within_1_mean': mean_within_1,
        'between_mean': mean_between,
        'cohens_d': cohens_d
    }

# =============================================================================
# SECTION 5: REPRESENTATIONAL SIMILARITY ANALYSIS (RSA)
# =============================================================================

def explain_rsa():
    """
    Explain RSA concepts and implementation.
    """
    print("\n=== Representational Similarity Analysis (RSA) ===")
    
    print("""
    RSA asks: "What is the structure of similarity between different stimuli?"
    
    Method:
    1. Calculate similarity (or distance) between every pair of stimuli
    2. Create a Representational Similarity Matrix (RSM)
    3. Compare RSMs between conditions, brain regions, or models
    
    Key Insight: The RSM captures the "geometry" of the representational space
    - Similar stimuli have high correlations (warm colors in heatmap)
    - Different stimuli have low correlations (cool colors in heatmap)
    """)

def compute_rsa_tutorial(X1, X2, condition_names=['Condition 1', 'Condition 2']):
    """
    Tutorial version of RSA with detailed explanations.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Neural patterns for two conditions
    condition_names : list
        Names for the conditions
    """
    print(f"\n=== Computing RSA for {condition_names[0]} vs {condition_names[1]} ===")
    
    # Step 1: Compute similarity matrices for each condition
    print("Step 1: Computing similarity matrices...")
    
    # We use correlation as our similarity metric
    # pdist with 'correlation' gives us 1 - correlation, so we need to convert
    rsm1_distances = pdist(X1, metric='correlation')
    rsm2_distances = pdist(X2, metric='correlation')
    
    # Convert to correlation (similarity) matrices
    rsm1 = 1 - squareform(rsm1_distances)
    rsm2 = 1 - squareform(rsm2_distances)
    
    print(f"RSM shape: {rsm1.shape} (each condition has {len(X1)} trials)")
    
    # Step 2: Compare the RSMs
    print("Step 2: Comparing RSM structures...")
    
    # Extract upper triangle (avoid diagonal and redundant lower triangle)
    upper_triangle_indices = np.triu_indices_from(rsm1, k=1)
    rsm1_upper = rsm1[upper_triangle_indices]
    rsm2_upper = rsm2[upper_triangle_indices]
    
    # Correlate the RSMs
    rsm_correlation, p_value = pearsonr(rsm1_upper, rsm2_upper)
    
    print(f"RSM correlation: {rsm_correlation:.3f} (p = {p_value:.3f})")
    print("Interpretation:")
    if rsm_correlation > 0.7:
        print("- High correlation: Very similar representational structure")
    elif rsm_correlation > 0.3:
        print("- Moderate correlation: Somewhat similar structure")
    else:
        print("- Low correlation: Different representational structures")
    
    # Step 3: Visualize the RSMs
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot RSM 1
    im1 = ax1.imshow(rsm1, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax1.set_title(f'{condition_names[0]} RSM')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Trial')
    plt.colorbar(im1, ax=ax1, label='Correlation')
    
    # Plot RSM 2
    im2 = ax2.imshow(rsm2, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax2.set_title(f'{condition_names[1]} RSM')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Trial')
    plt.colorbar(im2, ax=ax2, label='Correlation')
    
    # Plot RSM comparison
    ax3.scatter(rsm1_upper, rsm2_upper, alpha=0.5)
    ax3.plot([-1, 1], [-1, 1], 'r--', label='Perfect correlation')
    ax3.set_xlabel(f'{condition_names[0]} similarity')
    ax3.set_ylabel(f'{condition_names[1]} similarity')
    ax3.set_title(f'RSM Comparison\nr = {rsm_correlation:.3f}')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('rsa_tutorial.png')
    plt.close()
    
    return {
        'rsm1': rsm1,
        'rsm2': rsm2,
        'correlation': rsm_correlation,
        'p_value': p_value
    }

# =============================================================================
# SECTION 6: CROSS-VALIDATION - WHY IT'S CRUCIAL
# =============================================================================

def explain_cross_validation():
    """
    Explain why cross-validation is essential in MVPA.
    """
    print("\n=== Cross-Validation in MVPA ===")
    
    print("""
    Why do we need cross-validation?
    
    Problem: Overfitting
    - With many features (voxels) and few samples (trials), models can memorize noise
    - Results might not generalize to new data
    - This gives false confidence in our findings
    
    Solution: Cross-validation
    - Train model on subset of data
    - Test on independent subset
    - Repeat multiple times
    - Average results across folds
    
    This gives us:
    1. Honest estimate of performance
    2. Measure of reliability (standard deviation across folds)
    3. Protection against overfitting
    """)

def demonstrate_cross_validation():
    """
    Show the importance of cross-validation with a simple example.
    """
    print("\n=== Cross-Validation Demonstration ===")
    
    # Create synthetic data where there's actually no signal
    np.random.seed(42)
    n_samples, n_features = 50, 1000  # Few samples, many features
    X = np.random.normal(0, 1, (n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)  # Random labels
    
    print(f"Data: {n_samples} samples, {n_features} features")
    print("Ground truth: NO real signal (labels are random)")
    
    # Without cross-validation (training and testing on same data)
    from sklearn.svm import SVC
    classifier = SVC()
    classifier.fit(X, y)
    train_accuracy = classifier.score(X, y)
    
    print(f"\nWithout CV (train=test): {train_accuracy:.3f} accuracy")
    print("This looks great! But it's misleading...")
    
    # With cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(classifier, X, y, cv=5)
    
    print(f"With 5-fold CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("This is close to chance (0.5), which is the truth!")
    
    print("\nLesson: Always use cross-validation in MVPA!")

# =============================================================================
# SECTION 7: PUTTING IT ALL TOGETHER - COMPLETE ANALYSIS
# =============================================================================

def complete_mvpa_analysis_tutorial():
    """
    Demonstrate a complete MVPA analysis with synthetic delay discounting data.
    """
    print("\n" + "="*60)
    print("COMPLETE MVPA ANALYSIS TUTORIAL")
    print("="*60)
    
    # Step 1: Create synthetic delay discounting data
    print("\nStep 1: Creating synthetic delay discounting data...")
    
    np.random.seed(42)
    n_trials = 100
    n_voxels = 200
    
    # Create two conditions: sooner-smaller (SS) vs larger-later (LL)
    # SS trials: activity pattern A
    # LL trials: activity pattern B
    
    # Pattern A: higher activity in first half of voxels
    pattern_A = np.concatenate([np.random.normal(1, 0.5, n_voxels//2),
                               np.random.normal(0, 0.5, n_voxels//2)])
    
    # Pattern B: higher activity in second half of voxels  
    pattern_B = np.concatenate([np.random.normal(0, 0.5, n_voxels//2),
                               np.random.normal(1, 0.5, n_voxels//2)])
    
    # Create trials (50 of each condition)
    X_ss = np.array([pattern_A + np.random.normal(0, 0.3, n_voxels) 
                     for _ in range(50)])
    X_ll = np.array([pattern_B + np.random.normal(0, 0.3, n_voxels) 
                     for _ in range(50)])
    
    # Combine data
    X = np.vstack([X_ss, X_ll])
    labels = np.array([0]*50 + [1]*50)  # 0=SS, 1=LL
    
    # Create value differences (how much more valuable the chosen option was)
    value_diff = np.random.normal(0, 1, n_trials)
    
    print(f"Created {n_trials} trials with {n_voxels} voxels each")
    print(f"Conditions: {np.sum(labels==0)} SS trials, {np.sum(labels==1)} LL trials")
    
    # Step 2: Dimensionality reduction
    print("\nStep 2: Dimensionality reduction with PCA...")
    
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    
    print(f"Reduced from {n_voxels} to {X_reduced.shape[1]} dimensions")
    print(f"Explained variance: {pca.explained_variance_ratio_[:3].round(3)}")
    
    # Step 3: Pattern distinctness analysis
    print("\nStep 3: Pattern distinctness analysis...")
    distinctness_results = compute_pattern_distinctness_tutorial(X_reduced, labels)
    
    # Step 4: RSA analysis
    print("\nStep 4: Representational Similarity Analysis...")
    X_ss_reduced = X_reduced[labels == 0]
    X_ll_reduced = X_reduced[labels == 1]
    rsa_results = compute_rsa_tutorial(X_ss_reduced, X_ll_reduced, 
                                      ['Sooner-Smaller', 'Larger-Later'])
    
    # Step 5: Cross-validated decoding
    print("\nStep 5: Cross-validated decoding...")
    
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    classifier = SVC(kernel='linear')
    cv_scores = cross_val_score(classifier, X_reduced, labels, cv=5)
    
    print(f"Decoding accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Chance level: 0.500")
    
    if cv_scores.mean() > 0.6:
        print("Good decoding! Conditions are distinguishable.")
    else:
        print("Poor decoding. Conditions might not be well separated.")
    
    # Step 6: Visualization
    print("\nStep 6: Creating visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: PCA space colored by condition
    scatter = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='coolwarm')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Neural Patterns in PC Space')
    plt.colorbar(scatter, ax=ax1, label='Condition (SS=0, LL=1)')
    
    # Plot 2: PCA space colored by value difference
    scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=value_diff, cmap='RdYlBu')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('Value Differences in PC Space')
    plt.colorbar(scatter2, ax=ax2, label='Value Difference')
    
    # Plot 3: Pattern distances
    distances = [distinctness_results['within_0_mean'],
                distinctness_results['within_1_mean'], 
                distinctness_results['between_mean']]
    distance_labels = ['Within SS', 'Within LL', 'Between']
    bars = ax3.bar(distance_labels, distances, color=['lightblue', 'lightcoral', 'gold'])
    ax3.set_ylabel('Mean Distance')
    ax3.set_title(f'Pattern Distinctness (d={distinctness_results["cohens_d"]:.2f})')
    
    # Plot 4: Cross-validation results
    ax4.bar(['Decoding\nAccuracy'], [cv_scores.mean()], 
           yerr=[cv_scores.std()], color='green', alpha=0.7)
    ax4.axhline(y=0.5, color='red', linestyle='--', label='Chance')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Cross-validated Decoding')
    ax4.legend()
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('complete_mvpa_tutorial.png')
    plt.close()
    
    print("\nAnalysis complete! Check the generated plots.")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"1. Pattern Distinctness: d = {distinctness_results['cohens_d']:.3f}")
    print(f"2. RSM Correlation: r = {rsa_results['correlation']:.3f}")
    print(f"3. Decoding Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"4. Explained Variance (PC1): {pca.explained_variance_ratio_[0]:.1%}")
    
    print("\nInterpretation Guide:")
    print("- High distinctness (d > 0.5): Conditions have different neural patterns")
    print("- High RSM correlation (r > 0.3): Similar representational structure")
    print("- Above-chance decoding (> 0.5): Conditions are distinguishable")
    print("- High explained variance: PCA captures meaningful structure")

# =============================================================================
# MAIN TUTORIAL EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("MVPA AND REPRESENTATIONAL GEOMETRY TUTORIAL")
    print("=" * 50)
    
    # Run all tutorial sections
    demonstrate_mvpa_concept()
    explain_representational_space()
    demonstrate_pca()
    explain_pattern_distinctness()
    explain_rsa()
    explain_cross_validation()
    demonstrate_cross_validation()
    complete_mvpa_analysis_tutorial()
    
    print("\n" + "="*60)
    print("TUTORIAL COMPLETE!")
    print("="*60)
    print("Key files generated:")
    print("- pca_demonstration.png")
    print("- rsa_tutorial.png") 
    print("- complete_mvpa_tutorial.png")
    print("\nNext steps:")
    print("1. Adapt this code for your real fMRI data")
    print("2. Experiment with different parameters")
    print("3. Try different dimensionality reduction methods")
    print("4. Add more sophisticated statistical tests") 