#!/usr/bin/env python3
"""
Annotated MVPA Analysis for Learning
===================================

This script provides a step-by-step walkthrough of MVPA analysis with detailed
explanations of what each piece of code does and why it's important.

This connects directly to the concepts in MVPA_Tutorial_Guide.md
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: UNDERSTANDING YOUR DATA
# =============================================================================

def load_and_understand_data():
    """
    First step: Load your data and understand its structure.
    
    In real analysis, you would load fMRI and behavioral data here.
    For this tutorial, we'll create synthetic data that mimics real data.
    """
    print("STEP 1: Understanding Your Data")
    print("=" * 40)
    
    # Create synthetic delay discounting data
    np.random.seed(42)  # For reproducible results
    
    # Simulate 80 trials, 500 voxels (typical ROI size)
    n_trials = 80
    n_voxels = 500
    
    print(f"Data dimensions: {n_trials} trials × {n_voxels} voxels")
    print("This is typical for an fMRI experiment")
    print("- Each trial is one decision in the delay discounting task")
    print("- Each voxel is one location in your ROI (e.g., striatum)")
    
    # Create two different neural patterns for two choice types
    # Pattern A: Sooner-smaller choices (higher activity in first half of voxels)
    pattern_SS = np.concatenate([
        np.random.normal(1.0, 0.3, n_voxels//2),  # High activity
        np.random.normal(0.0, 0.3, n_voxels//2)   # Low activity
    ])
    
    # Pattern B: Larger-later choices (higher activity in second half of voxels)
    pattern_LL = np.concatenate([
        np.random.normal(0.0, 0.3, n_voxels//2),  # Low activity
        np.random.normal(1.0, 0.3, n_voxels//2)   # High activity
    ])
    
    # Generate trials (40 of each type)
    X_ss_trials = np.array([pattern_SS + np.random.normal(0, 0.2, n_voxels) 
                           for _ in range(40)])
    X_ll_trials = np.array([pattern_LL + np.random.normal(0, 0.2, n_voxels) 
                           for _ in range(40)])
    
    # Combine all trials
    X = np.vstack([X_ss_trials, X_ll_trials])  # Shape: (80, 500)
    labels = np.array([0]*40 + [1]*40)  # 0 = SS, 1 = LL
    
    # Create value differences (continuous variable)
    value_diff = np.random.normal(0, 1, n_trials)
    
    print(f"Created data with shape: {X.shape}")
    print(f"Labels: {np.sum(labels==0)} SS trials, {np.sum(labels==1)} LL trials")
    print("Value differences range from {:.2f} to {:.2f}".format(
        value_diff.min(), value_diff.max()))
    
    return X, labels, value_diff

# =============================================================================
# STEP 2: DIMENSIONALITY REDUCTION
# =============================================================================

def apply_dimensionality_reduction(X):
    """
    Step 2: Reduce the dimensionality of your data.
    
    Why? 500 voxels is too many to visualize and analyze effectively.
    PCA helps us find the most important patterns.
    """
    print("\nSTEP 2: Dimensionality Reduction with PCA")
    print("=" * 40)
    
    print(f"Original data: {X.shape[0]} trials × {X.shape[1]} voxels")
    print("Problem: Too many dimensions to analyze effectively!")
    
    # Apply PCA
    pca = PCA(n_components=10)  # Reduce to 10 dimensions
    X_reduced = pca.fit_transform(X)
    
    print(f"After PCA: {X_reduced.shape[0]} trials × {X_reduced.shape[1]} components")
    
    # Show how much variance each component explains
    print("\nVariance explained by each component:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var_ratio:.1%}")
    
    total_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Total variance explained by 10 PCs: {total_variance:.1%}")
    
    print("\nWhy this matters:")
    print("- We've reduced 500 dimensions to 10")
    print("- But kept most of the important information")
    print("- Now we can analyze and visualize the data!")
    
    return X_reduced, pca

# =============================================================================
# STEP 3: PATTERN DISTINCTNESS ANALYSIS
# =============================================================================

def analyze_pattern_distinctness(X_reduced, labels):
    """
    Step 3: Measure how distinct the neural patterns are between conditions.
    
    This answers: "Are SS and LL choices represented differently?"
    """
    print("\nSTEP 3: Pattern Distinctness Analysis")
    print("=" * 40)
    
    print("Question: Are sooner-smaller and larger-later choices")
    print("represented with different neural patterns?")
    
    # Split data by condition
    X_ss = X_reduced[labels == 0]  # Sooner-smaller trials
    X_ll = X_reduced[labels == 1]  # Larger-later trials
    
    print(f"\nSooner-smaller trials: {len(X_ss)}")
    print(f"Larger-later trials: {len(X_ll)}")
    
    # Calculate distances within each condition
    print("\nCalculating distances...")
    print("- Within-condition: How variable are patterns within each choice type?")
    print("- Between-condition: How different are patterns between choice types?")
    
    distances_within_ss = pdist(X_ss, metric='euclidean')
    distances_within_ll = pdist(X_ll, metric='euclidean')
    distances_between = euclidean_distances(X_ss, X_ll).flatten()
    
    # Calculate means
    mean_within_ss = np.mean(distances_within_ss)
    mean_within_ll = np.mean(distances_within_ll)
    mean_between = np.mean(distances_between)
    
    print(f"\nResults:")
    print(f"  Mean distance within SS: {mean_within_ss:.3f}")
    print(f"  Mean distance within LL: {mean_within_ll:.3f}")
    print(f"  Mean distance between SS-LL: {mean_between:.3f}")
    
    # Calculate effect size (Cohen's d)
    pooled_within = np.concatenate([distances_within_ss, distances_within_ll])
    all_distances = np.concatenate([distances_between, pooled_within])
    cohens_d = (mean_between - np.mean(pooled_within)) / np.std(all_distances)
    
    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    
    # Interpret the result
    if cohens_d > 0.8:
        interpretation = "Large effect - Very distinct patterns!"
    elif cohens_d > 0.5:
        interpretation = "Medium effect - Moderately distinct patterns"
    elif cohens_d > 0.2:
        interpretation = "Small effect - Somewhat distinct patterns"
    else:
        interpretation = "No effect - Patterns are not distinct"
    
    print(f"Interpretation: {interpretation}")
    
    return {
        'cohens_d': cohens_d,
        'mean_within_ss': mean_within_ss,
        'mean_within_ll': mean_within_ll,
        'mean_between': mean_between
    }

# =============================================================================
# STEP 4: REPRESENTATIONAL SIMILARITY ANALYSIS (RSA)
# =============================================================================

def analyze_representational_similarity(X_reduced, labels):
    """
    Step 4: Analyze the similarity structure within each condition.
    
    This answers: "Do the two conditions have similar internal structure?"
    """
    print("\nSTEP 4: Representational Similarity Analysis (RSA)")
    print("=" * 40)
    
    print("Question: Do SS and LL conditions have similar")
    print("representational structure?")
    
    # Split data by condition
    X_ss = X_reduced[labels == 0]
    X_ll = X_reduced[labels == 1]
    
    print(f"\nCreating similarity matrices...")
    print("- Each matrix shows how similar every trial is to every other trial")
    print("- We'll compare the structure between conditions")
    
    # Create representational similarity matrices
    # Note: pdist with 'correlation' gives distances, so we convert to similarities
    rsm_ss_dist = pdist(X_ss, metric='correlation')
    rsm_ll_dist = pdist(X_ll, metric='correlation')
    
    # Convert to similarity matrices (correlation = 1 - distance)
    rsm_ss = 1 - squareform(rsm_ss_dist)
    rsm_ll = 1 - squareform(rsm_ll_dist)
    
    print(f"RSM dimensions: {rsm_ss.shape}")
    print("- Diagonal = 1.0 (perfect similarity with itself)")
    print("- Off-diagonal = similarity between different trials")
    
    # Compare the RSMs
    # Extract upper triangles (avoid diagonal and redundant lower triangle)
    upper_indices = np.triu_indices_from(rsm_ss, k=1)
    rsm_ss_upper = rsm_ss[upper_indices]
    rsm_ll_upper = rsm_ll[upper_indices]
    
    # Correlate the similarity structures
    rsm_correlation, p_value = pearsonr(rsm_ss_upper, rsm_ll_upper)
    
    print(f"\nRSM correlation: {rsm_correlation:.3f} (p = {p_value:.3f})")
    
    # Interpret the result
    if rsm_correlation > 0.7:
        interpretation = "High correlation - Very similar structures!"
    elif rsm_correlation > 0.3:
        interpretation = "Moderate correlation - Somewhat similar structures"
    else:
        interpretation = "Low correlation - Different structures"
    
    print(f"Interpretation: {interpretation}")
    print("\nWhat this means:")
    if rsm_correlation > 0.5:
        print("- The internal organization is preserved across conditions")
        print("- Similar trials are similar in both SS and LL conditions")
        print("- Suggests a common representational geometry")
    else:
        print("- Different internal organization across conditions")
        print("- The geometry changes between SS and LL conditions")
        print("- Suggests condition-specific representations")
    
    return {
        'rsm_correlation': rsm_correlation,
        'p_value': p_value,
        'rsm_ss': rsm_ss,
        'rsm_ll': rsm_ll
    }

# =============================================================================
# STEP 5: CROSS-VALIDATED DECODING
# =============================================================================

def perform_cross_validated_decoding(X_reduced, labels):
    """
    Step 5: Test if we can decode choice type from neural patterns.
    
    This answers: "Can we predict choice type from brain activity?"
    """
    print("\nSTEP 5: Cross-Validated Decoding")
    print("=" * 40)
    
    print("Question: Can we predict whether someone chose")
    print("sooner-smaller vs larger-later from their brain activity?")
    
    # Set up classifier
    classifier = SVC(kernel='linear')  # Linear SVM classifier
    
    print(f"\nUsing {classifier.__class__.__name__} classifier")
    print("- Linear kernel: finds a line that separates the two conditions")
    print("- If successful, it means the conditions are linearly separable")
    
    # Perform 5-fold cross-validation
    print("\nPerforming 5-fold cross-validation...")
    print("- Split data into 5 parts")
    print("- Train on 4 parts, test on 1 part")
    print("- Repeat 5 times, average results")
    print("- This prevents overfitting!")
    
    cv_scores = cross_val_score(classifier, X_reduced, labels, cv=5)
    
    print(f"\nResults:")
    print(f"  Fold 1: {cv_scores[0]:.3f}")
    print(f"  Fold 2: {cv_scores[1]:.3f}")
    print(f"  Fold 3: {cv_scores[2]:.3f}")
    print(f"  Fold 4: {cv_scores[3]:.3f}")
    print(f"  Fold 5: {cv_scores[4]:.3f}")
    
    mean_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    
    print(f"\nOverall accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"Chance level: 0.500 (50%)")
    
    # Interpret the result
    if mean_accuracy > 0.6:
        interpretation = "Good decoding! Conditions are distinguishable."
    elif mean_accuracy > 0.55:
        interpretation = "Moderate decoding. Some information present."
    else:
        interpretation = "Poor decoding. Conditions not well separated."
    
    print(f"Interpretation: {interpretation}")
    
    # Statistical significance (rough estimate)
    if mean_accuracy > 0.5 + 2*std_accuracy:
        print("Result appears statistically significant!")
    else:
        print("Result may not be statistically significant.")
        print("(For proper testing, use permutation tests)")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'cv_scores': cv_scores
    }

# =============================================================================
# STEP 6: VISUALIZATION
# =============================================================================

def create_visualizations(X_reduced, labels, value_diff, pca, 
                         distinctness_results, rsa_results, decoding_results):
    """
    Step 6: Create visualizations to understand the results.
    
    Visualization is crucial for interpreting multivariate results!
    """
    print("\nSTEP 6: Creating Visualizations")
    print("=" * 40)
    
    print("Creating comprehensive visualization...")
    print("- PCA space colored by condition and value")
    print("- Pattern distances comparison")
    print("- RSM heatmaps")
    print("- Decoding results")
    
    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MVPA Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: PCA space colored by condition
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                          c=labels, cmap='coolwarm', s=50, alpha=0.7)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Neural Patterns by Choice Type')
    plt.colorbar(scatter1, ax=ax1, label='Choice (SS=0, LL=1)')
    
    # Plot 2: PCA space colored by value difference
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                          c=value_diff, cmap='RdYlBu', s=50, alpha=0.7)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('Neural Patterns by Value Difference')
    plt.colorbar(scatter2, ax=ax2, label='Value Difference')
    
    # Plot 3: Pattern distances
    ax3 = axes[0, 2]
    distances = [distinctness_results['mean_within_ss'],
                distinctness_results['mean_within_ll'],
                distinctness_results['mean_between']]
    distance_labels = ['Within SS', 'Within LL', 'Between SS-LL']
    colors = ['lightblue', 'lightcoral', 'gold']
    bars = ax3.bar(distance_labels, distances, color=colors)
    ax3.set_ylabel('Mean Distance')
    ax3.set_title(f'Pattern Distinctness\n(d = {distinctness_results["cohens_d"]:.2f})')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: RSM for SS condition
    ax4 = axes[1, 0]
    im4 = ax4.imshow(rsa_results['rsm_ss'], cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax4.set_title('RSM: Sooner-Smaller')
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Trial')
    plt.colorbar(im4, ax=ax4, label='Correlation')
    
    # Plot 5: RSM for LL condition
    ax5 = axes[1, 1]
    im5 = ax5.imshow(rsa_results['rsm_ll'], cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax5.set_title('RSM: Larger-Later')
    ax5.set_xlabel('Trial')
    ax5.set_ylabel('Trial')
    plt.colorbar(im5, ax=ax5, label='Correlation')
    
    # Plot 6: Decoding results
    ax6 = axes[1, 2]
    ax6.bar(['Cross-Validated\nAccuracy'], [decoding_results['mean_accuracy']], 
           yerr=[decoding_results['std_accuracy']], 
           color='green', alpha=0.7, capsize=5)
    ax6.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Decoding Performance')
    ax6.set_ylim([0, 1])
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('mvpa_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'mvpa_analysis_results.png'")

# =============================================================================
# STEP 7: PUTTING IT ALL TOGETHER
# =============================================================================

def main_analysis():
    """
    Main function that runs the complete MVPA analysis pipeline.
    """
    print("MVPA ANALYSIS TUTORIAL")
    print("=" * 50)
    print("This tutorial will walk you through a complete MVPA analysis")
    print("for delay discounting data, explaining each step in detail.\n")
    
    # Step 1: Load and understand data
    X, labels, value_diff = load_and_understand_data()
    
    # Step 2: Dimensionality reduction
    X_reduced, pca = apply_dimensionality_reduction(X)
    
    # Step 3: Pattern distinctness
    distinctness_results = analyze_pattern_distinctness(X_reduced, labels)
    
    # Step 4: RSA
    rsa_results = analyze_representational_similarity(X_reduced, labels)
    
    # Step 5: Cross-validated decoding
    decoding_results = perform_cross_validated_decoding(X_reduced, labels)
    
    # Step 6: Visualization
    create_visualizations(X_reduced, labels, value_diff, pca,
                         distinctness_results, rsa_results, decoding_results)
    
    # Step 7: Summary and interpretation
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    print(f"1. Pattern Distinctness: d = {distinctness_results['cohens_d']:.3f}")
    if distinctness_results['cohens_d'] > 0.5:
        print("   → Conditions have distinct neural representations")
    else:
        print("   → Conditions have similar neural representations")
    
    print(f"2. RSM Correlation: r = {rsa_results['rsm_correlation']:.3f}")
    if rsa_results['rsm_correlation'] > 0.3:
        print("   → Similar representational structure across conditions")
    else:
        print("   → Different representational structures")
    
    print(f"3. Decoding Accuracy: {decoding_results['mean_accuracy']:.3f} ± {decoding_results['std_accuracy']:.3f}")
    if decoding_results['mean_accuracy'] > 0.6:
        print("   → Good classification performance")
    else:
        print("   → Moderate classification performance")
    
    print(f"4. Variance Explained: {pca.explained_variance_ratio_[0]:.1%} (PC1)")
    print("   → First component captures main pattern differences")
    
    print("\n" + "=" * 50)
    print("WHAT THESE RESULTS MEAN")
    print("=" * 50)
    
    print("Your MVPA analysis reveals:")
    print("• How distinctly different choice types are represented")
    print("• Whether representational geometry is preserved across conditions")
    print("• How well neural patterns can predict behavior")
    print("• Which dimensions capture the most meaningful variance")
    
    print("\nNext steps for your research:")
    print("• Apply this pipeline to your real fMRI data")
    print("• Compare results across different brain regions")
    print("• Relate neural patterns to individual differences")
    print("• Test computational models against neural data")

if __name__ == "__main__":
    main_analysis() 