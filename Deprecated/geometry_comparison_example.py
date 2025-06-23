#!/usr/bin/env python3
"""
Example script demonstrating formal geometry comparison methods for neural manifolds

This script shows how to compare neural geometries between different conditions
using statistical methods including RSA, Procrustes analysis, permutation tests,
and dimensionality analysis.

Author: Joshua Buckholtz Lab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import sys
from pathlib import Path

# Add the main pipeline to path so we can import the GeometryAnalysis class
sys.path.append('.')
from delay_discounting_mvpa_pipeline import GeometryAnalysis, Config

def generate_synthetic_neural_data(n_trials=200, n_voxels=500, condition_separation=2.0):
    """
    Generate synthetic neural data with two conditions that have different geometries
    
    Parameters:
    -----------
    n_trials : int
        Number of trials per condition
    n_voxels : int
        Number of voxels/features
    condition_separation : float
        How separated the conditions should be in neural space
        
    Returns:
    --------
    dict : Contains neural data and condition labels
    """
    
    # Create two conditions with different geometric properties
    np.random.seed(42)
    
    # Condition 1: More compact, lower dimensional structure
    n_latent_1 = 3
    latent_1 = np.random.randn(n_trials, n_latent_1)
    mixing_matrix_1 = np.random.randn(n_latent_1, n_voxels) * 0.5
    condition_1 = latent_1 @ mixing_matrix_1 + np.random.randn(n_trials, n_voxels) * 0.1
    
    # Condition 2: More spread out, higher dimensional structure
    n_latent_2 = 8
    latent_2 = np.random.randn(n_trials, n_latent_2) * 1.5
    mixing_matrix_2 = np.random.randn(n_latent_2, n_voxels) * 0.3
    condition_2 = latent_2 @ mixing_matrix_2 + np.random.randn(n_trials, n_voxels) * 0.2
    
    # Add systematic separation between conditions
    condition_2 += condition_separation
    
    # Combine data
    neural_data = np.vstack([condition_1, condition_2])
    condition_labels = np.hstack([np.zeros(n_trials), np.ones(n_trials)])
    
    return {
        'neural_data': neural_data,
        'condition_labels': condition_labels,
        'condition_names': ['Compact Condition', 'Spread Condition']
    }

def run_geometry_comparison_example():
    """
    Complete example of neural geometry comparison analysis
    """
    
    print("Neural Geometry Comparison Example")
    print("=" * 50)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic neural data...")
    data = generate_synthetic_neural_data(n_trials=150, n_voxels=300, condition_separation=1.5)
    
    neural_data = data['neural_data']
    condition_labels = data['condition_labels'].astype(int)
    condition_names = data['condition_names']
    
    print(f"   Generated {neural_data.shape[0]} trials with {neural_data.shape[1]} voxels")
    print(f"   Condition 1: {np.sum(condition_labels == 0)} trials")
    print(f"   Condition 2: {np.sum(condition_labels == 1)} trials")
    
    # 2. Initialize geometry analysis
    print("\n2. Initializing geometry analysis...")
    config = Config()
    geometry_analysis = GeometryAnalysis(config)
    
    # 3. Dimensionality reduction
    print("\n3. Performing dimensionality reduction...")
    embedding_pca, reducer_pca = geometry_analysis.dimensionality_reduction(
        neural_data, method='pca', n_components=10
    )
    
    embedding_mds, reducer_mds = geometry_analysis.dimensionality_reduction(
        neural_data, method='mds', n_components=5
    )
    
    print(f"   PCA embedding shape: {embedding_pca.shape}")
    print(f"   MDS embedding shape: {embedding_mds.shape}")
    
    # 4. Formal geometry comparison
    print("\n4. Running formal geometry comparison...")
    
    # Compare PCA embeddings
    print("\n   PCA Embedding Comparison:")
    pca_comparison = geometry_analysis.compare_embeddings_by_condition(
        embedding=embedding_pca,
        condition_labels=condition_labels,
        condition_names=condition_names,
        n_permutations=1000
    )
    
    # Compare MDS embeddings
    print("\n   MDS Embedding Comparison:")
    mds_comparison = geometry_analysis.compare_embeddings_by_condition(
        embedding=embedding_mds,
        condition_labels=condition_labels,
        condition_names=condition_names,
        n_permutations=1000
    )
    
    # 5. Display results
    print("\n5. Results Summary")
    print("=" * 30)
    
    def print_comparison_results(comparison_results, method_name):
        """Helper function to print comparison results"""
        print(f"\n{method_name} Results:")
        print("-" * (len(method_name) + 9))
        
        # RSA results
        if 'rsa' in comparison_results:
            rsa = comparison_results['rsa']
            print(f"  Representational Similarity:")
            print(f"    Correlation: {rsa['correlation']:.3f}")
            print(f"    P-value: {rsa['p_value']:.3f}")
            print(f"    Interpretation: {rsa['interpretation']}")
        
        # Procrustes results
        if 'procrustes' in comparison_results:
            proc = comparison_results['procrustes']
            print(f"  Procrustes Analysis:")
            print(f"    Disparity: {proc['disparity']:.3f}")
            print(f"    Interpretation: {proc['interpretation']}")
        
        # Distance analysis
        if 'distance_analysis' in comparison_results:
            dist = comparison_results['distance_analysis']
            print(f"  Distance Analysis:")
            print(f"    Mean within-condition distance: {dist['mean_within_condition_distance']:.3f}")
            print(f"    Mean between-condition distance: {dist['mean_between_condition_distance']:.3f}")
            print(f"    Separation ratio: {dist['separation_ratio']:.3f}")
            print(f"    Separation p-value: {dist['separation_p_value']:.3f}")
            print(f"    Interpretation: {dist['interpretation']}")
        
        # Statistical tests
        if 'statistical_tests' in comparison_results:
            print(f"  Permutation Tests:")
            for prop, test_result in comparison_results['statistical_tests'].items():
                significance = "***" if test_result['p_value'] < 0.001 else \
                              "**" if test_result['p_value'] < 0.01 else \
                              "*" if test_result['p_value'] < 0.05 else "ns"
                print(f"    {prop.replace('_', ' ').title()}: "
                      f"diff={test_result['observed_difference']:.3f}, "
                      f"p={test_result['p_value']:.3f} {significance}")
        
        # Dimensionality analysis
        if 'dimensionality_analysis' in comparison_results:
            print(f"  Dimensionality Analysis:")
            for cond_name, dim_results in comparison_results['dimensionality_analysis'].items():
                print(f"    {cond_name}:")
                print(f"      Participation ratio: {dim_results['participation_ratio']:.3f}")
                print(f"      Intrinsic dim (80%): {dim_results['intrinsic_dimensionality_80']}")
                print(f"      Intrinsic dim (90%): {dim_results['intrinsic_dimensionality_90']}")
    
    print_comparison_results(pca_comparison, "PCA")
    print_comparison_results(mds_comparison, "MDS")
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    
    # Create output directory
    output_dir = Path("./example_geometry_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Plot PCA comparison
    pca_plots = geometry_analysis.plot_geometry_comparison(
        comparison_results=pca_comparison,
        embedding=embedding_pca,
        condition_labels=condition_labels,
        roi_name="PCA_Example",
        output_dir=output_dir
    )
    
    # Plot MDS comparison  
    mds_plots = geometry_analysis.plot_geometry_comparison(
        comparison_results=mds_comparison,
        embedding=embedding_mds,
        condition_labels=condition_labels,
        roi_name="MDS_Example",
        output_dir=output_dir
    )
    
    print(f"   Visualizations saved to {output_dir}")
    print(f"   PCA plots: {list(pca_plots.keys())}")
    print(f"   MDS plots: {list(mds_plots.keys())}")
    
    # 7. Advanced comparison: Testing different separation levels
    print("\n7. Testing different condition separation levels...")
    
    separation_levels = [0.5, 1.0, 1.5, 2.0, 2.5]
    separation_results = []
    
    for sep_level in separation_levels:
        test_data = generate_synthetic_neural_data(
            n_trials=100, n_voxels=200, condition_separation=sep_level
        )
        
        test_embedding, _ = geometry_analysis.dimensionality_reduction(
            test_data['neural_data'], method='pca', n_components=5
        )
        
        test_comparison = geometry_analysis.compare_embeddings_by_condition(
            embedding=test_embedding,
            condition_labels=test_data['condition_labels'].astype(int),
            condition_names=test_data['condition_names'],
            n_permutations=500
        )
        
        # Extract key metrics
        sep_ratio = test_comparison['distance_analysis']['separation_ratio']
        rsa_corr = test_comparison['rsa']['correlation']
        
        separation_results.append({
            'separation_level': sep_level,
            'separation_ratio': sep_ratio,
            'rsa_correlation': rsa_corr
        })
        
        print(f"   Separation {sep_level:.1f}: ratio={sep_ratio:.2f}, RSA_corr={rsa_corr:.3f}")
    
    # Plot separation analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sep_levels = [r['separation_level'] for r in separation_results]
    sep_ratios = [r['separation_ratio'] for r in separation_results]
    rsa_corrs = [r['rsa_correlation'] for r in separation_results]
    
    axes[0].plot(sep_levels, sep_ratios, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Condition Separation Level')
    axes[0].set_ylabel('Geometric Separation Ratio')
    axes[0].set_title('Geometry Separation vs Data Separation')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sep_levels, rsa_corrs, 's-', color='red', linewidth=2, markersize=8)
    axes[1].set_xlabel('Condition Separation Level')
    axes[1].set_ylabel('RSA Correlation')
    axes[1].set_title('Representational Similarity vs Data Separation')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'separation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Separation analysis plot saved to {output_dir}/separation_analysis.png")
    
    print("\n" + "=" * 50)
    print("GEOMETRY COMPARISON EXAMPLE COMPLETE!")
    print("=" * 50)
    
    return {
        'pca_comparison': pca_comparison,
        'mds_comparison': mds_comparison,
        'separation_analysis': separation_results,
        'visualization_files': {**pca_plots, **mds_plots}
    }

def main():
    """Run the complete geometry comparison example"""
    try:
        results = run_geometry_comparison_example()
        
        print("\nKey Takeaways:")
        print("- RSA correlation tells you how similar the representational geometries are")
        print("- Procrustes disparity measures shape differences after optimal alignment")
        print("- Separation ratio quantifies how well-separated conditions are in neural space")
        print("- Permutation tests provide statistical significance for geometric differences")
        print("- Dimensionality analysis reveals intrinsic complexity of each condition")
        print("\nFor your delay discounting data, use these methods to compare:")
        print("- Smaller-sooner vs larger-later choice trials")
        print("- High vs low subjective value trials")
        print("- High vs low unchosen option value trials")
        print("- Shorter vs longer delay trials (median split)")
        print("- Immediate (0-day) vs delayed trials")
        print("- Short delays (≤7 days) vs long delays (>30 days)")
        print("- High vs low discount rate subjects")
        print("- Different brain regions (striatum vs DLPFC vs VMPFC)")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 