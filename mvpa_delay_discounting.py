#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from nilearn import datasets, image, masking, plotting
from nilearn.decoding import Decoder
from nilearn.glm.first_level import compute_regressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, KFold, cross_val_score
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE, MDS
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, pearsonr, ttest_ind, mannwhitneyu, spearmanr
from scipy.spatial.distance import pdist, squareform

def load_roi_masks():
    """
    Load ROI masks for striatum, DLPFC, and VMPFC.
    Returns dictionary of ROI masks.
    """
    # You'll need to replace these paths with actual paths to your ROI masks on OAK
    roi_paths = {
        'striatum': '/oak/stanford/path/to/striatum_mask.nii.gz',
        'dlpfc': '/oak/stanford/path/to/dlpfc_mask.nii.gz',
        'vmpfc': '/oak/stanford/path/to/vmpfc_mask.nii.gz'
    }
    
    roi_masks = {}
    for roi_name, roi_path in roi_paths.items():
        roi_masks[roi_name] = nib.load(roi_path)
    return roi_masks

def load_delay_discounting_data(subject_id):
    """
    Load preprocessed fMRI data and behavioral data for a given subject.
    Returns fMRI data and corresponding behavioral data.
    """
    # Replace with actual paths to your data on OAK
    fmri_path = f'/oak/stanford/path/to/sub-{subject_id}/func/task-delaydiscount_bold.nii.gz'
    behavior_path = f'/oak/stanford/path/to/sub-{subject_id}/behavior/task-delaydiscount_events.tsv'
    
    # Load fMRI data
    fmri_data = nib.load(fmri_path)
    
    # Load behavioral data
    behavior_data = pd.read_csv(behavior_path, sep='\t')
    
    return fmri_data, behavior_data

def extract_trial_data(fmri_data, behavior_data, roi_masks):
    """
    Extract trial-by-trial neural patterns from ROIs.
    Returns dictionary of ROI patterns and behavioral labels.
    """
    trial_data = {
        'neural_patterns': {},
        'choice_labels': behavior_data['choice'].values,  # 1 for larger-later, 0 for sooner-smaller
        'value_diff': behavior_data['chosen_value'] - behavior_data['unchosen_value']
    }
    
    for roi_name, roi_mask in roi_masks.items():
        # Extract ROI timeseries
        roi_timeseries = masking.apply_mask(fmri_data, roi_mask)
        
        # Convert to trial patterns (you'll need to implement proper trial averaging)
        trial_patterns = extract_trial_patterns(roi_timeseries, behavior_data)
        trial_data['neural_patterns'][roi_name] = trial_patterns
    
    return trial_data

def create_hrf_matrix(behavior_data, tr, n_scans):
    """
    Create HRF-convolved design matrix for trial onsets.
    
    Parameters:
    -----------
    behavior_data : pd.DataFrame
        DataFrame containing trial information including onset times
    tr : float
        TR (repetition time) of the fMRI data
    n_scans : int
        Number of scans in the fMRI run
    
    Returns:
    --------
    np.ndarray
        Design matrix with HRF-convolved trial responses
    """
    frame_times = np.arange(n_scans) * tr
    
    # Create regressor for each trial
    trial_regressors = []
    for trial_idx in range(len(behavior_data)):
        onset = behavior_data.iloc[trial_idx]['onset']
        duration = behavior_data.iloc[trial_idx]['duration']
        
        # Create regressor for this trial
        reg, _ = compute_regressor(
            exp_condition=([onset], [duration], [1]),
            hrf_model='spm',
            frame_times=frame_times
        )
        trial_regressors.append(reg.reshape(-1))
    
    return np.array(trial_regressors).T

def extract_trial_patterns(roi_timeseries, behavior_data, tr=2.0, window_size=3):
    """
    Extract trial-specific patterns from ROI timeseries.
    
    Parameters:
    -----------
    roi_timeseries : np.ndarray
        ROI timeseries data (time x voxels)
    behavior_data : pd.DataFrame
        DataFrame containing trial information
    tr : float
        TR (repetition time) of the fMRI data
    window_size : int
        Number of TRs to include after trial onset
    
    Returns:
    --------
    np.ndarray
        Trial-specific patterns (trials x voxels)
    """
    n_scans = roi_timeseries.shape[0]
    
    # Create HRF-convolved design matrix
    hrf_matrix = create_hrf_matrix(behavior_data, tr, n_scans)
    
    # Extract trial patterns
    trial_patterns = []
    for trial_idx in range(len(behavior_data)):
        onset_tr = int(behavior_data.iloc[trial_idx]['onset'] / tr)
        if onset_tr + window_size <= n_scans:
            # Average across time window
            trial_pattern = roi_timeseries[onset_tr:onset_tr + window_size].mean(axis=0)
            trial_patterns.append(trial_pattern)
    
    trial_patterns = np.array(trial_patterns)
    
    # Z-score patterns across trials
    scaler = StandardScaler()
    trial_patterns = scaler.fit_transform(trial_patterns)
    
    return trial_patterns

def run_permutation_test(X, y, decoder, n_permutations=1000):
    """
    Run permutation test for significance testing.
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns
    y : np.ndarray
        Labels/values to predict
    decoder : nilearn.decoding.Decoder
        Initialized decoder object
    n_permutations : int
        Number of permutations to run
    
    Returns:
    --------
    float
        P-value from permutation test
    """
    # Get true score
    true_score = decoder.fit(X, y).cv_scores_.mean()
    
    # Run permutations
    null_scores = []
    for _ in range(n_permutations):
        y_perm = shuffle(y)
        decoder.fit(X, y_perm)
        null_scores.append(decoder.cv_scores_.mean())
    
    # Calculate p-value
    null_scores = np.array(null_scores)
    p_value = (np.sum(null_scores >= true_score) + 1) / (n_permutations + 1)
    
    return true_score, p_value, null_scores

def visualize_embeddings(reduced_patterns, labels, roi_name, save_path):
    """
    Visualize low-dimensional embeddings colored by choice and value.
    
    Parameters:
    -----------
    reduced_patterns : np.ndarray
        PCA-reduced patterns
    labels : dict
        Dictionary containing choice labels and value differences
    roi_name : str
        Name of the ROI being visualized
    save_path : str
        Path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot colored by choice
    scatter1 = ax1.scatter(reduced_patterns[:, 0], reduced_patterns[:, 1],
                          c=labels['choice_labels'], cmap='coolwarm')
    ax1.set_title(f'{roi_name} - Choice Patterns')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1, label='Choice (SS vs LL)')
    
    # Plot colored by value difference
    scatter2 = ax2.scatter(reduced_patterns[:, 0], reduced_patterns[:, 1],
                          c=labels['value_diff'], cmap='RdYlBu')
    ax2.set_title(f'{roi_name} - Value Difference')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.colorbar(scatter2, ax=ax2, label='Value Difference')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{roi_name}_embeddings.png')
    plt.close()

def run_mvpa_analysis(trial_data, roi_name):
    """
    Run MVPA analysis for choice decoding and value difference encoding.
    Returns decoding accuracy and encoding model results.
    """
    X = trial_data['neural_patterns'][roi_name]
    y_choice = trial_data['choice_labels']
    y_value = trial_data['value_diff']
    
    # Dimensionality reduction
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    
    # Choice decoding with permutation test
    choice_decoder = Decoder(estimator='svc', cv=LeaveOneGroupOut())
    choice_score, choice_p, choice_null = run_permutation_test(X_reduced, y_choice, choice_decoder)
    
    # Value difference encoding with permutation test
    value_encoder = Decoder(estimator='ridge', cv=LeaveOneGroupOut())
    value_score, value_p, value_null = run_permutation_test(X_reduced, y_value, value_encoder)
    
    # Visualize embeddings
    os.makedirs('results/embeddings', exist_ok=True)
    visualize_embeddings(X_reduced, 
                        {'choice_labels': y_choice, 'value_diff': y_value},
                        roi_name,
                        'results/embeddings')
    
    # Compare geometries
    os.makedirs('results/geometry', exist_ok=True)
    geometry_results = compare_geometries(X_reduced, y_choice, y_value, 
                                        roi_name, 'results/geometry')
    
    return {
        'choice_accuracy': choice_score,
        'choice_p_value': choice_p,
        'choice_null_dist': choice_null,
        'value_correlation': value_score,
        'value_p_value': value_p,
        'value_null_dist': value_null,
        'reduced_patterns': X_reduced,
        'pca_components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'geometry_comparison': geometry_results
    }

def visualize_results(results):
    """
    Create summary visualizations of MVPA results.
    """
    os.makedirs('results/summary', exist_ok=True)
    
    # Plot choice decoding results
    plt.figure(figsize=(15, 5))
    for i, roi_name in enumerate(['striatum', 'dlpfc', 'vmpfc']):
        plt.subplot(1, 3, i+1)
        null_dist = results[roi_name]['choice_null_dist']
        plt.hist(null_dist, bins=30, density=True, alpha=0.5)
        plt.axvline(results[roi_name]['choice_accuracy'], color='r', linestyle='--')
        plt.title(f'{roi_name}\np={results[roi_name]["choice_p_value"]:.3f}')
        plt.xlabel('Accuracy')
    plt.suptitle('Choice Decoding Results')
    plt.tight_layout()
    plt.savefig('results/summary/choice_decoding.png')
    plt.close()
    
    # Plot value encoding results
    plt.figure(figsize=(15, 5))
    for i, roi_name in enumerate(['striatum', 'dlpfc', 'vmpfc']):
        plt.subplot(1, 3, i+1)
        null_dist = results[roi_name]['value_null_dist']
        plt.hist(null_dist, bins=30, density=True, alpha=0.5)
        plt.axvline(results[roi_name]['value_correlation'], color='r', linestyle='--')
        plt.title(f'{roi_name}\np={results[roi_name]["value_p_value"]:.3f}')
        plt.xlabel('Correlation')
    plt.suptitle('Value Encoding Results')
    plt.tight_layout()
    plt.savefig('results/summary/value_encoding.png')
    plt.close()

def compute_pattern_distinctness(X, labels):
    """
    Compute pattern distinctness between conditions using distance metrics.
    """
    # Split patterns by condition
    X_cond1 = X[labels == 0]  # sooner-smaller
    X_cond2 = X[labels == 1]  # larger-later
    
    # Compute within and between condition distances
    dist_within1 = pdist(X_cond1)
    dist_within2 = pdist(X_cond2)
    dist_between = euclidean_distances(X_cond1, X_cond2).flatten()
    
    # Statistical tests
    stat_mw, p_mw = mannwhitneyu(dist_between, 
                                np.concatenate([dist_within1, dist_within2]))
    
    # Effect size (Cohen's d)
    d = (np.mean(dist_between) - np.mean(np.concatenate([dist_within1, dist_within2]))) / \
        np.std(np.concatenate([dist_between, dist_within1, dist_within2]))
    
    return {
        'within_dist1_mean': np.mean(dist_within1),
        'within_dist2_mean': np.mean(dist_within2),
        'between_dist_mean': np.mean(dist_between),
        'distinctness_score': d,
        'mw_statistic': stat_mw,
        'mw_pvalue': p_mw
    }

def compute_rsa_comparison(X1, X2):
    """
    Compare representational similarity matrices between conditions.
    """
    # Compute RSMs
    rsm1 = squareform(pdist(X1, metric='correlation'))
    rsm2 = squareform(pdist(X2, metric='correlation'))
    
    # Compute correlation between RSMs
    rsm_corr, rsm_p = pearsonr(rsm1[np.triu_indices_from(rsm1, k=1)],
                              rsm2[np.triu_indices_from(rsm2, k=1)])
    
    return {
        'rsm_correlation': rsm_corr,
        'rsm_pvalue': rsm_p,
        'rsm1': rsm1,
        'rsm2': rsm2
    }

def compute_noise_ceiling(X1, X2, n_splits=5):
    """
    Compute the noise ceiling for geometry comparisons.
    
    Parameters:
    -----------
    X1, X2 : np.ndarray
        Neural patterns for two conditions
    n_splits : int
        Number of splits for cross-validation
    
    Returns:
    --------
    dict
        Noise ceiling estimates
    """
    # Split data into folds
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    # Compute within-condition reliability
    reliability_1 = []
    reliability_2 = []
    
    for train_idx, test_idx in kf.split(X1):
        # Within condition 1
        rsm1_train = squareform(pdist(X1[train_idx], metric='correlation'))
        rsm1_test = squareform(pdist(X1[test_idx], metric='correlation'))
        r1, _ = spearmanr(rsm1_train.flatten(), rsm1_test.flatten())
        reliability_1.append(r1)
        
        # Within condition 2
        rsm2_train = squareform(pdist(X2[train_idx], metric='correlation'))
        rsm2_test = squareform(pdist(X2[test_idx], metric='correlation'))
        r2, _ = spearmanr(rsm2_train.flatten(), rsm2_test.flatten())
        reliability_2.append(r2)
    
    return {
        'ceiling_cond1': np.mean(reliability_1),
        'ceiling_cond2': np.mean(reliability_2),
        'lower_bound': np.min([np.mean(reliability_1), np.mean(reliability_2)]),
        'upper_bound': np.sqrt(np.mean(reliability_1) * np.mean(reliability_2))
    }

def cross_validate_geometry(X, labels, value_diff, n_splits=5):
    """
    Cross-validate geometry preservation metrics.
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns
    labels : np.ndarray
        Condition labels
    value_diff : np.ndarray
        Value differences
    n_splits : int
        Number of splits for cross-validation
    
    Returns:
    --------
    dict
        Cross-validated metrics
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    cv_metrics = {
        'distinctness_scores': [],
        'rsm_correlations': [],
        'value_correlations': []
    }
    
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        labels_train, labels_test = labels[train_idx], labels[test_idx]
        value_diff_train, value_diff_test = value_diff[train_idx], value_diff[test_idx]
        
        # Compute metrics on test set
        distinctness = compute_pattern_distinctness(X_test, labels_test)
        cv_metrics['distinctness_scores'].append(distinctness['distinctness_score'])
        
        # RSM correlation
        X_ss = X_test[labels_test == 0]
        X_ll = X_test[labels_test == 1]
        rsa_results = compute_rsa_comparison(X_ss, X_ll)
        cv_metrics['rsm_correlations'].append(rsa_results['rsm_correlation'])
        
        # Value correlation
        value_corr, _ = pearsonr(X_test[:, 0], value_diff_test)  # Using first PC
        cv_metrics['value_correlations'].append(value_corr)
    
    return {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in cv_metrics.items()}

def check_robustness(X, labels, value_diff):
    """
    Check robustness of geometry comparisons across different dimensionality reductions.
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns
    labels : np.ndarray
        Condition labels
    value_diff : np.ndarray
        Value differences
    
    Returns:
    --------
    dict
        Robustness check results
    """
    methods = {
        'pca': PCA(n_components=10),
        'ica': FastICA(n_components=10, random_state=42),
        'tsne': TSNE(n_components=2, random_state=42),
        'mds': MDS(n_components=2, random_state=42)
    }
    
    results = {}
    for method_name, method in methods.items():
        # Reduce dimensionality
        X_reduced = method.fit_transform(X)
        
        # Compute metrics
        distinctness = compute_pattern_distinctness(X_reduced, labels)
        X_ss = X_reduced[labels == 0]
        X_ll = X_reduced[labels == 1]
        rsa_results = compute_rsa_comparison(X_ss, X_ll)
        
        results[method_name] = {
            'distinctness_score': distinctness['distinctness_score'],
            'rsm_correlation': rsa_results['rsm_correlation']
        }
    
    return results

def compare_geometries(reduced_patterns, labels, value_diff, roi_name, save_path):
    """
    Comprehensive comparison of geometries between conditions with statistical controls.
    """
    # 1. Basic Pattern Distinctness Analysis
    distinctness = compute_pattern_distinctness(reduced_patterns, labels)
    
    # 2. RSA Comparison with noise ceiling
    X_ss = reduced_patterns[labels == 0]
    X_ll = reduced_patterns[labels == 1]
    rsa_results = compute_rsa_comparison(X_ss, X_ll)
    noise_ceiling = compute_noise_ceiling(X_ss, X_ll)
    
    # 3. Cross-validated geometry metrics
    cv_results = cross_validate_geometry(reduced_patterns, labels, value_diff)
    
    # 4. Robustness checks
    robustness_results = check_robustness(reduced_patterns, labels, value_diff)
    
    # Visualize results
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Pattern distances with CV
    ax1 = plt.subplot(231)
    data = [distinctness['within_dist1_mean'], 
            distinctness['within_dist2_mean'],
            distinctness['between_dist_mean']]
    labels_plot = ['Within SS', 'Within LL', 'Between']
    sns.barplot(x=labels_plot, y=data, ax=ax1)
    ax1.set_title(f'Pattern Distances\nd={cv_results["distinctness_scores"]["mean"]:.2f}±{cv_results["distinctness_scores"]["std"]:.2f}')
    
    # Plot 2: RSM Correlation with noise ceiling
    ax2 = plt.subplot(232)
    sns.heatmap(rsa_results['rsm1'], ax=ax2)
    ax2.set_title(f'RSM Correlation\nr={rsa_results["rsm_correlation"]:.2f}\nceiling={noise_ceiling["upper_bound"]:.2f}')
    
    # Plot 3: Robustness across methods
    ax3 = plt.subplot(233)
    method_names = list(robustness_results.keys())
    distinctness_scores = [r['distinctness_score'] for r in robustness_results.values()]
    sns.barplot(x=method_names, y=distinctness_scores, ax=ax3)
    ax3.set_title('Robustness Across Methods')
    ax3.set_xticklabels(method_names, rotation=45)
    
    # Plot 4: Cross-validation results
    ax4 = plt.subplot(234)
    cv_means = [v['mean'] for v in cv_results.values()]
    cv_stds = [v['std'] for v in cv_results.values()]
    metric_names = list(cv_results.keys())
    sns.barplot(x=metric_names, y=cv_means, yerr=cv_stds, ax=ax4)
    ax4.set_title('Cross-validated Metrics')
    ax4.set_xticklabels(metric_names, rotation=45)
    
    # Plot 5: Value coding consistency
    ax5 = plt.subplot(235)
    for label, marker in zip([0, 1], ['o', 's']):
        mask = labels == label
        ax5.scatter(reduced_patterns[mask, 0], value_diff[mask], 
                   marker=marker, alpha=0.5,
                   label='SS' if label == 0 else 'LL')
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('Value Difference')
    ax5.set_title(f'Value Coding\nr={cv_results["value_correlations"]["mean"]:.2f}±{cv_results["value_correlations"]["std"]:.2f}')
    ax5.legend()
    
    # Plot 6: Noise ceiling comparison
    ax6 = plt.subplot(236)
    ceiling_data = [noise_ceiling['lower_bound'], 
                   rsa_results['rsm_correlation'],
                   noise_ceiling['upper_bound']]
    ceiling_labels = ['Lower\nBound', 'Observed\nCorrelation', 'Upper\nBound']
    sns.barplot(x=ceiling_labels, y=ceiling_data, ax=ax6)
    ax6.set_title('Noise Ceiling Analysis')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{roi_name}_geometry_comparison.png')
    plt.close()
    
    return {
        'distinctness': distinctness,
        'rsa': rsa_results,
        'noise_ceiling': noise_ceiling,
        'cv_results': cv_results,
        'robustness': robustness_results
    }

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load ROI masks
    roi_masks = load_roi_masks()
    
    # Process each subject
    results = {}
    for subject_id in range(1, 104):  # Adjust range based on your subject IDs
        print(f"Processing subject {subject_id}")
        
        # Load subject data
        fmri_data, behavior_data = load_delay_discounting_data(subject_id)
        
        # Extract trial data
        trial_data = extract_trial_data(fmri_data, behavior_data, roi_masks)
        
        # Run MVPA for each ROI
        subject_results = {}
        for roi_name in roi_masks.keys():
            subject_results[roi_name] = run_mvpa_analysis(trial_data, roi_name)
        
        results[subject_id] = subject_results
        
        # Visualize results for this subject
        visualize_results(subject_results)
    
    # Save results
    np.save('results/mvpa_results.npy', results)

if __name__ == "__main__":
    main() 