#!/usr/bin/env python3
"""
MVPA Utilities Module for Delay Discounting Analysis
===================================================

This module provides centralized functions for all MVPA (Multi-Voxel Pattern Analysis)
procedures used in the delay discounting fMRI analysis pipeline. It eliminates code
duplication and provides consistent, well-tested implementations of common ML operations.

Key Functions:
- run_classification(): Binary and multi-class classification with multiple algorithms
- run_regression(): Continuous variable regression with multiple algorithms  
- run_cross_validation(): Standardized cross-validation procedures
- run_permutation_test(): Statistical significance testing
- extract_neural_patterns(): Advanced pattern extraction methods
- run_dimensionality_reduction(): PCA, MDS, t-SNE, Isomap with consistent interfaces
- run_searchlight_analysis(): Whole-brain searchlight procedures
- compute_feature_importance(): Feature selection and importance scoring

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Machine learning
from sklearn.model_selection import (
    StratifiedKFold, KFold, LeaveOneOut, GroupKFold,
    permutation_test_score, cross_val_score, cross_validate
)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline

# Neuroimaging
import nibabel as nib
from nilearn import image, masking
from nilearn.input_data import NiftiMasker
from nilearn.decoding import SearchLight

# Parallel processing
from joblib import Parallel, delayed

# Import data utilities for integration
try:
    from data_utils import DataError, load_mask
except ImportError:
    warnings.warn("Could not import data_utils. Some functionality may be limited.")
    DataError = Exception
    load_mask = None


class MVPAError(Exception):
    """Custom exception for MVPA-related errors"""
    pass


class MVPAConfig:
    """Configuration class for MVPA parameters"""
    
    # Cross-validation settings
    CV_FOLDS = 5
    CV_SHUFFLE = True
    CV_RANDOM_STATE = 42
    
    # Permutation testing
    N_PERMUTATIONS = 1000
    PERM_RANDOM_STATE = 42
    
    # Algorithm defaults
    DEFAULT_CLASSIFIER = 'svm'
    DEFAULT_REGRESSOR = 'ridge'
    
    # SVM parameters
    SVM_C = 1.0
    SVM_KERNEL = 'linear'
    
    # Ridge regression parameters
    RIDGE_ALPHA = 1.0
    
    # Feature selection
    FEATURE_SELECTION = False
    N_FEATURES = 1000
    
    # Preprocessing
    STANDARDIZE = True
    
    # Parallel processing
    N_JOBS = 1  # Conservative default for memory management
    
    # Quality control
    MIN_SAMPLES_PER_CLASS = 5
    MIN_VARIANCE_THRESHOLD = 1e-8


def validate_input_data(X: np.ndarray, y: np.ndarray, 
                       task_type: str = 'classification') -> None:
    """
    Validate input data for MVPA analysis
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    task_type : str
        Type of task ('classification' or 'regression')
    
    Raises:
    -------
    MVPAError
        If data validation fails
    """
    if X.shape[0] != len(y):
        raise MVPAError(f"Mismatch in number of samples: X has {X.shape[0]}, y has {len(y)}")
    
    if X.shape[0] < MVPAConfig.CV_FOLDS:
        raise MVPAError(f"Not enough samples for {MVPAConfig.CV_FOLDS}-fold CV: {X.shape[0]}")
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise MVPAError("X contains NaN or infinite values")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise MVPAError("y contains NaN or infinite values")
    
    if X.var() < MVPAConfig.MIN_VARIANCE_THRESHOLD:
        raise MVPAError("X has very low variance, check data preprocessing")
    
    if task_type == 'classification':
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise MVPAError("Classification requires at least 2 classes")
        
        for class_label in unique_classes:
            n_samples = np.sum(y == class_label)
            if n_samples < MVPAConfig.MIN_SAMPLES_PER_CLASS:
                raise MVPAError(f"Class {class_label} has only {n_samples} samples, "
                              f"minimum {MVPAConfig.MIN_SAMPLES_PER_CLASS} required")


def setup_classifier(algorithm: str = 'svm', **kwargs) -> Any:
    """
    Setup classifier with specified algorithm and parameters
    
    Parameters:
    -----------
    algorithm : str
        Classifier type ('svm', 'logistic', 'rf')
    **kwargs : dict
        Algorithm-specific parameters
    
    Returns:
    --------
    classifier : sklearn estimator
        Configured classifier
    """
    if algorithm == 'svm':
        return SVC(
            kernel=kwargs.get('kernel', MVPAConfig.SVM_KERNEL),
            C=kwargs.get('C', MVPAConfig.SVM_C),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
            probability=kwargs.get('probability', False)
        )
    elif algorithm == 'logistic':
        return LogisticRegression(
            C=kwargs.get('C', 1.0),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
            max_iter=kwargs.get('max_iter', 1000),
            solver=kwargs.get('solver', 'liblinear')
        )
    elif algorithm == 'rf':
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
            n_jobs=kwargs.get('n_jobs', 1)
        )
    else:
        raise MVPAError(f"Unknown classifier algorithm: {algorithm}")


def setup_regressor(algorithm: str = 'ridge', **kwargs) -> Any:
    """
    Setup regressor with specified algorithm and parameters
    
    Parameters:
    -----------
    algorithm : str
        Regressor type ('ridge', 'lasso', 'elastic', 'svr', 'rf')
    **kwargs : dict
        Algorithm-specific parameters
    
    Returns:
    --------
    regressor : sklearn estimator
        Configured regressor
    """
    if algorithm == 'ridge':
        return Ridge(
            alpha=kwargs.get('alpha', MVPAConfig.RIDGE_ALPHA),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE)
        )
    elif algorithm == 'lasso':
        return Lasso(
            alpha=kwargs.get('alpha', 1.0),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
            max_iter=kwargs.get('max_iter', 1000)
        )
    elif algorithm == 'elastic':
        return ElasticNet(
            alpha=kwargs.get('alpha', 1.0),
            l1_ratio=kwargs.get('l1_ratio', 0.5),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
            max_iter=kwargs.get('max_iter', 1000)
        )
    elif algorithm == 'svr':
        return SVR(
            kernel=kwargs.get('kernel', 'linear'),
            C=kwargs.get('C', 1.0),
            epsilon=kwargs.get('epsilon', 0.1)
        )
    elif algorithm == 'rf':
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
            n_jobs=kwargs.get('n_jobs', 1)
        )
    else:
        raise MVPAError(f"Unknown regressor algorithm: {algorithm}")


def setup_cross_validation(cv_type: str = 'stratified', 
                          n_splits: int = None,
                          groups: np.ndarray = None,
                          **kwargs) -> Any:
    """
    Setup cross-validation strategy
    
    Parameters:
    -----------
    cv_type : str
        CV type ('stratified', 'kfold', 'group', 'loo')
    n_splits : int
        Number of CV folds (default from config)
    groups : np.ndarray
        Group labels for GroupKFold
    **kwargs : dict
        Additional CV parameters
    
    Returns:
    --------
    cv : sklearn cross-validator
        Configured cross-validator
    """
    n_splits = n_splits or MVPAConfig.CV_FOLDS
    
    if cv_type == 'stratified':
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=kwargs.get('shuffle', MVPAConfig.CV_SHUFFLE),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE)
        )
    elif cv_type == 'kfold':
        return KFold(
            n_splits=n_splits,
            shuffle=kwargs.get('shuffle', MVPAConfig.CV_SHUFFLE),
            random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE)
        )
    elif cv_type == 'group':
        if groups is None:
            raise MVPAError("GroupKFold requires groups parameter")
        return GroupKFold(n_splits=n_splits)
    elif cv_type == 'loo':
        return LeaveOneOut()
    else:
        raise MVPAError(f"Unknown CV type: {cv_type}")


def run_classification(X: np.ndarray, y: np.ndarray,
                      algorithm: str = None,
                      cv_strategy: str = 'stratified',
                      n_permutations: int = None,
                      feature_selection: bool = None,
                      return_predictions: bool = False,
                      groups: np.ndarray = None,
                      roi_name: str = 'unknown',
                      **kwargs) -> Dict[str, Any]:
    """
    Run classification analysis with cross-validation and permutation testing
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Class labels (n_samples,)
    algorithm : str
        Classifier algorithm ('svm', 'logistic', 'rf')
    cv_strategy : str
        Cross-validation strategy
    n_permutations : int
        Number of permutation tests
    feature_selection : bool
        Whether to perform feature selection
    return_predictions : bool
        Whether to return cross-validation predictions
    groups : np.ndarray
        Group labels for GroupKFold
    roi_name : str
        Name of ROI for reporting
    **kwargs : dict
        Additional parameters
    
    Returns:
    --------
    dict : Classification results
    """
    # Set defaults from config
    algorithm = algorithm or MVPAConfig.DEFAULT_CLASSIFIER
    n_permutations = n_permutations or MVPAConfig.N_PERMUTATIONS
    feature_selection = feature_selection if feature_selection is not None else MVPAConfig.FEATURE_SELECTION
    
    try:
        # Validate input data
        validate_input_data(X, y, 'classification')
        
        # Setup preprocessing pipeline
        steps = []
        if MVPAConfig.STANDARDIZE:
            steps.append(('scaler', StandardScaler()))
        
        if feature_selection:
            steps.append(('selector', SelectKBest(
                f_classif, 
                k=min(MVPAConfig.N_FEATURES, X.shape[1])
            )))
        
        # Setup classifier
        classifier = setup_classifier(algorithm, **kwargs)
        steps.append(('classifier', classifier))
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        # Setup cross-validation
        cv = setup_cross_validation(cv_strategy, groups=groups, **kwargs)
        
        # Perform cross-validation
        if return_predictions:
            cv_results = cross_validate(
                pipeline, X, y, cv=cv,
                scoring=['accuracy', 'roc_auc'] if len(np.unique(y)) == 2 else ['accuracy'],
                return_train_score=True,
                n_jobs=MVPAConfig.N_JOBS
            )
            scores = cv_results['test_accuracy']
        else:
            scores = cross_val_score(
                pipeline, X, y, cv=cv,
                scoring='accuracy',
                n_jobs=MVPAConfig.N_JOBS
            )
        
        # Permutation test for statistical significance
        score_perm, perm_scores, p_value = permutation_test_score(
            pipeline, X, y, cv=cv,
            n_permutations=n_permutations,
            scoring='accuracy',
            n_jobs=MVPAConfig.N_JOBS,
            random_state=MVPAConfig.PERM_RANDOM_STATE
        )
        
        # Compile results
        results = {
            'success': True,
            'roi': roi_name,
            'algorithm': algorithm,
            'cv_strategy': cv_strategy,
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'scores': scores,
            'permutation_accuracy': score_perm,
            'permutation_scores': perm_scores,
            'p_value': p_value,
            'chance_level': 1.0 / len(np.unique(y)),
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
        
        # Add AUC for binary classification
        if len(np.unique(y)) == 2 and return_predictions:
            try:
                auc_scores = cv_results['test_roc_auc']
                results.update({
                    'mean_auc': np.mean(auc_scores),
                    'std_auc': np.std(auc_scores),
                    'auc_scores': auc_scores
                })
            except KeyError:
                pass
        
        # Add feature importance if available
        if hasattr(classifier, 'feature_importances_'):
            pipeline.fit(X, y)
            if feature_selection:
                # Get selected features
                selector = pipeline.named_steps['selector']
                selected_features = selector.get_support()
                feature_importance = np.zeros(X.shape[1])
                feature_importance[selected_features] = classifier.feature_importances_
            else:
                feature_importance = pipeline.named_steps['classifier'].feature_importances_
            
            results['feature_importance'] = feature_importance
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'roi': roi_name,
            'error': str(e),
            'algorithm': algorithm
        }


def run_regression(X: np.ndarray, y: np.ndarray,
                  algorithm: str = None,
                  cv_strategy: str = 'kfold',
                  n_permutations: int = None,
                  feature_selection: bool = None,
                  return_predictions: bool = False,
                  groups: np.ndarray = None,
                  roi_name: str = 'unknown',
                  variable_name: str = 'unknown',
                  **kwargs) -> Dict[str, Any]:
    """
    Run regression analysis with cross-validation and permutation testing
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Continuous target values (n_samples,)
    algorithm : str
        Regressor algorithm ('ridge', 'lasso', 'elastic', 'svr', 'rf')
    cv_strategy : str
        Cross-validation strategy
    n_permutations : int
        Number of permutation tests
    feature_selection : bool
        Whether to perform feature selection
    return_predictions : bool
        Whether to return cross-validation predictions
    groups : np.ndarray
        Group labels for GroupKFold
    roi_name : str
        Name of ROI for reporting
    variable_name : str
        Name of target variable
    **kwargs : dict
        Additional parameters
    
    Returns:
    --------
    dict : Regression results
    """
    # Set defaults from config
    algorithm = algorithm or MVPAConfig.DEFAULT_REGRESSOR
    n_permutations = n_permutations or MVPAConfig.N_PERMUTATIONS
    feature_selection = feature_selection if feature_selection is not None else MVPAConfig.FEATURE_SELECTION
    
    try:
        # Validate input data
        validate_input_data(X, y, 'regression')
        
        # Setup preprocessing pipeline
        steps = []
        if MVPAConfig.STANDARDIZE:
            steps.append(('scaler', StandardScaler()))
        
        if feature_selection:
            steps.append(('selector', SelectKBest(
                f_regression,
                k=min(MVPAConfig.N_FEATURES, X.shape[1])
            )))
        
        # Setup regressor
        regressor = setup_regressor(algorithm, **kwargs)
        steps.append(('regressor', regressor))
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        # Setup cross-validation
        cv = setup_cross_validation(cv_strategy, groups=groups, **kwargs)
        
        # Perform cross-validation
        if return_predictions:
            cv_results = cross_validate(
                pipeline, X, y, cv=cv,
                scoring=['r2', 'neg_mean_squared_error'],
                return_train_score=True,
                n_jobs=MVPAConfig.N_JOBS
            )
            r2_scores = cv_results['test_r2']
            mse_scores = -cv_results['test_neg_mean_squared_error']
        else:
            r2_scores = cross_val_score(
                pipeline, X, y, cv=cv,
                scoring='r2',
                n_jobs=MVPAConfig.N_JOBS
            )
            mse_scores = -cross_val_score(
                pipeline, X, y, cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=MVPAConfig.N_JOBS
            )
        
        # Permutation test for statistical significance
        score_perm, perm_scores, p_value = permutation_test_score(
            pipeline, X, y, cv=cv,
            n_permutations=n_permutations,
            scoring='r2',
            n_jobs=MVPAConfig.N_JOBS,
            random_state=MVPAConfig.PERM_RANDOM_STATE
        )
        
        # Compile results
        results = {
            'success': True,
            'roi': roi_name,
            'variable': variable_name,
            'algorithm': algorithm,
            'cv_strategy': cv_strategy,
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'r2_scores': r2_scores,
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'mse_scores': mse_scores,
            'permutation_r2': score_perm,
            'permutation_scores': perm_scores,
            'p_value': p_value,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'target_variance': np.var(y)
        }
        
        # Add feature importance if available
        if hasattr(regressor, 'feature_importances_'):
            pipeline.fit(X, y)
            if feature_selection:
                # Get selected features
                selector = pipeline.named_steps['selector']
                selected_features = selector.get_support()
                feature_importance = np.zeros(X.shape[1])
                feature_importance[selected_features] = regressor.feature_importances_
            else:
                feature_importance = pipeline.named_steps['regressor'].feature_importances_
            
            results['feature_importance'] = feature_importance
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'roi': roi_name,
            'variable': variable_name,
            'error': str(e),
            'algorithm': algorithm
        }


def extract_neural_patterns(img: Union[str, nib.Nifti1Image],
                           events_df: pd.DataFrame,
                           masker: NiftiMasker,
                           confounds: Optional[np.ndarray] = None,
                           pattern_type: str = 'single_timepoint',
                           tr: float = 0.68,
                           hemi_lag: int = 0,
                           window_size: int = 3) -> Dict[str, Any]:
    """
    Extract trial-wise neural patterns with multiple extraction methods
    
    Parameters:
    -----------
    img : str or nibabel image
        fMRI data
    events_df : pd.DataFrame
        Trial events with onsets
    masker : NiftiMasker
        Fitted masker object
    confounds : np.ndarray, optional
        Confound regressors
    pattern_type : str
        Extraction method:
        - 'single_timepoint': Single TR at peak
        - 'average_window': Average over time window
        - 'temporal_profile': Full temporal profile
        - 'peak_detection': Automatic peak detection
    tr : float
        Repetition time in seconds
    hemi_lag : int
        Hemodynamic lag in TRs
    window_size : int
        Size of time window for averaging
    
    Returns:
    --------
    dict : Extracted patterns and metadata
    """
    try:
        # Load and mask the data
        if confounds is not None:
            X = masker.fit_transform(img, confounds=confounds)
        else:
            X = masker.fit_transform(img)
        
        n_trials = len(events_df)
        n_voxels = X.shape[1]
        
        if pattern_type == 'single_timepoint':
            # Extract single timepoint at peak response
            patterns = np.zeros((n_trials, n_voxels))
            
            for i, (_, trial) in enumerate(events_df.iterrows()):
                onset_tr = int(trial['onset'] / tr)
                trial_tr = onset_tr + hemi_lag
                
                if trial_tr < X.shape[0]:
                    patterns[i, :] = X[trial_tr, :]
                else:
                    patterns[i, :] = X[-1, :]
        
        elif pattern_type == 'average_window':
            # Average over time window
            patterns = np.zeros((n_trials, n_voxels))
            
            for i, (_, trial) in enumerate(events_df.iterrows()):
                onset_tr = int(trial['onset'] / tr)
                start_tr = onset_tr + hemi_lag
                end_tr = start_tr + window_size
                
                if end_tr <= X.shape[0]:
                    patterns[i, :] = np.mean(X[start_tr:end_tr, :], axis=0)
                else:
                    patterns[i, :] = np.mean(X[start_tr:, :], axis=0)
        
        elif pattern_type == 'temporal_profile':
            # Extract full temporal profile around each trial
            profile_length = window_size * 2 + 1
            patterns = np.zeros((n_trials, n_voxels * profile_length))
            
            for i, (_, trial) in enumerate(events_df.iterrows()):
                onset_tr = int(trial['onset'] / tr)
                center_tr = onset_tr + hemi_lag
                start_tr = center_tr - window_size
                end_tr = center_tr + window_size + 1
                
                if start_tr >= 0 and end_tr <= X.shape[0]:
                    profile = X[start_tr:end_tr, :].flatten()
                    patterns[i, :] = profile
                else:
                    # Handle edge cases
                    patterns[i, :] = np.tile(X[center_tr, :], profile_length)
        
        elif pattern_type == 'peak_detection':
            # Automatically detect peak response
            patterns = np.zeros((n_trials, n_voxels))
            
            for i, (_, trial) in enumerate(events_df.iterrows()):
                onset_tr = int(trial['onset'] / tr)
                search_start = onset_tr
                search_end = min(onset_tr + window_size + hemi_lag, X.shape[0])
                
                # Find peak as maximum mean activity
                search_window = X[search_start:search_end, :]
                mean_activity = np.mean(search_window, axis=1)
                peak_idx = np.argmax(mean_activity)
                
                patterns[i, :] = search_window[peak_idx, :]
        
        else:
            raise MVPAError(f"Unknown pattern extraction type: {pattern_type}")
        
        return {
            'success': True,
            'patterns': patterns,
            'pattern_type': pattern_type,
            'n_trials': n_trials,
            'n_voxels': n_voxels,
            'window_size': window_size,
            'hemi_lag': hemi_lag
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'pattern_type': pattern_type
        }


def run_dimensionality_reduction(X: np.ndarray,
                               method: str = 'pca',
                               n_components: int = 10,
                               **kwargs) -> Dict[str, Any]:
    """
    Perform dimensionality reduction with multiple algorithms
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    method : str
        Reduction method ('pca', 'mds', 'tsne', 'isomap')
    n_components : int
        Number of components
    **kwargs : dict
        Method-specific parameters
    
    Returns:
    --------
    dict : Dimensionality reduction results
    """
    try:
        validate_input_data(X, np.zeros(X.shape[0]), 'regression')
        
        # Standardize data
        if kwargs.get('standardize', True):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None
        
        # Setup reducer
        if method == 'pca':
            reducer = PCA(
                n_components=n_components,
                random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE)
            )
        elif method == 'mds':
            reducer = MDS(
                n_components=n_components,
                random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
                dissimilarity=kwargs.get('dissimilarity', 'euclidean'),
                n_jobs=kwargs.get('n_jobs', MVPAConfig.N_JOBS)
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=min(n_components, 3),  # t-SNE typically limited to 2-3D
                random_state=kwargs.get('random_state', MVPAConfig.CV_RANDOM_STATE),
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000)
            )
        elif method == 'isomap':
            reducer = Isomap(
                n_components=n_components,
                n_neighbors=kwargs.get('n_neighbors', 5)
            )
        else:
            raise MVPAError(f"Unknown dimensionality reduction method: {method}")
        
        # Fit and transform
        embedding = reducer.fit_transform(X_scaled)
        
        # Calculate explained variance for methods that support it
        explained_variance = None
        if hasattr(reducer, 'explained_variance_ratio_'):
            explained_variance = reducer.explained_variance_ratio_
        elif hasattr(reducer, 'stress_'):
            explained_variance = [1.0 - reducer.stress_]  # Convert stress to explained variance
        
        return {
            'success': True,
            'method': method,
            'embedding': embedding,
            'reducer': reducer,
            'scaler': scaler,
            'n_components': n_components,
            'original_shape': X.shape,
            'embedding_shape': embedding.shape,
            'explained_variance': explained_variance
        }
        
    except Exception as e:
        return {
            'success': False,
            'method': method,
            'error': str(e)
        }


def run_searchlight_analysis(img: Union[str, nib.Nifti1Image],
                           y: np.ndarray,
                           mask_img: Union[str, nib.Nifti1Image] = None,
                           radius: float = 3.0,
                           estimator: str = 'svm',
                           cv_strategy: str = 'stratified',
                           n_jobs: int = 1,
                           **kwargs) -> Dict[str, Any]:
    """
    Run searchlight analysis across the brain
    
    Parameters:
    -----------
    img : str or nibabel image
        fMRI data
    y : np.ndarray
        Target values
    mask_img : str or nibabel image, optional
        Brain mask
    radius : float
        Searchlight radius in mm
    estimator : str
        Estimator type ('svm', 'logistic', 'ridge')
    cv_strategy : str
        Cross-validation strategy
    n_jobs : int
        Number of parallel jobs
    **kwargs : dict
        Additional parameters
    
    Returns:
    --------
    dict : Searchlight results
    """
    try:
        # Setup estimator
        if estimator == 'svm':
            clf = setup_classifier('svm', **kwargs)
        elif estimator == 'logistic':
            clf = setup_classifier('logistic', **kwargs)
        elif estimator == 'ridge':
            clf = setup_regressor('ridge', **kwargs)
        else:
            raise MVPAError(f"Unknown searchlight estimator: {estimator}")
        
        # Setup cross-validation
        cv = setup_cross_validation(cv_strategy, **kwargs)
        
        # Setup searchlight
        searchlight = SearchLight(
            mask_img=mask_img,
            radius=radius,
            estimator=clf,
            cv=cv,
            scoring='accuracy' if estimator in ['svm', 'logistic'] else 'r2',
            n_jobs=n_jobs,
            verbose=kwargs.get('verbose', 0)
        )
        
        # Fit searchlight
        searchlight.fit(img, y)
        
        return {
            'success': True,
            'searchlight': searchlight,
            'scores_img': searchlight.scores_,
            'estimator': estimator,
            'radius': radius,
            'cv_strategy': cv_strategy
        }
        
    except Exception as e:
        return {
            'success': False,
            'estimator': estimator,
            'error': str(e)
        }


def compute_feature_importance(X: np.ndarray, y: np.ndarray,
                             method: str = 'univariate',
                             task_type: str = 'classification',
                             **kwargs) -> Dict[str, Any]:
    """
    Compute feature importance/selection
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values
    method : str
        Importance method ('univariate', 'model_based', 'permutation')
    task_type : str
        Task type ('classification' or 'regression')
    **kwargs : dict
        Method-specific parameters
    
    Returns:
    --------
    dict : Feature importance results
    """
    try:
        validate_input_data(X, y, task_type)
        
        if method == 'univariate':
            # Univariate statistical tests
            if task_type == 'classification':
                selector = SelectKBest(f_classif, k='all')
            else:
                selector = SelectKBest(f_regression, k='all')
            
            selector.fit(X, y)
            scores = selector.scores_
            p_values = selector.pvalues_
            
            return {
                'success': True,
                'method': method,
                'scores': scores,
                'p_values': p_values,
                'ranking': np.argsort(scores)[::-1]
            }
        
        elif method == 'model_based':
            # Model-based feature importance
            if task_type == 'classification':
                estimator = setup_classifier('rf', **kwargs)
            else:
                estimator = setup_regressor('rf', **kwargs)
            
            estimator.fit(X, y)
            importance = estimator.feature_importances_
            
            return {
                'success': True,
                'method': method,
                'importance': importance,
                'ranking': np.argsort(importance)[::-1],
                'estimator': estimator
            }
        
        else:
            raise MVPAError(f"Unknown feature importance method: {method}")
        
    except Exception as e:
        return {
            'success': False,
            'method': method,
            'error': str(e)
        }


def run_permutation_test(X: np.ndarray, y: np.ndarray,
                        estimator: Any,
                        cv: Any = None,
                        n_permutations: int = None,
                        scoring: str = 'accuracy',
                        **kwargs) -> Dict[str, Any]:
    """
    Run permutation test for statistical significance
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    estimator : sklearn estimator
        Fitted estimator
    cv : cross-validator
        Cross-validation strategy
    n_permutations : int
        Number of permutations
    scoring : str
        Scoring metric
    **kwargs : dict
        Additional parameters
    
    Returns:
    --------
    dict : Permutation test results
    """
    try:
        n_permutations = n_permutations or MVPAConfig.N_PERMUTATIONS
        cv = cv or setup_cross_validation('stratified')
        
        score, perm_scores, p_value = permutation_test_score(
            estimator, X, y,
            cv=cv,
            n_permutations=n_permutations,
            scoring=scoring,
            n_jobs=MVPAConfig.N_JOBS,
            random_state=MVPAConfig.PERM_RANDOM_STATE
        )
        
        return {
            'success': True,
            'score': score,
            'permutation_scores': perm_scores,
            'p_value': p_value,
            'n_permutations': n_permutations,
            'scoring': scoring
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'scoring': scoring
        }


# Convenience functions for common use cases
def run_choice_classification(X: np.ndarray, choices: np.ndarray,
                            roi_name: str = 'unknown',
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function for choice classification
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns (n_trials, n_voxels)
    choices : np.ndarray
        Binary choice labels
    roi_name : str
        ROI name for reporting
    **kwargs : dict
        Additional parameters
    
    Returns:
    --------
    dict : Classification results
    """
    return run_classification(
        X, choices,
        algorithm='svm',
        cv_strategy='stratified',
        roi_name=roi_name,
        **kwargs
    )


def run_continuous_decoding(X: np.ndarray, values: np.ndarray,
                          variable_name: str = 'unknown',
                          roi_name: str = 'unknown',
                          **kwargs) -> Dict[str, Any]:
    """
    Convenience function for continuous variable decoding
    
    Parameters:
    -----------
    X : np.ndarray
        Neural patterns (n_trials, n_voxels)
    values : np.ndarray
        Continuous values to decode
    variable_name : str
        Variable name for reporting
    roi_name : str
        ROI name for reporting
    **kwargs : dict
        Additional parameters
    
    Returns:
    --------
    dict : Regression results
    """
    return run_regression(
        X, values,
        algorithm='ridge',
        cv_strategy='kfold',
        variable_name=variable_name,
        roi_name=roi_name,
        **kwargs
    )


# Configuration update function
def update_mvpa_config(**kwargs):
    """
    Update MVPA configuration parameters
    
    Parameters:
    -----------
    **kwargs : dict
        Configuration parameters to update
    """
    for key, value in kwargs.items():
        if hasattr(MVPAConfig, key.upper()):
            setattr(MVPAConfig, key.upper(), value)
        else:
            warnings.warn(f"Unknown configuration parameter: {key}")


# Export main functions
__all__ = [
    'MVPAError', 'MVPAConfig',
    'run_classification', 'run_regression',
    'extract_neural_patterns', 'run_dimensionality_reduction',
    'run_searchlight_analysis', 'compute_feature_importance',
    'run_permutation_test', 'run_choice_classification',
    'run_continuous_decoding', 'update_mvpa_config'
] 