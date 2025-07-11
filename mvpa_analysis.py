#!/usr/bin/env python3
"""
MVPA Analysis Module
==================

Refactored MVPA analysis class that inherits from BaseAnalysis.
Focuses on multi-voxel pattern analysis including decoding and pattern extraction.

Key Features:
- Neural pattern extraction from ROIs
- Choice decoding (binary classification)
- Continuous variable decoding (regression)
- Cross-validation and permutation testing
- Multiple classifier/regressor support
- Memory-efficient processing

Author: Cognitive Neuroscience Lab, Stanford University
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
import warnings

import nibabel as nib
from nilearn.input_data import NiftiMasker

# Import base analysis class
from analysis_base import BaseAnalysis, AnalysisError, AnalysisFactory
from oak_storage_config import OAKConfig
from data_utils import DataError

# Import MVPA utilities
from mvpa_utils import (
    run_classification, run_regression, extract_neural_patterns,
    run_dimensionality_reduction, MVPAError, update_mvpa_config
)


class MVPAAnalysis(BaseAnalysis):
    """
    MVPA analysis class for multi-voxel pattern analysis
    
    This class inherits from BaseAnalysis and implements MVPA-specific
    functionality including neural pattern extraction, decoding, and
    cross-validation procedures.
    """
    
    def __init__(self, config: OAKConfig = None, **kwargs):
        """
        Initialize MVPA analysis
        
        Parameters:
        -----------
        config : OAKConfig, optional
            Configuration object
        **kwargs : dict
            Additional arguments for base class
        """
        super().__init__(
            config=config,
            name='MVPAAnalysis',
            **kwargs
        )
        
        # MVPA-specific settings
        self.mvpa_params = {
            'algorithms': {
                'classification': ['svm', 'logistic', 'rf'],
                'regression': ['ridge', 'lasso', 'svr']
            },
            'cv_folds': self.config.CV_FOLDS,
            'n_permutations': self.config.N_PERMUTATIONS,
            'pattern_extraction': {
                'method': 'single_timepoint',
                'hemi_lag': self.config.HEMI_LAG,
                'window_size': 3
            }
        }
        
        # Initialize maskers dictionary
        self.maskers = {}
        
        # Configure MVPA utilities
        update_mvpa_config(
            cv_folds=self.mvpa_params['cv_folds'],
            n_permutations=self.mvpa_params['n_permutations'],
            n_jobs=1  # Conservative for memory management
        )
        
        self.logger.info("MVPA analysis initialized")
    
    def _initialize_analysis_components(self):
        """Initialize MVPA analysis specific components"""
        # Create maskers for ROIs
        self.maskers = self.create_maskers()
        
        # Create whole-brain masker if needed
        if 'whole_brain' not in self.maskers:
            self.create_whole_brain_masker()
        
        self.logger.info(f"Initialized {len(self.maskers)} maskers")
    
    def create_whole_brain_masker(self):
        """Create whole-brain masker"""
        self.maskers['whole_brain'] = NiftiMasker(
            standardize=True,
            detrend=True,
            high_pass=0.01,
            t_r=self.config.TR,
            memory='nilearn_cache',
            memory_level=1
        )
        self.logger.info("Created whole-brain masker")
    
    def extract_trial_data(self, img: nib.Nifti1Image, events_df: pd.DataFrame, 
                          roi_name: str, confounds: np.ndarray = None) -> np.ndarray:
        """
        Extract trial-wise neural data using pattern extraction
        
        Parameters:
        -----------
        img : nib.Nifti1Image
            fMRI data
        events_df : pd.DataFrame
            Trial events with onsets
        roi_name : str
            Name of ROI to extract from
        confounds : np.ndarray, optional
            Confound regressors
            
        Returns:
        --------
        np.ndarray : Trial-wise neural data (n_trials x n_voxels)
        """
        if roi_name not in self.maskers:
            raise AnalysisError(f"Masker for {roi_name} not found")
        
        masker = self.maskers[roi_name]
        
        # Use centralized pattern extraction
        result = extract_neural_patterns(
            img, events_df, masker,
            confounds=confounds,
            pattern_type=self.mvpa_params['pattern_extraction']['method'],
            tr=self.config.TR,
            hemi_lag=self.mvpa_params['pattern_extraction']['hemi_lag']
        )
        
        if result['success']:
            return result['patterns']
        else:
            raise AnalysisError(f"Pattern extraction failed: {result['error']}")
    
    def decode_choices(self, X: np.ndarray, y: np.ndarray, roi_name: str) -> Dict[str, Any]:
        """
        Decode choice (smaller_sooner vs larger_later) from neural data
        
        Parameters:
        -----------
        X : np.ndarray
            Neural data (n_trials x n_voxels)
        y : np.ndarray
            Choice labels (binary)
        roi_name : str
            Name of ROI
            
        Returns:
        --------
        Dict[str, Any] : Decoding results
        """
        try:
            result = run_classification(
                X, y, 
                roi_name=roi_name,
                algorithm='svm',
                cv_strategy='stratified',
                n_permutations=self.mvpa_params['n_permutations']
            )
            
            # Add ROI name to results
            result['roi_name'] = roi_name
            result['analysis_type'] = 'choice_decoding'
            
            return result
            
        except MVPAError as e:
            self.logger.error(f"Choice decoding failed for {roi_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'roi_name': roi_name,
                'analysis_type': 'choice_decoding'
            }
    
    def decode_continuous_variable(self, X: np.ndarray, y: np.ndarray, 
                                  roi_name: str, variable_name: str) -> Dict[str, Any]:
        """
        Decode continuous variable from neural data
        
        Parameters:
        -----------
        X : np.ndarray
            Neural data (n_trials x n_voxels)
        y : np.ndarray
            Continuous variable values
        roi_name : str
            Name of ROI
        variable_name : str
            Name of variable being decoded
            
        Returns:
        --------
        Dict[str, Any] : Decoding results
        """
        try:
            result = run_regression(
                X, y,
                variable_name=variable_name,
                roi_name=roi_name,
                algorithm='ridge',
                cv_strategy='kfold',
                n_permutations=self.mvpa_params['n_permutations']
            )
            
            # Add metadata to results
            result['roi_name'] = roi_name
            result['variable_name'] = variable_name
            result['analysis_type'] = 'continuous_decoding'
            
            return result
            
        except MVPAError as e:
            self.logger.error(f"Continuous decoding failed for {roi_name}, {variable_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'roi_name': roi_name,
                'variable_name': variable_name,
                'analysis_type': 'continuous_decoding'
            }
    
    def process_subject(self, subject_id: str, **kwargs) -> Dict[str, Any]:
        """
        Process MVPA analysis for a single subject
        
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
            
            # Process each ROI
            roi_results = {}
            available_rois = list(self.maskers.keys())
            
            for roi_name in available_rois:
                roi_start_time = time.time()
                
                try:
                    # Extract neural patterns
                    X = self.extract_trial_data(img, behavioral_data, roi_name, confounds)
                    
                    # Validate neural data
                    if X.shape[0] != len(behavioral_data):
                        self.logger.warning(f"Trial count mismatch for {roi_name}: "
                                          f"neural={X.shape[0]}, behavioral={len(behavioral_data)}")
                        continue
                    
                    # Choice decoding
                    if 'choice' in behavioral_data.columns:
                        choices = behavioral_data['choice'].values
                        valid_mask = ~np.isnan(choices)
                        
                        if np.sum(valid_mask) > 10:  # Minimum trials threshold
                            choice_result = self.decode_choices(
                                X[valid_mask], choices[valid_mask], roi_name
                            )
                        else:
                            choice_result = {
                                'success': False,
                                'error': 'Insufficient valid trials for choice decoding',
                                'roi_name': roi_name,
                                'analysis_type': 'choice_decoding'
                            }
                    else:
                        choice_result = {
                            'success': False,
                            'error': 'No choice data available',
                            'roi_name': roi_name,
                            'analysis_type': 'choice_decoding'
                        }
                    
                    # Continuous variable decoding
                    continuous_results = {}
                    continuous_vars = ['sv_diff', 'sv_sum', 'sv_chosen', 'sv_unchosen']
                    
                    for var_name in continuous_vars:
                        if var_name in behavioral_data.columns:
                            var_values = behavioral_data[var_name].values
                            valid_mask = ~np.isnan(var_values)
                            
                            if np.sum(valid_mask) > 10:  # Minimum trials threshold
                                var_result = self.decode_continuous_variable(
                                    X[valid_mask], var_values[valid_mask], roi_name, var_name
                                )
                            else:
                                var_result = {
                                    'success': False,
                                    'error': 'Insufficient valid trials',
                                    'roi_name': roi_name,
                                    'variable_name': var_name,
                                    'analysis_type': 'continuous_decoding'
                                }
                        else:
                            var_result = {
                                'success': False,
                                'error': f'Variable {var_name} not found in behavioral data',
                                'roi_name': roi_name,
                                'variable_name': var_name,
                                'analysis_type': 'continuous_decoding'
                            }
                        
                        continuous_results[var_name] = var_result
                    
                    # Store ROI results
                    roi_processing_time = time.time() - roi_start_time
                    roi_results[roi_name] = {
                        'success': True,
                        'choice_decoding': choice_result,
                        'continuous_decoding': continuous_results,
                        'n_voxels': X.shape[1],
                        'n_trials': X.shape[0],
                        'processing_time': roi_processing_time
                    }
                    
                    self.logger.info(f"Processed {roi_name} for {subject_id}: "
                                   f"{X.shape[1]} voxels, {X.shape[0]} trials")
                    
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
        Run MVPA analysis for multiple subjects
        
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
        
        self.logger.info(f"Starting MVPA analysis for {len(subjects)} subjects")
        
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
            # Collect accuracy scores for choice decoding
            choice_accuracies = {}
            continuous_scores = {}
            
            for subject_id in successful_subjects:
                subject_result = all_results[subject_id]
                
                for roi_name, roi_result in subject_result['roi_results'].items():
                    if roi_result['success']:
                        # Choice decoding accuracy
                        choice_result = roi_result['choice_decoding']
                        if choice_result['success']:
                            if roi_name not in choice_accuracies:
                                choice_accuracies[roi_name] = []
                            choice_accuracies[roi_name].append(choice_result['accuracy'])
                        
                        # Continuous decoding scores
                        for var_name, var_result in roi_result['continuous_decoding'].items():
                            if var_result['success']:
                                key = f"{roi_name}_{var_name}"
                                if key not in continuous_scores:
                                    continuous_scores[key] = []
                                continuous_scores[key].append(var_result['score'])
            
            # Calculate summary statistics
            choice_summary = {}
            for roi_name, accuracies in choice_accuracies.items():
                choice_summary[roi_name] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'n_subjects': len(accuracies)
                }
            
            continuous_summary = {}
            for key, scores in continuous_scores.items():
                continuous_summary[key] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'n_subjects': len(scores)
                }
            
            summary_stats = {
                'n_subjects_total': len(subjects),
                'n_subjects_successful': len(successful_subjects),
                'n_subjects_failed': len(failed_subjects),
                'success_rate': len(successful_subjects) / len(subjects),
                'choice_decoding_summary': choice_summary,
                'continuous_decoding_summary': continuous_summary
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
        
        self.logger.info(f"MVPA analysis complete: "
                        f"{len(successful_subjects)}/{len(subjects)} subjects successful")
        
        return {
            'results': all_results,
            'summary': summary_stats,
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects
        }
    
    def get_analysis_summary(self) -> str:
        """
        Get MVPA analysis specific summary
        
        Returns:
        --------
        str : Analysis summary
        """
        if '_summary' not in self.results:
            return "No analysis results available"
        
        summary = self.results['_summary']
        
        # Format choice decoding summary
        choice_summary = summary.get('choice_decoding_summary', {})
        choice_lines = []
        for roi_name, stats in choice_summary.items():
            accuracy = stats['mean_accuracy']
            n_subjects = stats['n_subjects']
            choice_lines.append(f"  {roi_name}: {accuracy:.3f} ± {stats['std_accuracy']:.3f} (n={n_subjects})")
        
        # Format continuous decoding summary
        continuous_summary = summary.get('continuous_decoding_summary', {})
        continuous_lines = []
        for key, stats in continuous_summary.items():
            score = stats['mean_score']
            n_subjects = stats['n_subjects']
            continuous_lines.append(f"  {key}: {score:.3f} ± {stats['std_score']:.3f} (n={n_subjects})")
        
        summary_text = f"""MVPA Analysis Summary:
- Subjects processed: {summary.get('n_subjects_successful', 0)}/{summary.get('n_subjects_total', 0)}
- Success rate: {summary.get('success_rate', 0.0):.1%}

Choice Decoding Results:
{chr(10).join(choice_lines) if choice_lines else "  No successful choice decoding results"}

Continuous Decoding Results:
{chr(10).join(continuous_lines) if continuous_lines else "  No successful continuous decoding results"}"""
        
        return summary_text
    
    def create_mvpa_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of MVPA results
        
        Returns:
        --------
        pd.DataFrame : Summary of MVPA results
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
                        # Choice decoding row
                        choice_result = roi_result['choice_decoding']
                        if choice_result['success']:
                            summary_data.append({
                                'subject_id': subject_id,
                                'roi_name': roi_name,
                                'analysis_type': 'choice_decoding',
                                'variable_name': 'choice',
                                'score': choice_result['accuracy'],
                                'p_value': choice_result.get('p_value', np.nan),
                                'n_trials': roi_result['n_trials'],
                                'n_voxels': roi_result['n_voxels']
                            })
                        
                        # Continuous decoding rows
                        for var_name, var_result in roi_result['continuous_decoding'].items():
                            if var_result['success']:
                                summary_data.append({
                                    'subject_id': subject_id,
                                    'roi_name': roi_name,
                                    'analysis_type': 'continuous_decoding',
                                    'variable_name': var_name,
                                    'score': var_result['score'],
                                    'p_value': var_result.get('p_value', np.nan),
                                    'n_trials': roi_result['n_trials'],
                                    'n_voxels': roi_result['n_voxels']
                                })
        
        return pd.DataFrame(summary_data)


# Register the class with the factory
AnalysisFactory.register('mvpa', MVPAAnalysis)


if __name__ == "__main__":
    # Example usage
    from oak_storage_config import OAKConfig
    
    # Create MVPA analysis instance
    config = OAKConfig()
    mvpa_analysis = MVPAAnalysis(config)
    
    # Run analysis on a few subjects
    subjects = mvpa_analysis.get_subject_list()[:2]  # Just first 2 subjects
    results = mvpa_analysis.run_analysis(subjects)
    
    print("Analysis Results:")
    print(results['summary'])
    
    # Create summary dataframe
    summary_df = mvpa_analysis.create_mvpa_summary_dataframe()
    print("\nSummary DataFrame:")
    print(summary_df) 