#!/usr/bin/env python3
"""
Mass Univariate Analysis Module for Delay Discounting fMRI Data

This module performs whole-brain GLM analysis including:
1. Spatial smoothing of preprocessed data
2. First-level (subject-wise) GLM modeling
3. Second-level (group) random effects analysis
4. Multiple comparisons correction (FDR/permutation)

Models:
- Model 1: Choice (sooner_smaller vs larger_later)
- Model 2: Subjective Value - Chosen option
- Model 3: Subjective Value - Unchosen option  
- Model 4: Both SV chosen and unchosen + contrast
- Model 5: Choice difficulty (SV difference)

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Neuroimaging
import nibabel as nib
from nilearn import image, plotting, datasets
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.image import smooth_img, math_img, mean_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, plot_glass_brain

# Statistical analysis
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats

# Parallel processing
from joblib import Parallel, delayed

# Configuration
from oak_storage_config import OAKConfig as Config

class MassUnivariateAnalysis:
    """Class for mass univariate GLM analysis of delay discounting data"""
    
    def __init__(self, config):
        self.config = config
        self.models = self._define_models()
        self.first_level_dir = f"{self.config.OUTPUT_DIR}/mass_univariate/first_level"
        self.second_level_dir = f"{self.config.OUTPUT_DIR}/mass_univariate/second_level"
        self.smoothed_data_dir = f"{self.config.OUTPUT_DIR}/mass_univariate/smoothed_data"
        
        # Create output directories
        for directory in [self.first_level_dir, self.second_level_dir, self.smoothed_data_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _define_models(self):
        """Define the 5 GLM models to be estimated"""
        models = {
            'model_1_choice': {
                'name': 'Choice (Sooner-Smaller vs Larger-Later)',
                'regressors': ['choice'],
                'contrasts': {'choice_effect': 'choice'}
            },
            'model_2_sv_chosen': {
                'name': 'Subjective Value - Chosen Option',
                'regressors': ['sv_chosen'],
                'contrasts': {'sv_chosen_effect': 'sv_chosen'}
            },
            'model_3_sv_unchosen': {
                'name': 'Subjective Value - Unchosen Option', 
                'regressors': ['sv_unchosen'],
                'contrasts': {'sv_unchosen_effect': 'sv_unchosen'}
            },
            'model_4_sv_both': {
                'name': 'SV Chosen vs Unchosen',
                'regressors': ['sv_chosen', 'sv_unchosen'],
                'contrasts': {
                    'sv_chosen_effect': 'sv_chosen',
                    'sv_unchosen_effect': 'sv_unchosen', 
                    'sv_chosen_vs_unchosen': 'sv_chosen - sv_unchosen'
                }
            },
            'model_5_difficulty': {
                'name': 'Choice Difficulty (SV Difference)',
                'regressors': ['sv_difference', 'choice'],
                'contrasts': {
                    'difficulty_effect': 'sv_difference',
                    'choice_effect': 'choice'
                }
            }
        }
        return models
    
    def smooth_subject_data(self, worker_id, fwhm=4.0):
        """
        Apply spatial smoothing to subject's preprocessed fMRI data
        
        Parameters:
        -----------
        worker_id : str
            Subject identifier
        fwhm : float
            Full-width half-maximum for Gaussian smoothing kernel (mm)
            
        Returns:
        --------
        str : Path to smoothed data file
        """
        print(f"Smoothing data for subject {worker_id} with {fwhm}mm FWHM...")
        
        # Find preprocessed fMRI file
        fmri_dir = f"{self.config.FMRIPREP_DIR}/{worker_id}/ses-2/func"
        pattern = f"{worker_id}_ses-2_task-discountFix_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        
        fmri_files = list(Path(fmri_dir).glob(pattern))
        if not fmri_files:
            raise FileNotFoundError(f"No fMRI files found for {worker_id}")
        
        # Use first run if multiple exist
        fmri_file = fmri_files[0]
        
        # Load and smooth data
        img = nib.load(str(fmri_file))
        smoothed_img = smooth_img(img, fwhm=fwhm)
        
        # Save smoothed data
        smoothed_file = f"{self.smoothed_data_dir}/{worker_id}_ses-2_task-discountFix_smoothed-{fwhm}mm_bold.nii.gz"
        smoothed_img.to_filename(smoothed_file)
        
        print(f"Smoothed data saved: {smoothed_file}")
        return smoothed_file
    
    def load_behavioral_data(self, worker_id):
        """
        Load and process behavioral data for GLM design matrix
        
        Parameters:
        -----------
        worker_id : str
            Subject identifier
            
        Returns:
        --------
        pd.DataFrame : Processed behavioral data with trial timing
        """
        behavior_file = f"{self.config.BEHAVIOR_DIR}/{worker_id}_discountFix_events.tsv"
        
        if not os.path.exists(behavior_file):
            raise FileNotFoundError(f"Behavioral file not found: {behavior_file}")
        
        # Load behavioral data
        events_df = pd.read_csv(behavior_file, sep='\t')
        
        # Process and create regressors
        events_df['choice'] = events_df['response'].map({1: 0, 2: 1})  # 0=SS, 1=LL
        
        # Create trial_type based on choice for proper contrast setup
        events_df['trial_type'] = events_df['choice'].map({0: 'sooner_smaller', 1: 'larger_later'})
        
        # Calculate subjective values (assuming they exist or need to be computed)
        if 'sv_chosen' not in events_df.columns:
            # If SV not pre-computed, calculate using behavioral model
            events_df = self._compute_subjective_values(events_df, worker_id)
        
        # Create additional regressors
        events_df['sv_difference'] = np.abs(events_df['sv_chosen'] - events_df['sv_unchosen'])
        
        # Set trial duration (4000ms = 4 seconds)
        events_df['duration'] = 4.0
        
        # trial_type already set based on choice above
        
        # Remove invalid trials
        valid_mask = (~events_df['choice'].isna()) & (~events_df['sv_chosen'].isna())
        events_df = events_df[valid_mask].reset_index(drop=True)
        
        return events_df
    
    def _compute_subjective_values(self, events_df, worker_id):
        """
        Compute subjective values using hyperbolic discounting model
        
        Parameters:
        -----------
        events_df : pd.DataFrame
            Events dataframe
        worker_id : str
            Subject identifier
            
        Returns:
        --------
        pd.DataFrame : Events with computed subjective values
        """
        # Import behavioral analysis
        from delay_discounting_mvpa_pipeline import BehavioralAnalysis
        
        behavioral_analysis = BehavioralAnalysis(self.config)
        
        # Extract choice data for fitting
        choices = events_df['choice'].values
        large_amounts = events_df['large_amount'].values
        delays = events_df['delay_days'].values
        
        # Fit discount rate
        fit_result = behavioral_analysis.fit_discount_rate(choices, large_amounts, delays)
        
        if fit_result['success']:
            k = fit_result['k']
            
            # Calculate subjective values
            sv_large = behavioral_analysis.subjective_value(large_amounts, delays, k)
            sv_small = 20  # Immediate reward amount
            
            # Assign chosen/unchosen based on choice
            events_df['sv_chosen'] = np.where(choices == 1, sv_large, sv_small)
            events_df['sv_unchosen'] = np.where(choices == 1, sv_small, sv_large)
        else:
            print(f"Warning: Could not fit discount rate for {worker_id}, using default values")
            events_df['sv_chosen'] = events_df['large_amount'] / 2  # Fallback
            events_df['sv_unchosen'] = events_df['large_amount'] / 2
        
        return events_df
    
    def load_confounds(self, worker_id):
        """
        Load fMRIPrep confounds for GLM
        
        Parameters:
        -----------
        worker_id : str
            Subject identifier
            
        Returns:
        --------
        pd.DataFrame : Confounds dataframe
        """
        # Find confounds file
        fmri_dir = f"{self.config.FMRIPREP_DIR}/{worker_id}/ses-2/func"
        pattern = f"{worker_id}_ses-2_task-discountFix_*_desc-confounds_timeseries.tsv"
        
        confounds_files = list(Path(fmri_dir).glob(pattern))
        if not confounds_files:
            print(f"Warning: No confounds file found for {worker_id}")
            return None
        
        confounds_file = confounds_files[0]
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        
        # Select relevant confounds
        confound_cols = []
        for pattern in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                       'csf', 'white_matter', 'global_signal']:
            matching_cols = [col for col in confounds_df.columns if pattern in col]
            confound_cols.extend(matching_cols[:1])  # Take first match for each pattern
        
        # Add motion derivatives if available
        for pattern in ['trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                       'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1']:
            if pattern in confounds_df.columns:
                confound_cols.append(pattern)
        
        if confound_cols:
            return confounds_df[confound_cols].fillna(0)
        else:
            print(f"Warning: No suitable confounds found for {worker_id}")
            return None
    
    def run_first_level_model(self, worker_id, model_name):
        """
        Run first-level GLM for a single subject and model
        
        Parameters:
        -----------
        worker_id : str
            Subject identifier
        model_name : str
            Name of model to run
            
        Returns:
        --------
        dict : First-level results and file paths
        """
        print(f"Running first-level model '{model_name}' for subject {worker_id}...")
        
        try:
            # Load smoothed data
            smoothed_file = f"{self.smoothed_data_dir}/{worker_id}_ses-2_task-discountFix_smoothed-4.0mm_bold.nii.gz"
            if not os.path.exists(smoothed_file):
                smoothed_file = self.smooth_subject_data(worker_id)
            
            # Load behavioral data
            events_df = self.load_behavioral_data(worker_id)
            
            # Load confounds
            confounds_df = self.load_confounds(worker_id)
            
            # Get model specification
            model_spec = self.models[model_name]
            
            # Create design matrix for this model - include trial_type as required by nilearn
            required_cols = ['onset', 'duration', 'trial_type'] + model_spec['regressors']
            design_events = events_df[required_cols].copy()
            
            # Initialize first-level model
            first_level_model = FirstLevelModel(
                t_r=self.config.TR,
                slice_time_ref=0.5,
                hrf_model='spm',
                drift_model='cosine',
                high_pass=1/128,  # 128 second high-pass filter
                standardize=False,
                signal_scaling=0,
                smoothing_fwhm=None  # Already smoothed
            )
            
            # Fit model
            first_level_model = first_level_model.fit(
                smoothed_file,
                events=design_events,
                confounds=confounds_df
            )
            
            # Get design matrix to check available columns
            design_matrix = first_level_model.design_matrices_[0]
            available_columns = list(design_matrix.columns)
            print(f"Available design matrix columns for {worker_id}: {available_columns}")
            
            # Compute contrasts
            contrast_maps = {}
            for contrast_name, contrast_def in model_spec['contrasts'].items():
                try:
                    # Check if contrast definition needs to be mapped to actual column names
                    if contrast_def == 'choice':
                        # Map choice contrast to actual available columns
                        if 'larger_later' in available_columns and 'sooner_smaller' in available_columns:
                            actual_contrast = 'larger_later - sooner_smaller'
                        elif 'larger_later' in available_columns:
                            actual_contrast = 'larger_later'
                        elif 'sooner_smaller' in available_columns:
                            actual_contrast = 'sooner_smaller'
                        else:
                            print(f"Warning: No choice-related columns found for {worker_id}")
                            continue
                    else:
                        actual_contrast = contrast_def
                    
                    contrast_map = first_level_model.compute_contrast(
                        actual_contrast, output_type='z_score'
                    )
                    
                    # Save contrast map
                    contrast_file = f"{self.first_level_dir}/{worker_id}_{model_name}_{contrast_name}_zstat.nii.gz"
                    contrast_map.to_filename(contrast_file)
                    contrast_maps[contrast_name] = contrast_file
                    
                except Exception as e:
                    print(f"Warning: Could not compute contrast '{contrast_name}' for {worker_id}: {e}")
                    continue
            
            return {
                'success': True,
                'worker_id': worker_id,
                'model_name': model_name,
                'contrast_maps': contrast_maps,
                'design_matrix': first_level_model.design_matrices_[0]
            }
            
        except Exception as e:
            print(f"Error in first-level model for {worker_id}, {model_name}: {e}")
            return {
                'success': False,
                'worker_id': worker_id,
                'model_name': model_name,
                'error': str(e)
            }
    
    def run_second_level_model(self, model_name, contrast_name, subject_list):
        """
        Run second-level (group) random effects analysis
        
        Parameters:
        -----------
        model_name : str
            Name of first-level model
        contrast_name : str
            Name of contrast
        subject_list : list
            List of subject IDs
            
        Returns:
        --------
        dict : Second-level results
        """
        print(f"Running second-level analysis for {model_name} - {contrast_name}...")
        
        # Collect first-level contrast maps
        contrast_maps = []
        valid_subjects = []
        
        for worker_id in subject_list:
            contrast_file = f"{self.first_level_dir}/{worker_id}_{model_name}_{contrast_name}_zstat.nii.gz"
            if os.path.exists(contrast_file):
                contrast_maps.append(contrast_file)
                valid_subjects.append(worker_id)
        
        if len(contrast_maps) < 3:
            print(f"Warning: Insufficient subjects ({len(contrast_maps)}) for {model_name} - {contrast_name}")
            return {'success': False, 'error': 'Insufficient subjects'}
        
        print(f"Including {len(contrast_maps)} subjects in group analysis")
        
        # Create design matrix (intercept only for one-sample t-test)
        design_matrix = pd.DataFrame({'intercept': np.ones(len(contrast_maps))})
        
        # Initialize second-level model
        second_level_model = SecondLevelModel()
        
        # Fit model
        second_level_model = second_level_model.fit(
            contrast_maps,
            design_matrix=design_matrix
        )
        
        # Compute group-level contrast
        group_stat_map = second_level_model.compute_contrast(
            'intercept', output_type='z_score'
        )
        
        # Save unthresholded map
        output_file = f"{self.second_level_dir}/{model_name}_{contrast_name}_group_zstat.nii.gz"
        group_stat_map.to_filename(output_file)
        
        # Apply multiple comparisons correction
        corrected_results = self.apply_multiple_comparisons_correction(
            group_stat_map, model_name, contrast_name
        )
        
        return {
            'success': True,
            'model_name': model_name,
            'contrast_name': contrast_name,
            'n_subjects': len(contrast_maps),
            'valid_subjects': valid_subjects,
            'unthresholded_map': output_file,
            'corrected_results': corrected_results
        }
    
    def apply_multiple_comparisons_correction(self, stat_map, model_name, contrast_name, alpha=0.05):
        """
        Apply multiple comparisons correction using FDR and cluster-based methods
        
        Parameters:
        -----------
        stat_map : Nifti image
            Statistical map to threshold
        model_name : str
            Model name for file naming
        contrast_name : str
            Contrast name for file naming
        alpha : float
            Significance threshold
            
        Returns:
        --------
        dict : Corrected results
        """
        print(f"Applying multiple comparisons correction (alpha = {alpha})...")
        
        results = {}
        
        try:
            # FDR correction
            fdr_map, fdr_threshold = threshold_stats_img(
                stat_map,
                alpha=alpha,
                height_control='fdr',
                cluster_threshold=0
            )
            
            fdr_file = f"{self.second_level_dir}/{model_name}_{contrast_name}_group_zstat_fdr{alpha}.nii.gz"
            fdr_map.to_filename(fdr_file)
            
            results['fdr'] = {
                'thresholded_map': fdr_file,
                'threshold': fdr_threshold,
                'method': f'FDR q < {alpha}'
            }
            
        except Exception as e:
            print(f"Warning: FDR correction failed: {e}")
            results['fdr'] = {'error': str(e)}
        
        try:
            # Cluster-based correction (cluster-forming threshold p < 0.001)
            cluster_map, cluster_threshold = threshold_stats_img(
                stat_map,
                alpha=alpha,
                height_control='fpr',
                cluster_threshold=10  # Minimum cluster size
            )
            
            cluster_file = f"{self.second_level_dir}/{model_name}_{contrast_name}_group_zstat_cluster{alpha}.nii.gz"
            cluster_map.to_filename(cluster_file)
            
            results['cluster'] = {
                'thresholded_map': cluster_file,
                'threshold': cluster_threshold,
                'method': f'Cluster-corrected p < {alpha}'
            }
            
        except Exception as e:
            print(f"Warning: Cluster correction failed: {e}")
            results['cluster'] = {'error': str(e)}
        
        return results
    
    def generate_summary_report(self, results):
        """
        Generate summary report of all analyses
        
        Parameters:
        -----------
        results : dict
            All analysis results
        """
        report_file = f"{self.second_level_dir}/mass_univariate_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("DELAY DISCOUNTING MASS UNIVARIATE ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis completed: {pd.Timestamp.now()}\n")
            f.write(f"Total subjects analyzed: {len(results.get('subject_list', []))}\n\n")
            
            f.write("MODELS ANALYZED:\n")
            f.write("-" * 20 + "\n")
            for model_name, model_spec in self.models.items():
                f.write(f"{model_name}: {model_spec['name']}\n")
                f.write(f"  Regressors: {', '.join(model_spec['regressors'])}\n")
                f.write(f"  Contrasts: {', '.join(model_spec['contrasts'].keys())}\n\n")
            
            f.write("SECOND-LEVEL RESULTS:\n")
            f.write("-" * 25 + "\n")
            
            for model_name in self.models.keys():
                if model_name in results.get('second_level', {}):
                    f.write(f"\n{model_name.upper()}:\n")
                    model_results = results['second_level'][model_name]
                    
                    for contrast_name, contrast_results in model_results.items():
                        if contrast_results.get('success', False):
                            f.write(f"  {contrast_name}:\n")
                            f.write(f"    Subjects: {contrast_results['n_subjects']}\n")
                            f.write(f"    Unthresholded map: {contrast_results['unthresholded_map']}\n")
                            
                            corrected = contrast_results.get('corrected_results', {})
                            if 'fdr' in corrected and 'error' not in corrected['fdr']:
                                f.write(f"    FDR corrected: {corrected['fdr']['thresholded_map']}\n")
                                f.write(f"    FDR threshold: {corrected['fdr']['threshold']:.3f}\n")
                            
                            if 'cluster' in corrected and 'error' not in corrected['cluster']:
                                f.write(f"    Cluster corrected: {corrected['cluster']['thresholded_map']}\n")
                                f.write(f"    Cluster threshold: {corrected['cluster']['threshold']:.3f}\n")
            
            f.write(f"\nOUTPUT DIRECTORIES:\n")
            f.write(f"  First-level results: {self.first_level_dir}\n")
            f.write(f"  Second-level results: {self.second_level_dir}\n")
            f.write(f"  Smoothed data: {self.smoothed_data_dir}\n")
        
        print(f"Summary report saved: {report_file}")

def run_full_mass_univariate_pipeline(subject_list, config=None):
    """
    Run the complete mass univariate analysis pipeline
    
    Parameters:
    -----------
    subject_list : list
        List of subject IDs to analyze
    config : Config object, optional
        Configuration object
        
    Returns:
    --------
    dict : Complete analysis results
    """
    if config is None:
        config = Config()
    
    # Initialize analysis
    analysis = MassUnivariateAnalysis(config)
    
    # Store all results
    results = {
        'subject_list': subject_list,
        'first_level': {},
        'second_level': {}
    }
    
    print("STARTING MASS UNIVARIATE ANALYSIS PIPELINE")
    print("=" * 50)
    
    # Step 1: Smooth all subject data
    print("\nStep 1: Smoothing subject data...")
    for worker_id in subject_list:
        try:
            analysis.smooth_subject_data(worker_id)
        except Exception as e:
            print(f"Warning: Could not smooth data for {worker_id}: {e}")
    
    # Step 2: Run first-level models
    print("\nStep 2: Running first-level models...")
    for model_name in analysis.models.keys():
        print(f"\nProcessing {model_name}...")
        results['first_level'][model_name] = {}
        
        for worker_id in subject_list:
            result = analysis.run_first_level_model(worker_id, model_name)
            results['first_level'][model_name][worker_id] = result
    
    # Step 3: Run second-level models
    print("\nStep 3: Running second-level models...")
    for model_name in analysis.models.keys():
        print(f"\nProcessing group analysis for {model_name}...")
        results['second_level'][model_name] = {}
        
        for contrast_name in analysis.models[model_name]['contrasts'].keys():
            result = analysis.run_second_level_model(model_name, contrast_name, subject_list)
            results['second_level'][model_name][contrast_name] = result
    
    # Step 4: Generate summary report
    print("\nStep 4: Generating summary report...")
    analysis.generate_summary_report(results)
    
    # Save complete results
    results_file = f"{analysis.second_level_dir}/complete_mass_univariate_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nAnalysis complete! Results saved to: {results_file}")
    return results

if __name__ == "__main__":
    # Get subject list (you may need to modify this based on your data structure)
    config = Config()
    
    # Example subject list - replace with your actual subjects
    subject_list = ['s001', 's002', 's003']  # Add your actual subject IDs
    
    # Run analysis
    results = run_full_mass_univariate_pipeline(subject_list, config) 