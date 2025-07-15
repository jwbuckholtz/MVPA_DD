#!/usr/bin/env python3
"""
Behavioral Analysis Module
=========================

Refactored behavioral analysis class that inherits from BaseAnalysis.
Focuses on hyperbolic discounting model fitting and behavioral data processing.

Key Features:
- Hyperbolic discounting parameter estimation
- Choice behavior modeling
- Subjective value calculation
- Trial-wise behavioral variable extraction
- Model validation and quality control

Author: Cognitive Neuroscience Lab, Stanford University
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any
from scipy import stats, optimize
import warnings

# Import base analysis class
from analysis_base import BaseAnalysis, AnalysisError, AnalysisFactory
from oak_storage_config import OAKConfig
from data_utils import DataError


class BehavioralAnalysis(BaseAnalysis):
    """
    Behavioral analysis class for delay discounting data
    
    This class inherits from BaseAnalysis and implements behavioral modeling
    specific to delay discounting experiments, including hyperbolic discounting
    model fitting and behavioral variable extraction.
    """
    
    def __init__(self, config: OAKConfig = None, **kwargs):
        """
        Initialize behavioral analysis
        
        Parameters:
        -----------
        config : OAKConfig, optional
            Configuration object
        **kwargs : dict
            Additional arguments for base class
        """
        super().__init__(
            config=config,
            name='BehavioralAnalysis',
            **kwargs
        )
        
        # Behavioral analysis specific settings
        self.model_params = {
            'default_small_amount': 20.0,
            'k_bounds': (1e-6, 1.0),
            'min_trials': 10,
            'max_iterations': 100
        }
        
        self.logger.info("Behavioral analysis initialized")
    
    def _initialize_analysis_components(self):
        """Initialize behavioral analysis specific components"""
        # Initialize any behavioral-specific components here
        pass
    
    # Core behavioral modeling methods
    def hyperbolic_discount_function(self, delay: np.ndarray, k: float) -> np.ndarray:
        """
        Hyperbolic discounting function: V = 1 / (1 + k * delay)
        
        Parameters:
        -----------
        delay : np.ndarray
            Delay in days
        k : float
            Discount rate parameter
            
        Returns:
        --------
        np.ndarray : Discounted value
        """
        return 1 / (1 + k * delay)
    
    def subjective_value(self, amount: np.ndarray, delay: np.ndarray, k: float) -> np.ndarray:
        """
        Calculate subjective value given amount, delay, and discount rate
        
        Parameters:
        -----------
        amount : np.ndarray
            Monetary amount
        delay : np.ndarray
            Delay in days
        k : float
            Discount rate parameter
            
        Returns:
        --------
        np.ndarray : Subjective value
        """
        return amount * self.hyperbolic_discount_function(delay, k)
    
    def fit_discount_rate(self, choices: np.ndarray, large_amounts: np.ndarray, 
                         delays: np.ndarray, 
                         small_amount: float = None) -> Dict[str, Any]:
        """
        Fit hyperbolic discount rate to choice data using logistic regression
        
        Parameters:
        -----------
        choices : np.ndarray
            Binary choices (1 = larger_later, 0 = smaller_sooner)
        large_amounts : np.ndarray
            Amounts for larger_later option
        delays : np.ndarray
            Delays for larger_later option
        small_amount : float, optional
            Amount for smaller_sooner option
            
        Returns:
        --------
        Dict[str, Any] : Fitted parameters and model statistics
        """
        if small_amount is None:
            small_amount = self.model_params['default_small_amount']
        
        # Create difference in subjective value as function of k
        def neg_log_likelihood(k):
            if k <= 0:
                return np.inf
            
            # Calculate subjective values
            sv_large = self.subjective_value(large_amounts, delays, k)
            sv_small = small_amount  # Immediate reward, no discounting
            
            # Difference in subjective value (larger_later - smaller_sooner)
            sv_diff = sv_large - sv_small
            
            # Logistic choice probability
            choice_prob = 1 / (1 + np.exp(-sv_diff))
            
            # Avoid log(0)
            choice_prob = np.clip(choice_prob, 1e-15, 1 - 1e-15)
            
            # Negative log likelihood
            nll = -np.sum(choices * np.log(choice_prob) + 
                         (1 - choices) * np.log(1 - choice_prob))
            
            return nll
        
        # Fit model
        try:
            result = optimize.minimize_scalar(
                neg_log_likelihood, 
                bounds=self.model_params['k_bounds'], 
                method='bounded'
            )
            
            k_fit = result.x
            nll_fit = result.fun
            
            # Calculate model statistics
            sv_large_fit = self.subjective_value(large_amounts, delays, k_fit)
            sv_small_fit = small_amount
            sv_diff_fit = sv_large_fit - sv_small_fit
            choice_prob_fit = 1 / (1 + np.exp(-sv_diff_fit))
            
            # Pseudo R-squared
            null_ll = -np.sum(choices * np.log(np.mean(choices)) + 
                             (1 - choices) * np.log(1 - np.mean(choices)))
            pseudo_r2 = 1 - nll_fit / null_ll
            
            return {
                'k': k_fit,
                'nll': nll_fit,
                'pseudo_r2': pseudo_r2,
                'choice_prob': choice_prob_fit,
                'sv_large': sv_large_fit,
                'sv_small': sv_small_fit,
                'sv_diff': sv_diff_fit,
                'sv_sum': sv_large_fit + sv_small_fit,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Fitting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_behavioral_data(self, behavioral_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate behavioral data quality
        
        Parameters:
        -----------
        behavioral_data : pd.DataFrame
            Behavioral data to validate
            
        Returns:
        --------
        Dict[str, Any] : Validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check minimum number of trials
        n_trials = len(behavioral_data)
        validation_results['stats']['n_trials'] = n_trials
        
        if n_trials < self.model_params['min_trials']:
            validation_results['errors'].append(
                f"Insufficient trials: {n_trials} < {self.model_params['min_trials']}"
            )
            validation_results['valid'] = False
        
        # Check for choice variability
        if 'choice' in behavioral_data.columns:
            choices = behavioral_data['choice'].values
            valid_choices = choices[~np.isnan(choices)]
            
            if len(valid_choices) > 0:
                choice_rate = np.mean(valid_choices)
                validation_results['stats']['choice_rate'] = choice_rate
                
                if choice_rate == 0 or choice_rate == 1:
                    validation_results['errors'].append(
                        "No choice variability: all choices are the same"
                    )
                    validation_results['valid'] = False
                elif choice_rate < 0.05 or choice_rate > 0.95:
                    validation_results['warnings'].append(
                        f"Low choice variability: {choice_rate:.2f}"
                    )
        
        # Check for missing required columns
        required_columns = ['choice', 'large_amount', 'delay_days']
        missing_columns = [col for col in required_columns if col not in behavioral_data.columns]
        
        if missing_columns:
            validation_results['errors'].append(
                f"Missing required columns: {missing_columns}"
            )
            validation_results['valid'] = False
        
        return validation_results
    
    def process_subject(self, subject_id: str, **kwargs) -> Dict[str, Any]:
        """
        Process behavioral data for a single subject
        
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
            # Load behavioral data using base class method
            behavioral_data = self.load_behavioral_data(subject_id, **kwargs)
            
            # Validate data
            validation_results = self.validate_behavioral_data(behavioral_data)
            
            if not validation_results['valid']:
                return {
                    'subject_id': subject_id,
                    'success': False,
                    'error': f"Data validation failed: {validation_results['errors']}",
                    'validation_results': validation_results
                }
            
            # Extract choices and trial parameters
            choices = behavioral_data['choice'].values
            large_amounts = behavioral_data['large_amount'].values
            delays = behavioral_data['delay_days'].values
            
            # Remove trials with missing data
            valid_mask = ~(np.isnan(choices) | np.isnan(large_amounts) | np.isnan(delays))
            choices = choices[valid_mask]
            large_amounts = large_amounts[valid_mask]
            delays = delays[valid_mask]
            
            if len(choices) < self.model_params['min_trials']:
                return {
                    'subject_id': subject_id,
                    'success': False,
                    'error': f"Insufficient valid trials: {len(choices)}"
                }
            
            # Fit discount rate
            fit_results = self.fit_discount_rate(choices, large_amounts, delays)
            
            if not fit_results['success']:
                return {
                    'subject_id': subject_id,
                    'success': False,
                    'error': fit_results['error']
                }
            
            # Add fitted parameters to behavioral data
            behavioral_data['k'] = fit_results['k']
            behavioral_data['choice_binary'] = behavioral_data['choice']  # For compatibility
            
            # Add model predictions if not already present
            if 'choice_prob' not in behavioral_data.columns:
                behavioral_data.loc[valid_mask, 'choice_prob'] = fit_results['choice_prob']
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update processing stats
            self.update_processing_stats(subject_id, processing_time, success=True)
            
            # Store results
            result = {
                'subject_id': subject_id,
                'success': True,
                'data': behavioral_data,
                'k': fit_results['k'],
                'pseudo_r2': fit_results['pseudo_r2'],
                'n_trials': len(behavioral_data),
                'n_valid_trials': len(choices),
                'choice_rate': np.mean(choices),
                'processing_time': processing_time,
                'validation_results': validation_results
            }
            
            self.results[subject_id] = result
            
            self.logger.info(f"Processed {subject_id}: k={fit_results['k']:.4f}, "
                           f"R²={fit_results['pseudo_r2']:.3f}, "
                           f"trials={len(choices)}")
            
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
        Run behavioral analysis for multiple subjects
        
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
        
        self.logger.info(f"Starting behavioral analysis for {len(subjects)} subjects")
        
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
            k_values = [all_results[subj]['k'] for subj in successful_subjects]
            r2_values = [all_results[subj]['pseudo_r2'] for subj in successful_subjects]
            
            summary_stats = {
                'n_subjects_total': len(subjects),
                'n_subjects_successful': len(successful_subjects),
                'n_subjects_failed': len(failed_subjects),
                'success_rate': len(successful_subjects) / len(subjects),
                'k_mean': np.mean(k_values),
                'k_std': np.std(k_values),
                'k_median': np.median(k_values),
                'r2_mean': np.mean(r2_values),
                'r2_std': np.std(r2_values),
                'r2_median': np.median(r2_values)
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
        
        self.logger.info(f"Behavioral analysis complete: "
                        f"{len(successful_subjects)}/{len(subjects)} subjects successful")
        
        return {
            'results': all_results,
            'summary': summary_stats,
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects
        }
    
    def get_analysis_summary(self) -> str:
        """
        Get behavioral analysis specific summary
        
        Returns:
        --------
        str : Analysis summary
        """
        if '_summary' not in self.results:
            return "No analysis results available"
        
        summary = self.results['_summary']
        
        summary_text = f"""Behavioral Analysis Summary:
- Subjects processed: {summary.get('n_subjects_successful', 0)}/{summary.get('n_subjects_total', 0)}
- Success rate: {summary.get('success_rate', 0.0):.1%}
- Discount rate (k): {summary.get('k_mean', 0.0):.4f} ± {summary.get('k_std', 0.0):.4f}
- Model fit (R²): {summary.get('r2_mean', 0.0):.3f} ± {summary.get('r2_std', 0.0):.3f}"""
        
        return summary_text.strip()
    
    def create_behavioral_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of behavioral results
        
        Returns:
        --------
        pd.DataFrame : Summary of behavioral results
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for subject_id, result in self.results.items():
            if subject_id.startswith('_'):  # Skip summary entries
                continue
                
            if result['success']:
                summary_data.append({
                    'subject_id': subject_id,
                    'k': result['k'],
                    'pseudo_r2': result['pseudo_r2'],
                    'n_trials': result['n_trials'],
                    'n_valid_trials': result['n_valid_trials'],
                    'choice_rate': result['choice_rate'],
                    'processing_time': result['processing_time']
                })
        
        return pd.DataFrame(summary_data)


# Register the class with the factory
AnalysisFactory.register('behavioral', BehavioralAnalysis)


if __name__ == "__main__":
    # Example usage
    from oak_storage_config import OAKConfig
    
    # Create behavioral analysis instance
    config = OAKConfig()
    behavioral_analysis = BehavioralAnalysis(config)
    
    # Run analysis on a few subjects
    subjects = behavioral_analysis.get_subject_list()[:3]  # Just first 3 subjects
    results = behavioral_analysis.run_analysis(subjects)
    
    print("Analysis Results:")
    print(results['summary'])
    
    # Create summary dataframe
    summary_df = behavioral_analysis.create_behavioral_summary_dataframe()
    print("\nSummary DataFrame:")
    print(summary_df) 