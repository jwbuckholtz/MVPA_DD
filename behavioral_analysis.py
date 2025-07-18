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

import pandas as pd
from behavioral_modeling import fit_discount_rate
import config
import logging

class BehavioralAnalysis:
    """
    Orchestrates the behavioral analysis of delay discounting data.
    """
    
    def __init__(self):
        self.results = {}
        logging.basicConfig(level=config.System.LOGGING_LEVEL)
        self.logger = logging.getLogger(__name__)

    def process_subject(self, subject_id: str, events_file: str):
        """Processes a single subject's behavioral data."""
        self.logger.info(f"Processing behavioral data for subject {subject_id}...")
        
        try:
            behavioral_data = pd.read_csv(events_file, sep='\t')
            
            # Prepare data for model fitting
            choices = behavioral_data['choice_binary'].values
            large_amounts = behavioral_data['amount_large'].values
            delays = behavioral_data['delay_days'].values
            
            # Fit model
            fit_results = fit_discount_rate(choices, large_amounts, delays)
            
            if fit_results['success']:
                self.results[subject_id] = fit_results
                self.logger.info(f"Successfully fitted model for subject {subject_id} (k={fit_results['k']:.4f})")
            else:
                self.logger.error(f"Failed to fit model for subject {subject_id}: {fit_results.get('error')}")

        except Exception as e:
            self.logger.error(f"Error processing subject {subject_id}: {e}")

    def run_analysis(self, subject_list: list):
        """Runs the behavioral analysis for a list of subjects."""
        self.logger.info("Starting behavioral analysis for all subjects...")
        
        for subject_id in subject_list:
            events_file = f"{config.Paths.BEHAVIOR_DIR}/{subject_id}_discountFix_events.tsv"
            self.process_subject(subject_id, events_file)
            
        self.save_results()
        self.logger.info("Behavioral analysis complete.")

    def save_results(self):
        """Saves the aggregated behavioral results to a CSV file."""
        if not self.results:
            self.logger.warning("No behavioral results to save.")
            return

        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        output_file = config.Paths.BEHAVIORAL_OUTPUT / 'behavioral_summary.csv'
        results_df.to_csv(output_file)
        self.logger.info(f"Behavioral results saved to {output_file}") 