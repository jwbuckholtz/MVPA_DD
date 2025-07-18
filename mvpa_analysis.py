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

# Import new decoding and data loading modules
from decoding import classify, regress
from data_loader import load_behavioral_data, load_fmri_data, extract_roi_data
import config
import logging


class MVPAAnalysis:
    """
    Orchestrates the MVPA analysis pipeline.
    This class manages data loading, feature extraction, and decoding,
    relying on helper modules for the core computations.
    """
    
    def __init__(self):
        self.results = {}
        logging.basicConfig(level=config.System.LOGGING_LEVEL)
        self.logger = logging.getLogger(__name__)

    def process_subject(self, subject_id: str):
        """Processes a single subject's fMRI data for MVPA."""
        self.logger.info(f"Processing MVPA for subject {subject_id}...")
        
        try:
            behavioral_data = load_behavioral_data(subject_id)
            fmri_data = load_fmri_data(subject_id)
            
            subject_results = {}
            
            for roi_name in config.ROI.CORE_ROIS:
                roi_mask_file = config.ROI.MASK_FILES[roi_name]
                roi_data = extract_roi_data(fmri_data, roi_mask_file)
                
                # Align data: ensure same number of trials
                n_trials = min(len(behavioral_data), len(roi_data))
                behavioral_data = behavioral_data.iloc[:n_trials]
                roi_data = roi_data[:n_trials]

                roi_results = {}
                
                # Classification
                for target in config.MVPA.CLASSIFICATION_TARGETS:
                    y = behavioral_data[target].values
                    class_results = classify(roi_data, y)
                    roi_results[f"classify_{target}"] = class_results
                
                # Regression
                for target in config.MVPA.REGRESSION_TARGETS:
                    y = behavioral_data[target].values
                    reg_results = regress(roi_data, y)
                    roi_results[f"regress_{target}"] = reg_results
                
                subject_results[roi_name] = roi_results
            
            self.results[subject_id] = subject_results
            self.logger.info(f"Successfully processed MVPA for subject {subject_id}")

        except Exception as e:
            self.logger.error(f"Error processing MVPA for subject {subject_id}: {e}")

    def run_analysis(self, subject_list: list):
        """Runs the MVPA analysis for a list of subjects."""
        self.logger.info("Starting MVPA analysis for all subjects...")
        
        for subject_id in subject_list:
            self.process_subject(subject_id)
            
        self.save_results()
        self.logger.info("MVPA analysis complete.")

    def save_results(self):
        """Saves the aggregated MVPA results to a CSV file."""
        if not self.results:
            self.logger.warning("No MVPA results to save.")
            return

        # Flatten the nested results dictionary for easy saving
        flattened_results = []
        for subject_id, subject_data in self.results.items():
            for roi_name, roi_data in subject_data.items():
                for analysis_name, scores in roi_data.items():
                    flattened_results.append({
                        'subject_id': subject_id,
                        'roi': roi_name,
                        'analysis': analysis_name,
                        'score': scores.get('score'),
                        'p_value': scores.get('p_value')
                    })
        
        results_df = pd.DataFrame(flattened_results)
        output_file = config.Paths.MVPA_OUTPUT / 'mvpa_summary.csv'
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"MVPA results saved to {output_file}")


# Register the class with the factory
AnalysisFactory.register('mvpa', MVPAAnalysis)


if __name__ == "__main__":
    # Example usage
    from oak_storage_config import OAKConfig
    
    # Create MVPA analysis instance
    config = OAKConfig()
    mvpa_analysis = MVPAAnalysis()
    
    # Run analysis on a few subjects
    subjects = mvpa_analysis.get_subject_list()[:2]  # Just first 2 subjects
    mvpa_analysis.run_analysis(subjects)
    
    print("Analysis Results:")
    # The new MVPAAnalysis class does not have a get_analysis_summary method
    # and the results are not in a structured format like the old one.
    # This example will print the raw results if they were saved.
    # For now, we'll just print the subject list and a placeholder message.
    print(f"Subjects processed: {subjects}")
    print("MVPA results are saved to mvpa_summary.csv in the output directory.") 