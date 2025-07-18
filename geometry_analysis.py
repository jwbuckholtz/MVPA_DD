#!/usr/bin/env python3
"""
Geometry Analysis Module
=======================

A refactored script for neural geometry analysis.
"""

import pandas as pd
from data_loader import load_behavioral_data, load_fmri_data, extract_roi_data
from geometry import compute_rdm, reduce_dimensionality, compare_conditions
import config
import logging

class GeometryAnalysis:
    """
    Orchestrates the neural geometry analysis pipeline.
    """
    
    def __init__(self):
        self.results = {}
        logging.basicConfig(level=config.System.LOGGING_LEVEL)
        self.logger = logging.getLogger(__name__)

    def process_subject(self, subject_id: str):
        """Processes a single subject's fMRI data for geometry analysis."""
        self.logger.info(f"Processing geometry analysis for subject {subject_id}...")
        
        try:
            behavioral_data = load_behavioral_data(subject_id)
            fmri_data = load_fmri_data(subject_id)
            
            subject_results = {}
            
            for roi_name in config.ROI.CORE_ROIS:
                roi_mask_file = config.ROI.MASK_FILES[roi_name]
                roi_data = extract_roi_data(fmri_data, roi_mask_file)
                
                n_trials = min(len(behavioral_data), len(roi_data))
                behavioral_data = behavioral_data.iloc[:n_trials]
                roi_data = roi_data[:n_trials]
                
                embedding, _ = reduce_dimensionality(roi_data, n_components=config.Geometry.N_COMPONENTS_PCA)
                
                roi_results = {}
                for comparison in config.Geometry.COMPARISONS:
                    if comparison in behavioral_data.columns:
                        condition_labels = behavioral_data[comparison].values
                        comp_results = compare_conditions(embedding, condition_labels)
                        roi_results[comparison] = comp_results
                
                subject_results[roi_name] = roi_results
            
            self.results[subject_id] = subject_results
            self.logger.info(f"Successfully processed geometry analysis for subject {subject_id}")

        except Exception as e:
            self.logger.error(f"Error processing geometry for subject {subject_id}: {e}")

    def run_analysis(self, subject_list: list):
        """Runs the geometry analysis for a list of subjects."""
        self.logger.info("Starting geometry analysis for all subjects...")
        
        for subject_id in subject_list:
            self.process_subject(subject_id)
            
        self.save_results()
        self.logger.info("Geometry analysis complete.")

    def save_results(self):
        """Saves the aggregated geometry analysis results to a CSV file."""
        if not self.results:
            self.logger.warning("No geometry results to save.")
            return

        flattened_results = []
        for subject_id, subject_data in self.results.items():
            for roi_name, roi_data in subject_data.items():
                for comparison, scores in roi_data.items():
                    flattened_results.append({
                        'subject_id': subject_id,
                        'roi': roi_name,
                        'comparison': comparison,
                        'distance': scores.get('observed_distance'),
                        'p_value': scores.get('p_value')
                    })
        
        results_df = pd.DataFrame(flattened_results)
        output_file = config.Paths.GEOMETRY_OUTPUT / 'geometry_summary.csv'
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Geometry results saved to {output_file}") 