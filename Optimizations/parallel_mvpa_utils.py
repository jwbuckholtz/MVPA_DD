#!/usr/bin/env python3
"""
Parallel Processing Utilities for MVPA Pipeline
==============================================

This module provides enhanced parallel processing capabilities for the delay
discounting MVPA pipeline using joblib.Parallel. It enables efficient 
per-subject and per-ROI parallelization while maintaining proper logging
and error handling.

Key Features:
- Per-subject parallel processing
- Per-ROI parallel processing within subjects
- Memory-efficient chunking for large datasets
- Comprehensive logging and error handling
- Integration with existing pipeline components

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from joblib import Memory

# Import logger utilities for progress tracking
from logger_utils import PipelineLogger, setup_script_logging

# Import existing pipeline components
from oak_storage_config import OAKConfig
from data_utils import get_complete_subjects, DataError
from mvpa_utils import MVPAConfig, update_mvpa_config


class ParallelMVPAConfig:
    """Configuration for parallel MVPA processing"""
    
    # Parallel processing settings
    N_JOBS_SUBJECTS = 2          # Number of subjects to process in parallel
    N_JOBS_ROIS = 2              # Number of ROIs to process in parallel per subject
    N_JOBS_MVPA = 1              # Number of cores for MVPA algorithms (per ROI)
    
    # Memory management
    MAX_MEMORY_GB = 16           # Maximum memory per job in GB
    CHUNK_SIZE = 50              # Number of subjects per chunk
    
    # Backend settings
    BACKEND = 'loky'             # 'loky', 'threading', 'multiprocessing'
    BATCH_SIZE = 'auto'          # Batch size for parallel processing
    
    # Logging and debugging
    VERBOSE = 1                  # Verbosity level for joblib
    TEMP_FOLDER = None           # Temporary folder for joblib
    
    # Error handling
    MAX_RETRIES = 2              # Maximum retries for failed subjects
    CONTINUE_ON_ERROR = True     # Continue processing if some subjects fail
    
    # Performance monitoring
    ENABLE_PROFILING = False     # Enable performance profiling
    PROFILE_OUTPUT_DIR = './profiling_results'


class ParallelMVPAProcessor:
    """
    Main class for parallel MVPA processing
    """
    
    def __init__(self, config: OAKConfig = None, parallel_config: ParallelMVPAConfig = None):
        self.config = config or OAKConfig()
        self.parallel_config = parallel_config or ParallelMVPAConfig()
        
        # Setup logging
        self.logger = setup_script_logging(
            script_name='parallel_mvpa',
            log_level='INFO',
            log_file=f"{self.config.OUTPUT_DIR}/parallel_mvpa.log"
        )
        
        # Initialize components (will be passed to parallel workers)
        self.component_configs = {
            'behavioral_analysis': {'config': self.config},
            'fmri_preprocessing': {'config': self.config},
            'mvpa_analysis': {'config': self.config},
            'geometry_analysis': {'config': self.config}
        }
        
        # Configure MVPA utilities for parallel processing
        self.configure_mvpa_parallel()
        
        # Setup memory management
        if self.parallel_config.TEMP_FOLDER:
            self.memory = Memory(self.parallel_config.TEMP_FOLDER, verbose=0)
        else:
            self.memory = None
    
    def configure_mvpa_parallel(self):
        """Configure MVPA utilities for parallel processing"""
        update_mvpa_config(
            n_jobs=self.parallel_config.N_JOBS_MVPA,
            cv_folds=self.config.CV_FOLDS,
            n_permutations=self.config.N_PERMUTATIONS
        )
        
        self.logger.logger.info(f"Configured MVPA for parallel processing:")
        self.logger.logger.info(f"  - Subject parallelization: {self.parallel_config.N_JOBS_SUBJECTS} jobs")
        self.logger.logger.info(f"  - ROI parallelization: {self.parallel_config.N_JOBS_ROIS} jobs")
        self.logger.logger.info(f"  - MVPA algorithms: {self.parallel_config.N_JOBS_MVPA} jobs")
        self.logger.logger.info(f"  - Backend: {self.parallel_config.BACKEND}")
    
    def process_subjects_parallel(self, subjects: List[str] = None, 
                                enable_roi_parallel: bool = True) -> Dict[str, Any]:
        """
        Process multiple subjects in parallel
        
        Parameters:
        -----------
        subjects : List[str], optional
            List of subject IDs to process
        enable_roi_parallel : bool
            Whether to enable ROI-level parallelization within subjects
            
        Returns:
        --------
        Dict[str, Any] : Results for all subjects
        """
        if subjects is None:
            subjects = get_complete_subjects(self.config)
        
        self.logger.logger.info(f"Starting parallel processing of {len(subjects)} subjects")
        self.logger.logger.info(f"Parallel configuration: {self.parallel_config.N_JOBS_SUBJECTS} subject jobs")
        
        # Split subjects into chunks for memory management
        subject_chunks = self._chunk_subjects(subjects)
        all_results = {}
        
        start_time = time.time()
        
        for chunk_idx, chunk_subjects in enumerate(subject_chunks):
            self.logger.logger.info(f"Processing chunk {chunk_idx + 1}/{len(subject_chunks)} "
                                   f"({len(chunk_subjects)} subjects)")
            
            # Process chunk in parallel
            chunk_results = self._process_subject_chunk(chunk_subjects, enable_roi_parallel)
            all_results.update(chunk_results)
            
            # Log progress
            completed = sum(len(chunk) for chunk in subject_chunks[:chunk_idx + 1])
            self.logger.logger.info(f"Completed {completed}/{len(subjects)} subjects")
        
        total_time = time.time() - start_time
        self.logger.logger.info(f"Parallel processing completed in {total_time:.2f} seconds")
        
        return all_results
    
    def _chunk_subjects(self, subjects: List[str]) -> List[List[str]]:
        """Split subjects into chunks for memory management"""
        chunk_size = self.parallel_config.CHUNK_SIZE
        chunks = []
        
        for i in range(0, len(subjects), chunk_size):
            chunk = subjects[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _process_subject_chunk(self, subjects: List[str], 
                             enable_roi_parallel: bool) -> Dict[str, Any]:
        """Process a chunk of subjects in parallel"""
        
        # Create partial function with common parameters
        process_func = partial(
            process_single_subject,
            config=self.config,
            parallel_config=self.parallel_config,
            enable_roi_parallel=enable_roi_parallel
        )
        
        # Process subjects in parallel
        with parallel_backend(self.parallel_config.BACKEND):
            results = Parallel(
                n_jobs=self.parallel_config.N_JOBS_SUBJECTS,
                verbose=self.parallel_config.VERBOSE,
                batch_size=self.parallel_config.BATCH_SIZE
            )(delayed(process_func)(subject_id) for subject_id in subjects)
        
        # Convert results to dictionary
        results_dict = {}
        for subject_id, result in zip(subjects, results):
            results_dict[subject_id] = result
        
        return results_dict
    
    def process_rois_parallel(self, subject_id: str, img: Any, behavioral_data: pd.DataFrame,
                            confounds: Any, available_rois: List[str]) -> Dict[str, Any]:
        """
        Process multiple ROIs for a single subject in parallel
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        img : nibabel image
            fMRI data
        behavioral_data : pd.DataFrame
            Behavioral data for the subject
        confounds : array-like
            Confound regressors
        available_rois : List[str]
            List of available ROI names
            
        Returns:
        --------
        Dict[str, Any] : Results for all ROIs
        """
        self.logger.logger.info(f"Processing {len(available_rois)} ROIs in parallel for {subject_id}")
        
        # Create partial function with common parameters
        process_func = partial(
            process_single_roi,
            subject_id=subject_id,
            img=img,
            behavioral_data=behavioral_data,
            confounds=confounds,
            config=self.config,
            parallel_config=self.parallel_config
        )
        
        # Process ROIs in parallel
        with parallel_backend(self.parallel_config.BACKEND):
            results = Parallel(
                n_jobs=self.parallel_config.N_JOBS_ROIS,
                verbose=self.parallel_config.VERBOSE,
                batch_size=self.parallel_config.BATCH_SIZE
            )(delayed(process_func)(roi_name) for roi_name in available_rois)
        
        # Convert results to dictionary
        results_dict = {}
        for roi_name, result in zip(available_rois, results):
            results_dict[roi_name] = result
        
        return results_dict
    
    def get_optimal_n_jobs(self, task_type: str = 'subject') -> int:
        """
        Get optimal number of jobs based on available resources
        
        Parameters:
        -----------
        task_type : str
            Type of task ('subject', 'roi', 'mvpa')
            
        Returns:
        --------
        int : Optimal number of jobs
        """
        # Get available resources
        cpu_count = mp.cpu_count()
        available_memory_gb = self._get_available_memory_gb()
        
        if task_type == 'subject':
            # Subject-level parallelization
            max_jobs_cpu = max(1, cpu_count // 2)  # Conservative CPU usage
            max_jobs_memory = max(1, int(available_memory_gb // self.parallel_config.MAX_MEMORY_GB))
            optimal_jobs = min(max_jobs_cpu, max_jobs_memory, self.parallel_config.N_JOBS_SUBJECTS)
            
        elif task_type == 'roi':
            # ROI-level parallelization
            max_jobs_cpu = max(1, cpu_count // 4)  # More conservative for nested parallelization
            max_jobs_memory = max(1, int(available_memory_gb // (self.parallel_config.MAX_MEMORY_GB // 2)))
            optimal_jobs = min(max_jobs_cpu, max_jobs_memory, self.parallel_config.N_JOBS_ROIS)
            
        elif task_type == 'mvpa':
            # MVPA algorithm parallelization
            optimal_jobs = min(cpu_count, self.parallel_config.N_JOBS_MVPA)
            
        else:
            optimal_jobs = 1
        
        return optimal_jobs
    
    def _get_available_memory_gb(self) -> float:
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # Fallback estimate
            return 16.0  # Assume 16GB available


def process_single_subject(subject_id: str, config: OAKConfig, 
                         parallel_config: ParallelMVPAConfig,
                         enable_roi_parallel: bool = True) -> Dict[str, Any]:
    """
    Process a single subject (designed for parallel execution)
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    config : OAKConfig
        Main configuration
    parallel_config : ParallelMVPAConfig
        Parallel processing configuration
    enable_roi_parallel : bool
        Whether to enable ROI-level parallelization
        
    Returns:
    --------
    Dict[str, Any] : Subject results
    """
    try:
        # Import here to avoid issues with parallel processing
        from delay_discounting_mvpa_pipeline import (
            BehavioralAnalysis, fMRIPreprocessing, MVPAAnalysis
        )
        from geometry_analysis import GeometryAnalysis
        
        # Initialize analysis classes
        behavioral_analysis = BehavioralAnalysis(config)
        fmri_preprocessing = fMRIPreprocessing(config)
        mvpa_analysis = MVPAAnalysis(config)
        geometry_analysis = GeometryAnalysis(config)
        
        # Create maskers
        available_rois = mvpa_analysis.create_roi_maskers()
        
        # Process behavioral data
        behavior_result = behavioral_analysis.process_subject_behavior(subject_id)
        if not behavior_result['success']:
            return {
                'success': False,
                'error': f"Behavioral processing failed: {behavior_result['error']}",
                'subject_id': subject_id
            }
        
        # Load fMRI data
        fmri_result = fmri_preprocessing.load_subject_fmri(subject_id)
        if not fmri_result['success']:
            return {
                'success': False,
                'error': f"fMRI loading failed: {fmri_result['error']}",
                'subject_id': subject_id
            }
        
        # Extract common data
        behavioral_data = behavior_result['data']
        img = fmri_result['img']
        confounds = fmri_result['confounds']
        
        # Process ROIs (parallel or serial)
        if enable_roi_parallel and len(available_rois) > 1:
            # Use parallel ROI processing
            processor = ParallelMVPAProcessor(config, parallel_config)
            mvpa_results = processor.process_rois_parallel(
                subject_id, img, behavioral_data, confounds, available_rois
            )
        else:
            # Use serial ROI processing
            mvpa_results = {}
            for roi_name in available_rois:
                roi_result = process_single_roi(
                    roi_name, subject_id, img, behavioral_data, confounds,
                    config, parallel_config
                )
                mvpa_results[roi_name] = roi_result
        
        # Geometry analysis (if requested)
        geometry_results = {}
        if hasattr(config, 'ENABLE_GEOMETRY_ANALYSIS') and config.ENABLE_GEOMETRY_ANALYSIS:
            for roi_name in available_rois:
                if roi_name in mvpa_results and mvpa_results[roi_name].get('success', False):
                    try:
                        geometry_result = process_roi_geometry(
                            roi_name, subject_id, img, behavioral_data, confounds,
                            config, geometry_analysis
                        )
                        geometry_results[roi_name] = geometry_result
                    except Exception as e:
                        geometry_results[roi_name] = {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'subject_id': subject_id,
            'behavioral_data': behavioral_data,
            'mvpa_results': mvpa_results,
            'geometry_results': geometry_results,
            'processing_time': time.time()
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'subject_id': subject_id
        }


def process_single_roi(roi_name: str, subject_id: str, img: Any, 
                     behavioral_data: pd.DataFrame, confounds: Any,
                     config: OAKConfig, parallel_config: ParallelMVPAConfig) -> Dict[str, Any]:
    """
    Process a single ROI for a subject (designed for parallel execution)
    
    Parameters:
    -----------
    roi_name : str
        ROI name
    subject_id : str
        Subject identifier
    img : nibabel image
        fMRI data
    behavioral_data : pd.DataFrame
        Behavioral data
    confounds : array-like
        Confound regressors
    config : OAKConfig
        Main configuration
    parallel_config : ParallelMVPAConfig
        Parallel processing configuration
        
    Returns:
    --------
    Dict[str, Any] : ROI results
    """
    try:
        # Import here to avoid issues with parallel processing
        from delay_discounting_mvpa_pipeline import MVPAAnalysis
        from mvpa_utils import run_classification, run_regression
        
        # Initialize MVPA analysis
        mvpa_analysis = MVPAAnalysis(config)
        mvpa_analysis.create_roi_maskers()
        
        # Extract trial-wise data
        X = mvpa_analysis.extract_trial_data(img, behavioral_data, roi_name, confounds)
        
        # Run MVPA analyses
        results = {
            'success': True,
            'roi_name': roi_name,
            'subject_id': subject_id,
            'n_trials': X.shape[0],
            'n_voxels': X.shape[1]
        }
        
        # Choice decoding
        if 'choice_binary' in behavioral_data.columns:
            choice_result = run_classification(
                X, behavioral_data['choice_binary'].values,
                algorithm='svm',
                roi_name=roi_name,
                n_permutations=config.N_PERMUTATIONS
            )
            results['choice_decoding'] = choice_result
        
        # Continuous variable decoding
        continuous_vars = ['sv_diff', 'sv_sum', 'sv_chosen', 'sv_unchosen', 'later_delay']
        continuous_results = {}
        
        for var_name in continuous_vars:
            if var_name in behavioral_data.columns:
                cont_result = run_regression(
                    X, behavioral_data[var_name].values,
                    algorithm='ridge',
                    roi_name=roi_name,
                    variable_name=var_name,
                    n_permutations=config.N_PERMUTATIONS
                )
                continuous_results[var_name] = cont_result
        
        results['continuous_decoding'] = continuous_results
        
        return results
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'roi_name': roi_name,
            'subject_id': subject_id
        }


def process_roi_geometry(roi_name: str, subject_id: str, img: Any,
                       behavioral_data: pd.DataFrame, confounds: Any,
                       config: OAKConfig, geometry_analysis: Any) -> Dict[str, Any]:
    """
    Process geometry analysis for a single ROI
    
    Parameters:
    -----------
    roi_name : str
        ROI name
    subject_id : str
        Subject identifier
    img : nibabel image
        fMRI data
    behavioral_data : pd.DataFrame
        Behavioral data
    confounds : array-like
        Confound regressors
    config : OAKConfig
        Main configuration
    geometry_analysis : GeometryAnalysis
        Geometry analysis instance
        
    Returns:
    --------
    Dict[str, Any] : Geometry results
    """
    try:
        # Import here to avoid issues with parallel processing
        from delay_discounting_mvpa_pipeline import MVPAAnalysis
        
        # Initialize MVPA analysis
        mvpa_analysis = MVPAAnalysis(config)
        mvpa_analysis.create_roi_maskers()
        
        # Extract neural patterns
        X = mvpa_analysis.extract_trial_data(img, behavioral_data, roi_name, confounds)
        
        # Dimensionality reduction
        embedding, reducer = geometry_analysis.dimensionality_reduction(X, method='pca')
        
        # Behavioral correlations
        behavioral_vars = {
            'sv_diff': behavioral_data['sv_diff'].values,
            'sv_sum': behavioral_data['sv_sum'].values,
            'sv_chosen': behavioral_data['sv_chosen'].values,
            'sv_unchosen': behavioral_data['sv_unchosen'].values,
            'later_delay': behavioral_data['later_delay'].values,
            'choice': behavioral_data['choice_binary'].values
        }
        
        correlations = geometry_analysis.behavioral_geometry_correlation(
            embedding, behavioral_vars
        )
        
        return {
            'success': True,
            'roi_name': roi_name,
            'subject_id': subject_id,
            'embedding': embedding,
            'correlations': correlations,
            'reducer': reducer
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'roi_name': roi_name,
            'subject_id': subject_id
        }


def optimize_parallel_config(config: OAKConfig, subjects: List[str]) -> ParallelMVPAConfig:
    """
    Optimize parallel configuration based on available resources and dataset size
    
    Parameters:
    -----------
    config : OAKConfig
        Main configuration
    subjects : List[str]
        List of subjects to process
        
    Returns:
    --------
    ParallelMVPAConfig : Optimized parallel configuration
    """
    # Get system resources
    cpu_count = mp.cpu_count()
    
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        total_memory_gb = 16.0  # Fallback
        available_memory_gb = 12.0
    
    # Create optimized configuration
    parallel_config = ParallelMVPAConfig()
    
    # Optimize based on dataset size
    n_subjects = len(subjects)
    n_rois = len(config.ROI_MASKS)
    
    if n_subjects <= 10:
        # Small dataset: maximize ROI parallelization
        parallel_config.N_JOBS_SUBJECTS = min(2, cpu_count // 2)
        parallel_config.N_JOBS_ROIS = min(n_rois, cpu_count // 2)
    elif n_subjects <= 50:
        # Medium dataset: balance subject and ROI parallelization
        parallel_config.N_JOBS_SUBJECTS = min(4, cpu_count // 2)
        parallel_config.N_JOBS_ROIS = min(n_rois, 2)
    else:
        # Large dataset: maximize subject parallelization
        parallel_config.N_JOBS_SUBJECTS = min(8, cpu_count)
        parallel_config.N_JOBS_ROIS = 1  # Serial ROI processing
    
    # Memory-based optimization
    memory_per_job = available_memory_gb / (parallel_config.N_JOBS_SUBJECTS * parallel_config.N_JOBS_ROIS)
    if memory_per_job < 2.0:  # Less than 2GB per job
        # Reduce parallelization
        parallel_config.N_JOBS_SUBJECTS = max(1, parallel_config.N_JOBS_SUBJECTS // 2)
        parallel_config.N_JOBS_ROIS = max(1, parallel_config.N_JOBS_ROIS // 2)
    
    # MVPA algorithm parallelization
    parallel_config.N_JOBS_MVPA = 1  # Conservative for nested parallelization
    
    # Chunk size optimization
    parallel_config.CHUNK_SIZE = min(50, max(10, n_subjects // 4))
    
    return parallel_config


def create_parallel_pipeline(config: OAKConfig = None, 
                           parallel_config: ParallelMVPAConfig = None,
                           subjects: List[str] = None) -> ParallelMVPAProcessor:
    """
    Create optimized parallel MVPA pipeline
    
    Parameters:
    -----------
    config : OAKConfig, optional
        Main configuration
    parallel_config : ParallelMVPAConfig, optional
        Parallel processing configuration
    subjects : List[str], optional
        List of subjects to process
        
    Returns:
    --------
    ParallelMVPAProcessor : Configured parallel processor
    """
    if config is None:
        config = OAKConfig()
    
    if subjects is None:
        subjects = get_complete_subjects(config)
    
    if parallel_config is None:
        parallel_config = optimize_parallel_config(config, subjects)
    
    # Create processor
    processor = ParallelMVPAProcessor(config, parallel_config)
    
    return processor 