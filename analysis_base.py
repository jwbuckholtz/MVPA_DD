#!/usr/bin/env python3
"""
Base Analysis Class for MVPA Pipeline
====================================

This module provides a base class that abstracts common patterns across all analysis
types (behavioral, MVPA, geometry) to eliminate code duplication and provide
consistent interfaces.

Common patterns abstracted:
- Configuration management
- Data loading (behavioral and fMRI)
- Result storage and retrieval
- Error handling and logging
- Progress tracking
- Resource management

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pickle
import json

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker

# Import utilities
from oak_storage_config import OAKConfig
from data_utils import (
    load_behavioral_data, load_fmri_data, load_confounds,
    extract_roi_timeseries, get_complete_subjects, check_mask_exists,
    load_mask, DataError, SubjectManager
)

# Import logging utilities
try:
    from logger_utils import setup_script_logging, PipelineLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    warnings.warn("logger_utils not available. Using basic logging.")

# Import memory-efficient utilities if available
try:
    from memory_efficient_data import (
        MemoryConfig, create_memory_efficient_loader, 
        MemoryEfficientDataLoader
    )
    MEMORY_EFFICIENT_AVAILABLE = True
except ImportError:
    MEMORY_EFFICIENT_AVAILABLE = False
    MemoryConfig = None
    MemoryEfficientDataLoader = None


class AnalysisError(Exception):
    """Base exception for analysis-related errors"""
    pass


class BaseAnalysis(ABC):
    """
    Abstract base class for all analysis types
    
    This class provides common functionality and enforces consistent interfaces
    across different analysis types (behavioral, MVPA, geometry).
    """
    
    def __init__(self, config: OAKConfig = None, 
                 name: str = None,
                 enable_memory_efficient: bool = False,
                 memory_config: MemoryConfig = None):
        """
        Initialize base analysis class
        
        Parameters:
        -----------
        config : OAKConfig, optional
            Configuration object
        name : str, optional
            Name of the analysis (used for logging)
        enable_memory_efficient : bool
            Whether to enable memory-efficient data loading
        memory_config : MemoryConfig, optional
            Memory configuration for efficient loading
        """
        self.config = config or OAKConfig()
        self.name = name or self.__class__.__name__
        
        # Memory efficiency settings
        self.enable_memory_efficient = enable_memory_efficient
        self.memory_config = memory_config or (MemoryConfig() if MEMORY_EFFICIENT_AVAILABLE else None)
        
        # Initialize memory loader if requested
        if self.enable_memory_efficient and MEMORY_EFFICIENT_AVAILABLE:
            self.memory_loader = create_memory_efficient_loader(
                self.config, self.memory_config
            )
        else:
            self.memory_loader = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize data storage
        self.results = {}
        self.metadata = {
            'analysis_type': self.name,
            'config': self.config.__dict__.copy() if hasattr(self.config, '__dict__') else {},
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'memory_efficient': self.enable_memory_efficient
        }
        
        # Initialize resource tracking
        self.processing_stats = {
            'subjects_processed': 0,
            'subjects_failed': 0,
            'processing_times': [],
            'memory_usage': []
        }
        
        # Cache for loaded data
        self._data_cache = {}
        
        # Initialize analysis-specific components
        self._initialize_analysis_components()
        
        self.logger.info(f"Initialized {self.name} analysis")
        if self.enable_memory_efficient:
            self.logger.info("Memory-efficient loading enabled")
    
    def _setup_logging(self):
        """Setup logging for the analysis"""
        if LOGGER_AVAILABLE:
            logger = setup_script_logging(
                script_name=self.name.lower(),
                log_level='INFO',
                log_file=f"{self.config.OUTPUT_DIR}/{self.name.lower()}.log"
            )
            return logger.logger
        else:
            # Fallback to basic logging
            import logging
            logger = logging.getLogger(self.name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            return logger
    
    def _initialize_analysis_components(self):
        """Initialize analysis-specific components (override in subclasses)"""
        pass
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def process_subject(self, subject_id: str, **kwargs) -> Dict[str, Any]:
        """
        Process a single subject
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        **kwargs : dict
            Additional parameters specific to analysis type
            
        Returns:
        --------
        Dict[str, Any] : Processing results
        """
        pass
    
    @abstractmethod
    def run_analysis(self, subjects: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline
        
        Parameters:
        -----------
        subjects : List[str], optional
            List of subject IDs to process
        **kwargs : dict
            Additional parameters specific to analysis type
            
        Returns:
        --------
        Dict[str, Any] : Complete analysis results
        """
        pass
    
    # Common data loading methods
    def load_behavioral_data(self, subject_id: str, **kwargs) -> pd.DataFrame:
        """
        Load behavioral data for a subject
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        **kwargs : dict
            Additional parameters for data loading
            
        Returns:
        --------
        pd.DataFrame : Behavioral data
        """
        cache_key = f"behavioral_{subject_id}"
        
        if cache_key in self._data_cache:
            self.logger.debug(f"Using cached behavioral data for {subject_id}")
            return self._data_cache[cache_key]
        
        try:
            behavioral_data = load_behavioral_data(
                subject_id, self.config, 
                validate=kwargs.get('validate', True),
                compute_sv=kwargs.get('compute_sv', True)
            )
            
            # Cache the data
            self._data_cache[cache_key] = behavioral_data
            
            self.logger.info(f"Loaded behavioral data for {subject_id}: {len(behavioral_data)} trials")
            return behavioral_data
            
        except DataError as e:
            self.logger.error(f"Failed to load behavioral data for {subject_id}: {e}")
            raise AnalysisError(f"Behavioral data loading failed: {e}")
    
    def load_fmri_data(self, subject_id: str, **kwargs) -> Tuple[nib.Nifti1Image, Optional[np.ndarray]]:
        """
        Load fMRI data for a subject
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        **kwargs : dict
            Additional parameters for data loading
            
        Returns:
        --------
        Tuple[nib.Nifti1Image, Optional[np.ndarray]] : fMRI image and confounds
        """
        cache_key = f"fmri_{subject_id}"
        
        if cache_key in self._data_cache:
            self.logger.debug(f"Using cached fMRI data for {subject_id}")
            return self._data_cache[cache_key]
        
        try:
            if self.enable_memory_efficient and self.memory_loader:
                # Use memory-efficient loading
                fmri_data = self.memory_loader.load_fmri_data(
                    subject_id, 
                    force_memmap=kwargs.get('force_memmap', False)
                )
                img = fmri_data['img']
                confounds = fmri_data.get('confounds')
            else:
                # Standard loading
                img = load_fmri_data(subject_id, self.config)
                confounds = load_confounds(subject_id, self.config)
            
            # Cache the data
            self._data_cache[cache_key] = (img, confounds)
            
            self.logger.info(f"Loaded fMRI data for {subject_id}: {img.shape}")
            return img, confounds
            
        except Exception as e:
            self.logger.error(f"Failed to load fMRI data for {subject_id}: {e}")
            raise AnalysisError(f"fMRI data loading failed: {e}")
    
    def create_maskers(self, roi_names: List[str] = None) -> Dict[str, NiftiMasker]:
        """
        Create NiftiMasker objects for specified ROIs
        
        Parameters:
        -----------
        roi_names : List[str], optional
            List of ROI names to create maskers for
            
        Returns:
        --------
        Dict[str, NiftiMasker] : Dictionary of maskers
        """
        if roi_names is None:
            roi_names = list(self.config.ROI_MASKS.keys())
        
        maskers = {}
        
        for roi_name in roi_names:
            if roi_name not in self.config.ROI_MASKS:
                self.logger.warning(f"ROI {roi_name} not found in configuration")
                continue
                
            mask_path = self.config.ROI_MASKS[roi_name]
            
            if check_mask_exists(mask_path):
                try:
                    # Validate mask before using
                    mask_img = load_mask(mask_path, validate=True)
                    
                    maskers[roi_name] = NiftiMasker(
                        mask_img=mask_path,
                        standardize=True,
                        detrend=True,
                        high_pass=0.01,
                        t_r=self.config.TR,
                        memory='nilearn_cache',
                        memory_level=1
                    )
                    
                    self.logger.info(f"Created masker for {roi_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create masker for {roi_name}: {e}")
            else:
                self.logger.warning(f"ROI mask not found: {roi_name} ({mask_path})")
        
        self.logger.info(f"Created {len(maskers)} maskers: {', '.join(maskers.keys())}")
        return maskers
    
    # Common result handling methods
    def save_results(self, output_path: str = None, include_metadata: bool = True) -> str:
        """
        Save analysis results to file
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save results to
        include_metadata : bool
            Whether to include metadata in saved results
            
        Returns:
        --------
        str : Path to saved file
        """
        if output_path is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = f"{self.config.OUTPUT_DIR}/{self.name.lower()}_results_{timestamp}.pkl"
        
        # Prepare data to save
        save_data = {
            'results': self.results,
            'processing_stats': self.processing_stats
        }
        
        if include_metadata:
            save_data['metadata'] = self.metadata
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Saved results to {output_path}")
        return output_path
    
    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load analysis results from file
        
        Parameters:
        -----------
        input_path : str
            Path to load results from
            
        Returns:
        --------
        Dict[str, Any] : Loaded results
        """
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update instance variables
            self.results = data.get('results', {})
            self.processing_stats = data.get('processing_stats', {})
            
            if 'metadata' in data:
                self.metadata.update(data['metadata'])
            
            self.logger.info(f"Loaded results from {input_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load results from {input_path}: {e}")
            raise AnalysisError(f"Result loading failed: {e}")
    
    def export_results_summary(self, output_path: str = None) -> str:
        """
        Export a summary of results in human-readable format
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save summary to
            
        Returns:
        --------
        str : Path to saved summary
        """
        if output_path is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = f"{self.config.OUTPUT_DIR}/{self.name.lower()}_summary_{timestamp}.txt"
        
        # Create summary
        summary_lines = [
            f"Analysis Summary: {self.name}",
            "=" * 50,
            f"Created: {self.metadata.get('created_at', 'Unknown')}",
            f"Subjects processed: {self.processing_stats.get('subjects_processed', 0)}",
            f"Subjects failed: {self.processing_stats.get('subjects_failed', 0)}",
            f"Memory efficient: {self.enable_memory_efficient}",
            "",
            "Processing Times:",
            f"  Mean: {np.mean(self.processing_stats.get('processing_times', [0])):.2f}s",
            f"  Std: {np.std(self.processing_stats.get('processing_times', [0])):.2f}s",
            f"  Total: {np.sum(self.processing_stats.get('processing_times', [0])):.2f}s",
            "",
            "Results Overview:",
            f"  Number of result keys: {len(self.results)}",
            f"  Result keys: {', '.join(self.results.keys())}",
            "",
        ]
        
        # Add analysis-specific summary
        analysis_summary = self.get_analysis_summary()
        if analysis_summary:
            summary_lines.extend(["Analysis-Specific Summary:", analysis_summary])
        
        # Write summary
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        self.logger.info(f"Exported summary to {output_path}")
        return output_path
    
    def get_analysis_summary(self) -> str:
        """
        Get analysis-specific summary (override in subclasses)
        
        Returns:
        --------
        str : Analysis-specific summary
        """
        return ""
    
    # Common utility methods
    def get_subject_list(self, subjects: List[str] = None) -> List[str]:
        """
        Get list of subjects to process
        
        Parameters:
        -----------
        subjects : List[str], optional
            Specific subjects to process
            
        Returns:
        --------
        List[str] : List of subject IDs
        """
        if subjects is None:
            subjects = get_complete_subjects(self.config)
        
        self.logger.info(f"Processing {len(subjects)} subjects")
        return subjects
    
    def update_processing_stats(self, subject_id: str, 
                              processing_time: float = None,
                              success: bool = True,
                              memory_usage: float = None):
        """
        Update processing statistics
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        processing_time : float, optional
            Processing time in seconds
        success : bool
            Whether processing was successful
        memory_usage : float, optional
            Memory usage in MB
        """
        if success:
            self.processing_stats['subjects_processed'] += 1
        else:
            self.processing_stats['subjects_failed'] += 1
        
        if processing_time is not None:
            self.processing_stats['processing_times'].append(processing_time)
        
        if memory_usage is not None:
            self.processing_stats['memory_usage'].append(memory_usage)
    
    def clear_cache(self):
        """Clear data cache to free memory"""
        self._data_cache.clear()
        self.logger.info("Cleared data cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data
        
        Returns:
        --------
        Dict[str, Any] : Cache information
        """
        return {
            'cache_size': len(self._data_cache),
            'cached_keys': list(self._data_cache.keys()),
            'memory_usage_mb': sum(
                sys.getsizeof(v) for v in self._data_cache.values()
            ) / (1024 * 1024)
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', subjects_processed={self.processing_stats['subjects_processed']})"
    
    def __str__(self) -> str:
        return f"{self.name} Analysis - {self.processing_stats['subjects_processed']} subjects processed"


class AnalysisFactory:
    """Factory class for creating analysis instances"""
    
    _analysis_registry = {}
    
    @classmethod
    def register(cls, name: str, analysis_class: type):
        """Register an analysis class"""
        cls._analysis_registry[name] = analysis_class
    
    @classmethod
    def create(cls, analysis_type: str, **kwargs) -> BaseAnalysis:
        """
        Create an analysis instance
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis to create
        **kwargs : dict
            Arguments to pass to analysis constructor
            
        Returns:
        --------
        BaseAnalysis : Analysis instance
        """
        if analysis_type not in cls._analysis_registry:
            raise ValueError(f"Unknown analysis type: {analysis_type}. "
                           f"Available types: {list(cls._analysis_registry.keys())}")
        
        analysis_class = cls._analysis_registry[analysis_type]
        return analysis_class(**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available analysis types"""
        return list(cls._analysis_registry.keys())


# Utility functions for backwards compatibility
def create_analysis(analysis_type: str, **kwargs) -> BaseAnalysis:
    """
    Create an analysis instance (convenience function)
    
    Parameters:
    -----------
    analysis_type : str
        Type of analysis ('behavioral', 'mvpa', 'geometry')
    **kwargs : dict
        Arguments to pass to analysis constructor
        
    Returns:
    --------
    BaseAnalysis : Analysis instance
    """
    return AnalysisFactory.create(analysis_type, **kwargs)


def setup_analysis_environment(config: OAKConfig = None) -> Dict[str, Any]:
    """
    Setup common analysis environment
    
    Parameters:
    -----------
    config : OAKConfig, optional
        Configuration object
        
    Returns:
    --------
    Dict[str, Any] : Environment information
    """
    config = config or OAKConfig()
    
    # Ensure output directories exist
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check for required data
    subjects = get_complete_subjects(config)
    
    return {
        'config': config,
        'subjects': subjects,
        'n_subjects': len(subjects),
        'memory_efficient_available': MEMORY_EFFICIENT_AVAILABLE,
        'logger_available': LOGGER_AVAILABLE
    }


if __name__ == "__main__":
    # Example usage
    config = OAKConfig()
    env = setup_analysis_environment(config)
    
    print("Analysis Environment Setup:")
    print(f"  Available subjects: {env['n_subjects']}")
    print(f"  Memory efficient available: {env['memory_efficient_available']}")
    print(f"  Logger available: {env['logger_available']}")
    print(f"  Available analysis types: {AnalysisFactory.list_available()}") 