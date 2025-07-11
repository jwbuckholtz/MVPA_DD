#!/usr/bin/env python3
"""
Caching Utilities for MVPA Pipeline
==================================

This module provides comprehensive caching capabilities for the delay discounting
MVPA pipeline using joblib Memory and content-based hashing. It caches:

1. Behavioral modeling results
2. Beta image extraction
3. MVPA decoding results
4. Geometry analysis results

Key Features:
- Content-based hashing for cache invalidation
- Versioning for analysis code changes
- Hierarchical caching (subject -> ROI -> analysis)
- Cache management and cleanup tools
- Integration with parallel processing

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import hashlib
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import json

import numpy as np
import pandas as pd
from joblib import Memory, delayed, Parallel

# Import logger utilities
from logger_utils import PipelineLogger

# Configuration
from oak_storage_config import OAKConfig

# Add memory-efficient data loading integration to existing imports
from memory_efficient_data import (
    MemoryEfficientLoader, MemoryConfig, MemoryEfficientContext,
    create_memory_efficient_loader
)


class CacheConfig:
    """Configuration for caching system"""
    
    # Cache settings
    CACHE_DIR = './cache'
    CACHE_VERBOSE = 1
    ENABLE_CACHING = True
    
    # Cache levels
    CACHE_BEHAVIORAL = True
    CACHE_BETA_EXTRACTION = True
    CACHE_MVPA_DECODING = True
    CACHE_GEOMETRY = True
    
    # Versioning
    PIPELINE_VERSION = "1.0.0"
    CACHE_VERSION = "1.0.0"
    
    # Cache management
    MAX_CACHE_SIZE_GB = 50.0  # Maximum cache size in GB
    CACHE_CLEANUP_THRESHOLD = 0.9  # Cleanup when 90% full
    CACHE_RETENTION_DAYS = 30  # Keep cache for 30 days
    
    # Performance monitoring
    ENABLE_CACHE_STATS = True
    STATS_FILE = './cache_stats.json'


class ContentHasher:
    """Generate content-based hashes for cache keys"""
    
    @staticmethod
    def hash_array(arr: np.ndarray, precision: int = 6) -> str:
        """Hash numpy array with specified precision"""
        if arr.dtype.kind in ['U', 'S', 'O']:  # String or object arrays
            return hashlib.md5(str(arr).encode()).hexdigest()[:16]
        
        # Round to specified precision for numerical stability
        arr_rounded = np.round(arr.astype(float), precision)
        return hashlib.md5(arr_rounded.tobytes()).hexdigest()[:16]
    
    @staticmethod
    def hash_dataframe(df: pd.DataFrame, precision: int = 6) -> str:
        """Hash pandas DataFrame"""
        # Include column names and dtypes
        cols_hash = hashlib.md5(str(df.columns.tolist()).encode()).hexdigest()[:8]
        dtypes_hash = hashlib.md5(str(df.dtypes.tolist()).encode()).hexdigest()[:8]
        
        # Hash values
        values_list = []
        for col in df.columns:
            if df[col].dtype.kind in ['U', 'S', 'O']:
                values_list.append(str(df[col].tolist()))
            else:
                values_list.append(ContentHasher.hash_array(df[col].values, precision))
        
        values_hash = hashlib.md5(''.join(values_list).encode()).hexdigest()[:16]
        
        return f"{cols_hash}_{dtypes_hash}_{values_hash}"
    
    @staticmethod
    def hash_dict(data: Dict[str, Any], precision: int = 6) -> str:
        """Hash dictionary recursively"""
        items = []
        for key in sorted(data.keys()):
            value = data[key]
            key_hash = hashlib.md5(str(key).encode()).hexdigest()[:8]
            
            if isinstance(value, np.ndarray):
                value_hash = ContentHasher.hash_array(value, precision)
            elif isinstance(value, pd.DataFrame):
                value_hash = ContentHasher.hash_dataframe(value, precision)
            elif isinstance(value, dict):
                value_hash = ContentHasher.hash_dict(value, precision)
            elif isinstance(value, (list, tuple)):
                value_hash = ContentHasher.hash_array(np.array(value), precision)
            else:
                value_hash = hashlib.md5(str(value).encode()).hexdigest()[:8]
            
            items.append(f"{key_hash}:{value_hash}")
        
        return hashlib.md5('|'.join(items).encode()).hexdigest()[:16]
    
    @staticmethod
    def hash_file(file_path: Union[str, Path]) -> str:
        """Hash file contents"""
        file_path = Path(file_path)
        if not file_path.exists():
            return "file_not_found"
        
        # Include file modification time and size for efficiency
        stat = file_path.stat()
        file_info = f"{stat.st_size}_{stat.st_mtime}"
        
        return hashlib.md5(file_info.encode()).hexdigest()[:16]
    
    @staticmethod
    def create_cache_key(prefix: str, subject_id: str, roi_name: str = None, 
                        **kwargs) -> str:
        """Create hierarchical cache key"""
        parts = [prefix, subject_id]
        
        if roi_name:
            parts.append(roi_name)
        
        # Add version information
        parts.append(f"v{CacheConfig.PIPELINE_VERSION}")
        
        # Add parameter hash
        if kwargs:
            param_hash = ContentHasher.hash_dict(kwargs)
            parts.append(param_hash)
        
        return "_".join(parts)


class CacheManager:
    """Manage cache storage, cleanup, and statistics"""
    
    def __init__(self, cache_dir: str = None, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache_dir = Path(cache_dir or self.config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup joblib Memory
        self.memory = Memory(
            location=str(self.cache_dir),
            verbose=self.config.CACHE_VERBOSE
        )
        
        # Initialize logger
        self.logger = PipelineLogger('caching_utils').logger
        
        # Cache statistics
        self.stats = self._load_stats()
        
        self.logger.info(f"Cache manager initialized: {self.cache_dir}")
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load cache statistics"""
        stats_file = Path(self.config.STATS_FILE)
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache stats: {e}")
        
        return {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time_saved': 0.0,
            'last_cleanup': None,
            'cache_operations': {}
        }
    
    def _save_stats(self):
        """Save cache statistics"""
        try:
            with open(self.config.STATS_FILE, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache stats: {e}")
    
    def record_hit(self, operation: str, time_saved: float):
        """Record cache hit"""
        self.stats['cache_hits'] += 1
        self.stats['total_time_saved'] += time_saved
        
        if operation not in self.stats['cache_operations']:
            self.stats['cache_operations'][operation] = {'hits': 0, 'misses': 0, 'time_saved': 0.0}
        
        self.stats['cache_operations'][operation]['hits'] += 1
        self.stats['cache_operations'][operation]['time_saved'] += time_saved
        
        self._save_stats()
    
    def record_miss(self, operation: str):
        """Record cache miss"""
        self.stats['cache_misses'] += 1
        
        if operation not in self.stats['cache_operations']:
            self.stats['cache_operations'][operation] = {'hits': 0, 'misses': 0, 'time_saved': 0.0}
        
        self.stats['cache_operations'][operation]['misses'] += 1
        
        self._save_stats()
    
    def get_cache_size(self) -> float:
        """Get current cache size in GB"""
        total_size = 0
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024**3)  # Convert to GB
    
    def cleanup_cache(self, force: bool = False) -> Dict[str, Any]:
        """Clean up old cache files"""
        current_size = self.get_cache_size()
        
        if not force and current_size < (self.config.MAX_CACHE_SIZE_GB * self.config.CACHE_CLEANUP_THRESHOLD):
            return {
                'cleanup_performed': False,
                'current_size_gb': current_size,
                'reason': 'Cache size below threshold'
            }
        
        self.logger.info(f"Starting cache cleanup. Current size: {current_size:.2f} GB")
        
        # Get all cache files with modification times
        cache_files = []
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                cache_files.append((file_path, file_path.stat().st_mtime))
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove files until we're under the threshold
        target_size = self.config.MAX_CACHE_SIZE_GB * 0.7  # Clean to 70% of max
        removed_files = 0
        removed_size = 0
        
        for file_path, mtime in cache_files:
            if current_size <= target_size:
                break
            
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                removed_files += 1
                removed_size += file_size
                current_size -= file_size / (1024**3)
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {file_path}: {e}")
        
        self.stats['last_cleanup'] = time.time()
        self._save_stats()
        
        cleanup_result = {
            'cleanup_performed': True,
            'removed_files': removed_files,
            'removed_size_gb': removed_size / (1024**3),
            'final_size_gb': current_size
        }
        
        self.logger.info(f"Cache cleanup completed: {cleanup_result}")
        return cleanup_result
    
    def clear_cache(self, pattern: str = None) -> Dict[str, Any]:
        """Clear cache files matching pattern"""
        removed_files = 0
        removed_size = 0
        
        if pattern:
            # Remove files matching pattern
            for file_path in self.cache_dir.rglob(f"*{pattern}*"):
                if file_path.is_file():
                    try:
                        removed_size += file_path.stat().st_size
                        file_path.unlink()
                        removed_files += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {file_path}: {e}")
        else:
            # Clear entire cache
            try:
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.memory = Memory(location=str(self.cache_dir), verbose=self.config.CACHE_VERBOSE)
                removed_files = "all"
            except Exception as e:
                self.logger.error(f"Failed to clear cache: {e}")
                return {'error': str(e)}
        
        return {
            'removed_files': removed_files,
            'removed_size_gb': removed_size / (1024**3) if isinstance(removed_files, int) else 'all'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.stats.copy()
        stats['current_size_gb'] = self.get_cache_size()
        stats['hit_rate'] = (
            self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        )
        return stats


class CachedMVPAProcessor:
    """MVPA processor with comprehensive caching and memory efficiency"""
    
    def __init__(self, config: OAKConfig, cache_config: CacheConfig = None, 
                 memory_config: MemoryConfig = None):
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.cache_manager = CacheManager(config=self.cache_config)
        self.logger = PipelineLogger('cached_mvpa').logger
        
        # Initialize memory-efficient loader
        self.memory_loader = create_memory_efficient_loader(config, self.memory_config)
        
        # Setup cached functions
        self._setup_cached_functions()
    
    def _setup_cached_functions(self):
        """Setup cached versions of core functions"""
        
        # Cache behavioral modeling
        if self.cache_config.CACHE_BEHAVIORAL:
            self._cached_behavioral_analysis = self.cache_manager.memory.cache(
                self._behavioral_analysis_impl,
                ignore=['self']
            )
        
        # Cache beta extraction
        if self.cache_config.CACHE_BETA_EXTRACTION:
            self._cached_beta_extraction = self.cache_manager.memory.cache(
                self._beta_extraction_impl,
                ignore=['self']
            )
        
        # Cache MVPA decoding
        if self.cache_config.CACHE_MVPA_DECODING:
            self._cached_mvpa_decoding = self.cache_manager.memory.cache(
                self._mvpa_decoding_impl,
                ignore=['self']
            )
        
        # Cache geometry analysis
        if self.cache_config.CACHE_GEOMETRY:
            self._cached_geometry_analysis = self.cache_manager.memory.cache(
                self._geometry_analysis_impl,
                ignore=['self']
            )
    
    def _behavioral_analysis_impl(self, subject_id: str, config_hash: str, 
                                 data_hash: str) -> Dict[str, Any]:
        """Implementation of behavioral analysis (for caching)"""
        from delay_discounting_mvpa_pipeline import BehavioralAnalysis
        
        behavioral_analysis = BehavioralAnalysis(self.config)
        result = behavioral_analysis.process_subject_behavior(subject_id)
        
        return result
    
    def _beta_extraction_impl(self, subject_id: str, roi_name: str, 
                             img_hash: str, behavioral_hash: str, 
                             confounds_hash: str, config_hash: str) -> Dict[str, Any]:
        """Implementation of beta extraction with memory-efficient loading"""
        from delay_discounting_mvpa_pipeline import fMRIPreprocessing, MVPAAnalysis
        from data_utils import load_behavioral_data
        
        # Load behavioral data
        behavioral_data = load_behavioral_data(subject_id, self.config, 
                                             validate=True, compute_sv=True)
        
        # Use memory-efficient fMRI loading
        fmri_data = self.memory_loader.load_fmri_memmap(subject_id)
        
        # Create maskers and extract data
        mvpa_analysis = MVPAAnalysis(self.config)
        mvpa_analysis.create_roi_maskers()
        
        # Extract trial-wise data with memory efficiency
        if roi_name in mvpa_analysis.maskers:
            masker = mvpa_analysis.maskers[roi_name]
            
            # Memory-efficient trial extraction
            X = self._extract_trial_data_memmap(fmri_data, behavioral_data, masker)
        else:
            return {'success': False, 'error': f'ROI {roi_name} not found'}
        
        return {
            'success': True,
            'neural_data': X,
            'behavioral_data': behavioral_data,
            'n_trials': X.shape[0],
            'n_voxels': X.shape[1],
            'memory_mapped': isinstance(fmri_data, self.memory_loader.__class__.__dict__.get('MemoryMappedArray', type(None)))
        }
    
    def _mvpa_decoding_impl(self, neural_hash: str, behavioral_hash: str,
                           analysis_type: str, roi_name: str, variable_name: str,
                           config_hash: str) -> Dict[str, Any]:
        """Implementation of MVPA decoding (for caching)"""
        # This will be called with the actual data by the wrapper
        # The wrapper handles the hashing
        pass  # Implementation will be in the wrapper methods
    
    def _geometry_analysis_impl(self, neural_hash: str, behavioral_hash: str,
                               roi_name: str, method: str, config_hash: str) -> Dict[str, Any]:
        """Implementation of geometry analysis (for caching)"""
        # This will be called with the actual data by the wrapper
        # The wrapper handles the hashing
        pass  # Implementation will be in the wrapper methods
    
    def process_behavioral_cached(self, subject_id: str) -> Dict[str, Any]:
        """Process behavioral data with caching"""
        start_time = time.time()
        
        # Create cache key
        config_hash = ContentHasher.hash_dict({
            'cv_folds': self.config.CV_FOLDS,
            'n_permutations': self.config.N_PERMUTATIONS,
            'version': self.cache_config.PIPELINE_VERSION
        })
        
        # Get data file hash for cache invalidation
        behavioral_file = Path(self.config.BEHAVIOR_DIR) / f"{subject_id}_discountFix_events.tsv"
        data_hash = ContentHasher.hash_file(behavioral_file)
        
        cache_key = ContentHasher.create_cache_key(
            'behavioral', subject_id, config_hash=config_hash, data_hash=data_hash
        )
        
        try:
            if self.cache_config.CACHE_BEHAVIORAL:
                result = self._cached_behavioral_analysis(subject_id, config_hash, data_hash)
                self.cache_manager.record_hit('behavioral', time.time() - start_time)
                self.logger.info(f"Behavioral cache hit: {subject_id}")
            else:
                raise ValueError("Caching disabled")
                
        except Exception:
            # Cache miss - compute result
            result = self._behavioral_analysis_impl(subject_id, config_hash, data_hash)
            self.cache_manager.record_miss('behavioral')
            self.logger.info(f"Behavioral cache miss: {subject_id}")
        
        return result
    
    def extract_betas_cached(self, subject_id: str, roi_name: str) -> Dict[str, Any]:
        """Extract beta patterns with caching"""
        start_time = time.time()
        
        # Create hashes for cache invalidation
        config_hash = ContentHasher.hash_dict({
            'tr': self.config.TR,
            'hemi_lag': self.config.HEMI_LAG,
            'version': self.cache_config.PIPELINE_VERSION
        })
        
        # Get file hashes
        fmri_file = (Path(self.config.FMRIPREP_DIR) / subject_id / 'ses-2' / 'func' / 
                    f"{subject_id}_ses-2_task-discountFix_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        behavioral_file = Path(self.config.BEHAVIOR_DIR) / f"{subject_id}_discountFix_events.tsv"
        
        img_hash = ContentHasher.hash_file(fmri_file)
        behavioral_hash = ContentHasher.hash_file(behavioral_file)
        confounds_hash = "no_confounds"  # Simplified for now
        
        try:
            if self.cache_config.CACHE_BETA_EXTRACTION:
                result = self._cached_beta_extraction(
                    subject_id, roi_name, img_hash, behavioral_hash, confounds_hash, config_hash
                )
                self.cache_manager.record_hit('beta_extraction', time.time() - start_time)
                self.logger.info(f"Beta extraction cache hit: {subject_id} {roi_name}")
            else:
                raise ValueError("Caching disabled")
                
        except Exception:
            # Cache miss - compute result
            result = self._beta_extraction_impl(
                subject_id, roi_name, img_hash, behavioral_hash, confounds_hash, config_hash
            )
            self.cache_manager.record_miss('beta_extraction')
            self.logger.info(f"Beta extraction cache miss: {subject_id} {roi_name}")
        
        return result
    
    def decode_cached(self, X: np.ndarray, y: np.ndarray, analysis_type: str,
                     roi_name: str, variable_name: str = None) -> Dict[str, Any]:
        """Run MVPA decoding with caching"""
        start_time = time.time()
        
        # Create hashes
        neural_hash = ContentHasher.hash_array(X)
        behavioral_hash = ContentHasher.hash_array(y)
        config_hash = ContentHasher.hash_dict({
            'cv_folds': self.config.CV_FOLDS,
            'n_permutations': self.config.N_PERMUTATIONS,
            'analysis_type': analysis_type,
            'version': self.cache_config.PIPELINE_VERSION
        })
        
        cache_key = ContentHasher.create_cache_key(
            f'mvpa_{analysis_type}', 'computed', roi_name,
            neural_hash=neural_hash, behavioral_hash=behavioral_hash,
            variable_name=variable_name or 'none', config_hash=config_hash
        )
        
        # Check cache
        cache_file = self.cache_manager.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists() and self.cache_config.CACHE_MVPA_DECODING:
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                self.cache_manager.record_hit('mvpa_decoding', time.time() - start_time)
                self.logger.info(f"MVPA decoding cache hit: {roi_name} {analysis_type}")
                return result
            except Exception as e:
                self.logger.warning(f"Failed to load cached result: {e}")
        
        # Cache miss - compute result
        from mvpa_utils import run_classification, run_regression
        
        if analysis_type == 'classification':
            result = run_classification(
                X, y,
                algorithm='svm',
                roi_name=roi_name,
                n_permutations=self.config.N_PERMUTATIONS
            )
        elif analysis_type == 'regression':
            result = run_regression(
                X, y,
                algorithm='ridge',
                roi_name=roi_name,
                variable_name=variable_name,
                n_permutations=self.config.N_PERMUTATIONS
            )
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Save to cache
        if self.cache_config.CACHE_MVPA_DECODING:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                self.logger.warning(f"Failed to save to cache: {e}")
        
        self.cache_manager.record_miss('mvpa_decoding')
        self.logger.info(f"MVPA decoding cache miss: {roi_name} {analysis_type}")
        
        return result

    def _extract_trial_data_memmap(self, fmri_data, behavioral_data, masker):
        """Extract trial-wise data from memory-mapped fMRI data"""
        from memory_efficient_data import MemoryMappedArray
        
        if isinstance(fmri_data, MemoryMappedArray):
            # Memory-efficient extraction for memory-mapped data
            return self._extract_from_memmap(fmri_data, behavioral_data, masker)
        else:
            # Standard extraction for regular arrays
            return self._extract_from_array(fmri_data, behavioral_data, masker)
    
    def _extract_from_memmap(self, memmap_data, behavioral_data, masker):
        """Extract trial data from memory-mapped fMRI data"""
        # Get trial onsets
        onsets = behavioral_data['onset'].values
        tr = self.config.TR
        hemi_lag = self.config.HEMI_LAG
        
        # Convert onsets to TRs
        onset_trs = (onsets / tr + hemi_lag).astype(int)
        
        # Load and apply mask
        if hasattr(masker, 'mask_img_'):
            mask = masker.mask_img_.get_fdata().astype(bool)
        else:
            # Fit masker if not already fitted
            temp_img = nib.Nifti1Image(memmap_data.array[..., 0], affine=np.eye(4))
            masker.fit(temp_img)
            mask = masker.mask_img_.get_fdata().astype(bool)
        
        # Extract ROI voxels for each trial
        roi_indices = np.where(mask.flatten())[0]
        n_voxels = len(roi_indices)
        n_trials = len(onset_trs)
        
        # Memory-efficient trial extraction
        X = np.zeros((n_trials, n_voxels), dtype=memmap_data.dtype)
        
        for i, tr in enumerate(onset_trs):
            if tr < memmap_data.shape[3]:  # Check bounds
                volume = memmap_data.array[..., tr].flatten()
                X[i, :] = volume[roi_indices]
        
        return X
    
    def _extract_from_array(self, fmri_data, behavioral_data, masker):
        """Extract trial data from regular array (fallback)"""
        # Standard nilearn-based extraction
        img = nib.Nifti1Image(fmri_data, affine=np.eye(4))
        
        # Use existing extraction method
        from mvpa_utils import extract_neural_patterns
        
        result = extract_neural_patterns(
            img, behavioral_data, masker,
            pattern_type='single_timepoint',
            tr=self.config.TR,
            hemi_lag=self.config.HEMI_LAG
        )
        
        if result['success']:
            return result['patterns']
        else:
            raise Exception(f"Pattern extraction failed: {result['error']}")
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        cache_stats = self.cache_manager.get_stats()
        memory_report = self.memory_loader.get_memory_usage_report()
        
        return {
            'cache_stats': cache_stats,
            'memory_stats': memory_report,
            'combined_efficiency': {
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'memory_mapped_files': memory_report.get('active_memmaps', 0),
                'total_memmap_size_gb': memory_report.get('memmap_usage_gb', 0),
                'system_memory_usage': memory_report.get('system_memory', {})
            }
        }
    
    def cleanup(self):
        """Cleanup cache and memory-mapped files"""
        self.cache_manager.clear_cache()
        self.memory_loader.cleanup()


def create_cached_processor(config: OAKConfig = None, 
                          cache_config: CacheConfig = None,
                          memory_config: MemoryConfig = None) -> CachedMVPAProcessor:
    """Create cached MVPA processor with memory efficiency"""
    if config is None:
        config = OAKConfig()
    
    if cache_config is None:
        cache_config = CacheConfig()
        # Set cache directory relative to output directory
        cache_config.CACHE_DIR = str(Path(config.OUTPUT_DIR) / 'cache')
        cache_config.STATS_FILE = str(Path(config.OUTPUT_DIR) / 'cache_stats.json')
    
    if memory_config is None:
        memory_config = MemoryConfig()
        # Set memory map temp directory relative to output directory
        memory_config.MEMMAP_TEMP_DIR = str(Path(config.OUTPUT_DIR) / 'memmap_temp')
    
    return CachedMVPAProcessor(config, cache_config, memory_config)


def cache_info() -> Dict[str, Any]:
    """Get cache information and statistics"""
    cache_manager = CacheManager()
    return cache_manager.get_stats()


def cleanup_cache(force: bool = False) -> Dict[str, Any]:
    """Clean up cache"""
    cache_manager = CacheManager()
    return cache_manager.cleanup_cache(force=force)


def clear_cache(pattern: str = None) -> Dict[str, Any]:
    """Clear cache files"""
    cache_manager = CacheManager()
    return cache_manager.clear_cache(pattern=pattern) 