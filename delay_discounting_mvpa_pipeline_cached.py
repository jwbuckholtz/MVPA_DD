#!/usr/bin/env python3
"""
Cached Delay Discounting MVPA Analysis Pipeline
===============================================

Enhanced version of the delay discounting MVPA pipeline with comprehensive caching
to eliminate redundant computation. This version caches:

1. Behavioral modeling results
2. Beta image extraction  
3. MVPA decoding results
4. Geometry analysis results

Key Benefits:
- Eliminates redundant computation on re-runs
- Content-based cache invalidation
- Versioning for analysis code changes
- Significant speedup for iterative analysis
- Backward compatibility with original pipeline

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
warnings.filterwarnings('ignore')

# Core scientific computing
import numpy as np
import pandas as pd
import pickle

# Configuration and utilities
from oak_storage_config import OAKConfig
from data_utils import get_complete_subjects, DataError

# Import caching utilities
from caching_utils import (
    CachedMVPAProcessor, CacheConfig, ContentHasher,
    create_cached_processor, cache_info, cleanup_cache
)

# Import existing pipeline components
from delay_discounting_mvpa_pipeline import (
    BehavioralAnalysis, fMRIPreprocessing, MVPAAnalysis, GeometryAnalysis,
    setup_directories, get_subject_list
)

# Import MVPA utilities
from mvpa_utils import update_mvpa_config

# Import logger utilities
from logger_utils import setup_pipeline_environment, create_analysis_parser


class CachedMVPAPipeline:
    """Enhanced MVPA pipeline with comprehensive caching"""
    
    def __init__(self, config: OAKConfig = None, cache_config: CacheConfig = None):
        """Initialize cached MVPA pipeline"""
        self.config = config or OAKConfig()
        self.cache_config = cache_config or CacheConfig()
        
        # Setup directories
        setup_directories(self.config)
        
        # Create cached processor
        self.cached_processor = create_cached_processor(self.config, self.cache_config)
        
        # Initialize analysis classes
        self.behavioral_analysis = BehavioralAnalysis(self.config)
        self.fmri_preprocessing = fMRIPreprocessing(self.config)
        self.mvpa_analysis = MVPAAnalysis(self.config)
        self.geometry_analysis = GeometryAnalysis(self.config)
        
        # Setup logging
        self.logger = setup_pipeline_environment('cached_mvpa_pipeline')
        
        # Configure MVPA utilities
        update_mvpa_config(
            cv_folds=self.config.CV_FOLDS,
            n_permutations=self.config.N_PERMUTATIONS,
            n_jobs=1  # Conservative for memory management
        )
        
        # Create maskers
        self.mvpa_analysis.create_roi_maskers()
        self.mvpa_analysis.create_whole_brain_masker()
        
        self.logger.info(f"Cached MVPA Pipeline initialized with {len(self.mvpa_analysis.maskers)} ROIs")
        self.logger.info(f"Cache directory: {self.cached_processor.cache_manager.cache_dir}")
    
    def process_subject_cached(self, subject_id: str) -> Dict[str, Any]:
        """Process a single subject with comprehensive caching"""
        self.logger.info(f"Processing subject {subject_id} with caching...")
        
        subject_start_time = time.time()
        
        # 1. Behavioral analysis (cached)
        self.logger.info(f"  - Processing behavioral data (cached)...")
        behavioral_result = self.cached_processor.process_behavioral_cached(subject_id)
        
        if not behavioral_result['success']:
            self.logger.error(f"  - Behavioral processing failed: {behavioral_result['error']}")
            return {
                'success': False,
                'error': behavioral_result['error'],
                'subject_id': subject_id
            }
        
        behavioral_data = behavioral_result['data']
        self.logger.info(f"  - Behavioral analysis complete. k={behavioral_result['k']:.4f}")
        
        # 2. MVPA analysis (cached per ROI)
        self.logger.info(f"  - Running MVPA analysis (cached)...")
        mvpa_results = {}
        
        for roi_name in self.mvpa_analysis.maskers.keys():
            self.logger.info(f"    - Analyzing {roi_name}...")
            
            try:
                # Extract neural data (cached)
                beta_result = self.cached_processor.extract_betas_cached(subject_id, roi_name)
                
                if not beta_result['success']:
                    self.logger.error(f"      - Beta extraction failed: {beta_result['error']}")
                    mvpa_results[roi_name] = {'error': beta_result['error']}
                    continue
                
                X = beta_result['neural_data']
                behavioral_data_extracted = beta_result['behavioral_data']
                
                # Choice decoding (cached)
                choice_result = self.cached_processor.decode_cached(
                    X, behavioral_data_extracted['choice_binary'].values,
                    'classification', roi_name
                )
                
                # Continuous variable decoding (cached)
                continuous_results = {}
                continuous_vars = ['sv_diff', 'sv_sum', 'sv_chosen', 'sv_unchosen', 'later_delay']
                
                for var_name in continuous_vars:
                    if var_name in behavioral_data_extracted.columns:
                        cont_result = self.cached_processor.decode_cached(
                            X, behavioral_data_extracted[var_name].values,
                            'regression', roi_name, var_name
                        )
                        continuous_results[var_name] = cont_result
                
                mvpa_results[roi_name] = {
                    'choice_decoding': choice_result,
                    'continuous_decoding': continuous_results,
                    'n_trials': X.shape[0],
                    'n_voxels': X.shape[1]
                }
                
            except Exception as e:
                self.logger.error(f"      - Error in {roi_name}: {e}")
                mvpa_results[roi_name] = {'error': str(e)}
        
        # 3. Geometry analysis (cached per ROI)
        self.logger.info(f"  - Running geometry analysis (cached)...")
        geometry_results = {}
        
        for roi_name in self.mvpa_analysis.maskers.keys():
            if roi_name in mvpa_results and 'error' not in mvpa_results[roi_name]:
                try:
                    # Use cached beta extraction result
                    beta_result = self.cached_processor.extract_betas_cached(subject_id, roi_name)
                    X = beta_result['neural_data']
                    behavioral_data_extracted = beta_result['behavioral_data']
                    
                    # Cache geometry analysis separately
                    geometry_result = self._process_geometry_cached(
                        X, behavioral_data_extracted, roi_name, subject_id
                    )
                    
                    geometry_results[roi_name] = geometry_result
                    
                except Exception as e:
                    self.logger.error(f"      - Error in geometry analysis for {roi_name}: {e}")
                    geometry_results[roi_name] = {'error': str(e)}
        
        subject_time = time.time() - subject_start_time
        self.logger.info(f"  - Subject {subject_id} completed in {subject_time:.2f} seconds")
        
        return {
            'success': True,
            'subject_id': subject_id,
            'behavioral': behavioral_result,
            'mvpa': mvpa_results,
            'geometry': geometry_results,
            'processing_time': subject_time
        }
    
    def _process_geometry_cached(self, X: np.ndarray, behavioral_data: pd.DataFrame,
                                roi_name: str, subject_id: str) -> Dict[str, Any]:
        """Process geometry analysis with caching"""
        # Create cache key for geometry analysis
        neural_hash = ContentHasher.hash_array(X)
        behavioral_hash = ContentHasher.hash_dataframe(behavioral_data)
        config_hash = ContentHasher.hash_dict({
            'method': 'pca',
            'n_components': 10,
            'version': self.cache_config.PIPELINE_VERSION
        })
        
        cache_key = ContentHasher.create_cache_key(
            'geometry', subject_id, roi_name,
            neural_hash=neural_hash, behavioral_hash=behavioral_hash,
            config_hash=config_hash
        )
        
        # Check cache
        cache_file = self.cached_processor.cache_manager.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists() and self.cache_config.CACHE_GEOMETRY:
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                self.logger.info(f"      - Geometry cache hit: {roi_name}")
                return result
            except Exception as e:
                self.logger.warning(f"Failed to load cached geometry result: {e}")
        
        # Cache miss - compute result
        start_time = time.time()
        
        # Dimensionality reduction
        embedding, reducer = self.geometry_analysis.dimensionality_reduction(X, method='pca')
        
        # Behavioral correlations
        behavioral_vars = {
            'sv_diff': behavioral_data['sv_diff'].values,
            'sv_sum': behavioral_data['sv_sum'].values,
            'sv_chosen': behavioral_data['sv_chosen'].values,
            'sv_unchosen': behavioral_data['sv_unchosen'].values,
            'later_delay': behavioral_data['later_delay'].values,
            'choice': behavioral_data['choice_binary'].values
        }
        
        correlations = self.geometry_analysis.behavioral_geometry_correlation(
            embedding, behavioral_vars
        )
        
        result = {
            'embedding': embedding,
            'correlations': correlations,
            'explained_variance': (
                reducer.explained_variance_ratio_ 
                if hasattr(reducer, 'explained_variance_ratio_') else None
            ),
            'n_components': embedding.shape[1]
        }
        
        # Save to cache
        if self.cache_config.CACHE_GEOMETRY:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                self.logger.warning(f"Failed to save geometry result to cache: {e}")
        
        self.logger.info(f"      - Geometry cache miss: {roi_name} ({time.time() - start_time:.2f}s)")
        
        return result
    
    def run_analysis(self, subjects: List[str] = None, 
                    show_cache_stats: bool = True) -> Dict[str, Any]:
        """Run complete analysis pipeline with caching"""
        self.logger.info("Starting Cached Delay Discounting MVPA Analysis Pipeline")
        self.logger.info("=" * 60)
        
        # Show initial cache stats
        if show_cache_stats:
            initial_stats = cache_info()
            self.logger.info(f"Initial cache stats: {initial_stats['cache_hits']} hits, "
                           f"{initial_stats['cache_misses']} misses, "
                           f"hit rate: {initial_stats['hit_rate']:.2%}")
        
        # Get subjects if not provided
        if subjects is None:
            subjects = get_subject_list(self.config)
        
        self.logger.info(f"Processing {len(subjects)} subjects")
        
        # Process each subject
        all_results = {}
        analysis_start_time = time.time()
        
        for i, subject_id in enumerate(subjects):
            self.logger.info(f"\n--- Processing subject {i+1}/{len(subjects)}: {subject_id} ---")
            
            subject_result = self.process_subject_cached(subject_id)
            all_results[subject_id] = subject_result
            
            if not subject_result['success']:
                self.logger.error(f"Subject {subject_id} failed: {subject_result['error']}")
        
        # Save results
        self.logger.info("\nSaving results...")
        results_file = Path(self.config.OUTPUT_DIR) / "all_results_cached.pkl"
        
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(all_results, f)
            self.logger.info(f"Results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
        
        # Show final cache stats
        if show_cache_stats:
            final_stats = cache_info()
            self.logger.info("\n" + "=" * 60)
            self.logger.info("CACHE STATISTICS")
            self.logger.info("=" * 60)
            self.logger.info(f"Cache hits: {final_stats['cache_hits']}")
            self.logger.info(f"Cache misses: {final_stats['cache_misses']}")
            self.logger.info(f"Hit rate: {final_stats['hit_rate']:.2%}")
            self.logger.info(f"Total time saved: {final_stats['total_time_saved']:.2f} seconds")
            self.logger.info(f"Cache size: {final_stats['current_size_gb']:.2f} GB")
            
            # Show per-operation stats
            if 'cache_operations' in final_stats:
                self.logger.info("\nPer-operation statistics:")
                for op, stats in final_stats['cache_operations'].items():
                    hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
                    self.logger.info(f"  {op}: {stats['hits']} hits, {stats['misses']} misses, "
                                   f"hit rate: {hit_rate:.2%}, time saved: {stats['time_saved']:.2f}s")
        
        total_time = time.time() - analysis_start_time
        successful_subjects = sum(1 for r in all_results.values() if r['success'])
        
        self.logger.info(f"\nAnalysis completed in {total_time:.2f} seconds")
        self.logger.info(f"Successfully processed {successful_subjects}/{len(subjects)} subjects")
        
        return {
            'results': all_results,
            'summary': {
                'total_subjects': len(subjects),
                'successful_subjects': successful_subjects,
                'total_time': total_time,
                'cache_stats': final_stats if show_cache_stats else None
            }
        }
    
    def clear_cache(self, pattern: str = None) -> Dict[str, Any]:
        """Clear cache with optional pattern matching"""
        return self.cached_processor.cache_manager.clear_cache(pattern)
    
    def cleanup_cache(self, force: bool = False) -> Dict[str, Any]:
        """Cleanup old cache files"""
        return self.cached_processor.cache_manager.cleanup_cache(force)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        return cache_info()


def main():
    """Main analysis function"""
    # Parse command line arguments
    parser = create_analysis_parser()
    parser.add_argument('--disable-cache', action='store_true',
                       help='Disable caching (run without cache)')
    parser.add_argument('--clear-cache', type=str, nargs='?', const='all',
                       help='Clear cache before running (optionally with pattern)')
    parser.add_argument('--cache-stats-only', action='store_true',
                       help='Show cache statistics and exit')
    parser.add_argument('--cleanup-cache', action='store_true',
                       help='Cleanup old cache files')
    parser.add_argument('--cache-size-gb', type=float, default=50.0,
                       help='Maximum cache size in GB (default: 50.0)')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='List of specific subjects to process')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = OAKConfig()
    
    # Setup cache configuration
    cache_config = CacheConfig()
    cache_config.CACHE_DIR = str(Path(config.OUTPUT_DIR) / 'cache')
    cache_config.STATS_FILE = str(Path(config.OUTPUT_DIR) / 'cache_stats.json')
    cache_config.MAX_CACHE_SIZE_GB = args.cache_size_gb
    
    # Disable caching if requested
    if args.disable_cache:
        cache_config.ENABLE_CACHING = False
        cache_config.CACHE_BEHAVIORAL = False
        cache_config.CACHE_BETA_EXTRACTION = False
        cache_config.CACHE_MVPA_DECODING = False
        cache_config.CACHE_GEOMETRY = False
        print("Caching disabled")
    
    # Handle cache management options
    if args.cache_stats_only:
        stats = cache_info()
        print("\nCache Statistics:")
        print("=" * 40)
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"Total time saved: {stats['total_time_saved']:.2f} seconds")
        print(f"Cache size: {stats['current_size_gb']:.2f} GB")
        return
    
    if args.cleanup_cache:
        result = cleanup_cache(force=True)
        print(f"Cache cleanup: {result}")
        return
    
    if args.clear_cache:
        pattern = args.clear_cache if args.clear_cache != 'all' else None
        result = clear_cache(pattern)
        print(f"Cache cleared: {result}")
        if args.clear_cache == 'all':
            return
    
    # Initialize pipeline
    pipeline = CachedMVPAPipeline(config, cache_config)
    
    # Run analysis
    result = pipeline.run_analysis(
        subjects=args.subjects,
        show_cache_stats=True
    )
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {config.OUTPUT_DIR}/all_results_cached.pkl")


if __name__ == "__main__":
    main() 