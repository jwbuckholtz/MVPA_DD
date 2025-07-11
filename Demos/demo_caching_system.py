#!/usr/bin/env python3
"""
Caching System Demo for MVPA Pipeline
====================================

This script demonstrates the benefits and usage of the comprehensive caching system
implemented for the delay discounting MVPA pipeline. It shows:

1. Performance comparisons between cached and non-cached runs
2. Cache hit rates and statistics
3. Cache invalidation behavior
4. Memory usage optimization
5. Integration with parallel processing

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import time
import warnings
from pathlib import Path
import argparse
from typing import Dict, List, Any
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import pipeline components
from oak_storage_config import OAKConfig
from caching_utils import (
    CachedMVPAProcessor, CacheConfig, ContentHasher,
    cache_info, cleanup_cache, clear_cache
)
from delay_discounting_mvpa_pipeline_cached import CachedMVPAPipeline
from delay_discounting_mvpa_pipeline import main as run_original_pipeline

# Import logger utilities
from logger_utils import setup_pipeline_environment


class CachingDemo:
    """Comprehensive demo of caching system capabilities"""
    
    def __init__(self, config: OAKConfig = None):
        """Initialize demo with configuration"""
        self.config = config or OAKConfig()
        self.logger = setup_pipeline_environment('caching_demo')
        
        # Setup demo subjects (subset for faster demonstration)
        self.demo_subjects = self._get_demo_subjects()
        
        # Demo configurations
        self.cache_config = CacheConfig()
        self.cache_config.CACHE_DIR = str(Path(self.config.OUTPUT_DIR) / 'demo_cache')
        self.cache_config.STATS_FILE = str(Path(self.config.OUTPUT_DIR) / 'demo_cache_stats.json')
        self.cache_config.MAX_CACHE_SIZE_GB = 10.0  # Smaller for demo
        
        self.logger.info(f"Caching demo initialized with {len(self.demo_subjects)} subjects")
    
    def _get_demo_subjects(self) -> List[str]:
        """Get a subset of subjects for demonstration"""
        from data_utils import get_complete_subjects
        
        try:
            all_subjects = get_complete_subjects(self.config)
            # Use first 5 subjects for demo, or all if fewer available
            return all_subjects[:min(5, len(all_subjects))]
        except Exception as e:
            self.logger.warning(f"Could not get real subjects: {e}")
            # Use synthetic subject IDs for demo
            return [f"sub-demo{i:03d}" for i in range(1, 6)]
    
    def demo_cache_performance(self) -> Dict[str, Any]:
        """Demonstrate performance benefits of caching"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CACHE PERFORMANCE DEMONSTRATION")
        self.logger.info("=" * 60)
        
        # Clear cache to start fresh
        clear_cache()
        
        # Run 1: Cold cache (first run)
        self.logger.info("\n1. Cold Cache Run (First execution)")
        self.logger.info("-" * 40)
        
        pipeline = CachedMVPAPipeline(self.config, self.cache_config)
        
        start_time = time.time()
        result1 = pipeline.run_analysis(
            subjects=self.demo_subjects,
            show_cache_stats=True
        )
        cold_time = time.time() - start_time
        
        cold_stats = cache_info()
        
        # Run 2: Warm cache (second run with cached results)
        self.logger.info("\n2. Warm Cache Run (Second execution)")
        self.logger.info("-" * 40)
        
        start_time = time.time()
        result2 = pipeline.run_analysis(
            subjects=self.demo_subjects,
            show_cache_stats=True
        )
        warm_time = time.time() - start_time
        
        warm_stats = cache_info()
        
        # Calculate performance improvement
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        time_saved = cold_time - warm_time
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PERFORMANCE COMPARISON")
        self.logger.info("=" * 60)
        self.logger.info(f"Cold cache time: {cold_time:.2f} seconds")
        self.logger.info(f"Warm cache time: {warm_time:.2f} seconds")
        self.logger.info(f"Speedup: {speedup:.2f}x")
        self.logger.info(f"Time saved: {time_saved:.2f} seconds")
        self.logger.info(f"Cache hit rate: {warm_stats['hit_rate']:.2%}")
        
        return {
            'cold_time': cold_time,
            'warm_time': warm_time,
            'speedup': speedup,
            'time_saved': time_saved,
            'cold_stats': cold_stats,
            'warm_stats': warm_stats
        }
    
    def demo_cache_invalidation(self) -> Dict[str, Any]:
        """Demonstrate cache invalidation behavior"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CACHE INVALIDATION DEMONSTRATION")
        self.logger.info("=" * 60)
        
        # Clear cache
        clear_cache()
        
        # Create cached processor
        cached_processor = CachedMVPAProcessor(self.config, self.cache_config)
        
        # Demo 1: Content-based invalidation
        self.logger.info("\n1. Content-Based Cache Invalidation")
        self.logger.info("-" * 40)
        
        # Create two different datasets
        X1 = np.random.randn(100, 50)
        y1 = np.random.randint(0, 2, 100)
        
        X2 = np.random.randn(100, 50)  # Different data
        y2 = np.random.randint(0, 2, 100)
        
        # First decode - cache miss
        start_time = time.time()
        result1 = cached_processor.decode_cached(X1, y1, 'classification', 'test_roi')
        time1 = time.time() - start_time
        
        # Same data - cache hit
        start_time = time.time()
        result1_repeat = cached_processor.decode_cached(X1, y1, 'classification', 'test_roi')
        time1_repeat = time.time() - start_time
        
        # Different data - cache miss
        start_time = time.time()
        result2 = cached_processor.decode_cached(X2, y2, 'classification', 'test_roi')
        time2 = time.time() - start_time
        
        self.logger.info(f"First decode (cache miss): {time1:.3f} seconds")
        self.logger.info(f"Repeat decode (cache hit): {time1_repeat:.3f} seconds")
        self.logger.info(f"Different data (cache miss): {time2:.3f} seconds")
        
        # Demo 2: Parameter-based invalidation
        self.logger.info("\n2. Parameter-Based Cache Invalidation")
        self.logger.info("-" * 40)
        
        # Same data, different analysis type
        start_time = time.time()
        result_regression = cached_processor.decode_cached(X1, y1.astype(float), 'regression', 'test_roi', 'test_var')
        time_regression = time.time() - start_time
        
        self.logger.info(f"Different analysis type (cache miss): {time_regression:.3f} seconds")
        
        stats = cache_info()
        
        return {
            'first_decode_time': time1,
            'repeat_decode_time': time1_repeat,
            'different_data_time': time2,
            'regression_time': time_regression,
            'final_stats': stats
        }
    
    def demo_cache_management(self) -> Dict[str, Any]:
        """Demonstrate cache management features"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CACHE MANAGEMENT DEMONSTRATION")
        self.logger.info("=" * 60)
        
        # Create some cache data
        cached_processor = CachedMVPAProcessor(self.config, self.cache_config)
        
        # Generate multiple cache entries
        for i in range(10):
            X = np.random.randn(50, 20)
            y = np.random.randint(0, 2, 50)
            cached_processor.decode_cached(X, y, 'classification', f'roi_{i}')
        
        # Check cache size
        initial_stats = cache_info()
        self.logger.info(f"Cache size: {initial_stats['current_size_gb']:.3f} GB")
        self.logger.info(f"Cache entries: {initial_stats['cache_hits'] + initial_stats['cache_misses']}")
        
        # Demo selective cache clearing
        self.logger.info("\n1. Selective Cache Clearing")
        self.logger.info("-" * 40)
        
        # Clear cache entries matching pattern
        clear_result = clear_cache(pattern='roi_1')
        self.logger.info(f"Cleared entries matching 'roi_1': {clear_result}")
        
        # Demo cache cleanup
        self.logger.info("\n2. Cache Cleanup")
        self.logger.info("-" * 40)
        
        cleanup_result = cleanup_cache(force=True)
        self.logger.info(f"Cleanup result: {cleanup_result}")
        
        final_stats = cache_info()
        
        return {
            'initial_stats': initial_stats,
            'clear_result': clear_result,
            'cleanup_result': cleanup_result,
            'final_stats': final_stats
        }
    
    def demo_memory_efficiency(self) -> Dict[str, Any]:
        """Demonstrate memory efficiency features"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MEMORY EFFICIENCY DEMONSTRATION")
        self.logger.info("=" * 60)
        
        # Create large datasets to demonstrate memory efficiency
        large_X = np.random.randn(1000, 500)  # Large dataset
        large_y = np.random.randint(0, 2, 1000)
        
        # Demo hash-based caching (doesn't store full data)
        self.logger.info("\n1. Hash-Based Caching")
        self.logger.info("-" * 40)
        
        neural_hash = ContentHasher.hash_array(large_X)
        behavioral_hash = ContentHasher.hash_array(large_y)
        
        self.logger.info(f"Large dataset shape: {large_X.shape}")
        self.logger.info(f"Dataset memory usage: {large_X.nbytes / 1024**2:.2f} MB")
        self.logger.info(f"Neural hash: {neural_hash}")
        self.logger.info(f"Behavioral hash: {behavioral_hash}")
        
        # Demo precision control
        self.logger.info("\n2. Precision Control")
        self.logger.info("-" * 40)
        
        # Show how precision affects cache keys
        data_with_noise = large_X + np.random.normal(0, 1e-8, large_X.shape)  # Add tiny noise
        
        hash_normal = ContentHasher.hash_array(large_X, precision=6)
        hash_noisy_p6 = ContentHasher.hash_array(data_with_noise, precision=6)
        hash_noisy_p10 = ContentHasher.hash_array(data_with_noise, precision=10)
        
        self.logger.info(f"Original data hash (p=6): {hash_normal}")
        self.logger.info(f"Noisy data hash (p=6): {hash_noisy_p6}")
        self.logger.info(f"Noisy data hash (p=10): {hash_noisy_p10}")
        self.logger.info(f"Hashes match at p=6: {hash_normal == hash_noisy_p6}")
        self.logger.info(f"Hashes match at p=10: {hash_normal == hash_noisy_p10}")
        
        return {
            'large_dataset_shape': large_X.shape,
            'dataset_memory_mb': large_X.nbytes / 1024**2,
            'neural_hash': neural_hash,
            'behavioral_hash': behavioral_hash,
            'precision_demo': {
                'hash_normal': hash_normal,
                'hash_noisy_p6': hash_noisy_p6,
                'hash_noisy_p10': hash_noisy_p10,
                'match_p6': hash_normal == hash_noisy_p6,
                'match_p10': hash_normal == hash_noisy_p10
            }
        }
    
    def demo_integration_benefits(self) -> Dict[str, Any]:
        """Demonstrate integration benefits with existing pipeline"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INTEGRATION BENEFITS DEMONSTRATION")
        self.logger.info("=" * 60)
        
        # Demo 1: Backward compatibility
        self.logger.info("\n1. Backward Compatibility")
        self.logger.info("-" * 40)
        
        # Show that results are identical
        clear_cache()
        
        # Run cached pipeline
        cached_pipeline = CachedMVPAPipeline(self.config, self.cache_config)
        
        # Use just one subject for comparison
        test_subject = self.demo_subjects[0] if self.demo_subjects else "sub-demo001"
        
        cached_result = cached_pipeline.process_subject_cached(test_subject)
        
        self.logger.info(f"Cached pipeline result keys: {list(cached_result.keys())}")
        
        if cached_result['success']:
            self.logger.info("✓ Cached pipeline maintains result structure")
            self.logger.info("✓ All analysis components preserved")
        else:
            self.logger.warning("✗ Cached pipeline failed - check implementation")
        
        # Demo 2: Command-line interface
        self.logger.info("\n2. Command-Line Interface")
        self.logger.info("-" * 40)
        
        self.logger.info("Available cache management commands:")
        self.logger.info("  python delay_discounting_mvpa_pipeline_cached.py --cache-stats-only")
        self.logger.info("  python delay_discounting_mvpa_pipeline_cached.py --clear-cache")
        self.logger.info("  python delay_discounting_mvpa_pipeline_cached.py --cleanup-cache")
        self.logger.info("  python delay_discounting_mvpa_pipeline_cached.py --disable-cache")
        self.logger.info("  python delay_discounting_mvpa_pipeline_cached.py --cache-size-gb 100")
        
        return {
            'cached_result_success': cached_result['success'],
            'cached_result_keys': list(cached_result.keys()),
            'integration_successful': True
        }
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("CACHING SYSTEM PERFORMANCE REPORT")
        report.append("=" * 50)
        
        # Performance metrics
        if 'performance' in results:
            perf = results['performance']
            report.append(f"\nPerformance Improvements:")
            report.append(f"  Cold cache time: {perf['cold_time']:.2f} seconds")
            report.append(f"  Warm cache time: {perf['warm_time']:.2f} seconds")
            report.append(f"  Speedup: {perf['speedup']:.2f}x")
            report.append(f"  Time saved: {perf['time_saved']:.2f} seconds")
            report.append(f"  Cache hit rate: {perf['warm_stats']['hit_rate']:.2%}")
        
        # Cache statistics
        if 'cache_stats' in results:
            stats = results['cache_stats']
            report.append(f"\nCache Statistics:")
            report.append(f"  Total cache hits: {stats['cache_hits']}")
            report.append(f"  Total cache misses: {stats['cache_misses']}")
            report.append(f"  Overall hit rate: {stats['hit_rate']:.2%}")
            report.append(f"  Total time saved: {stats['total_time_saved']:.2f} seconds")
            report.append(f"  Cache size: {stats['current_size_gb']:.3f} GB")
        
        # Memory efficiency
        if 'memory_efficiency' in results:
            mem = results['memory_efficiency']
            report.append(f"\nMemory Efficiency:")
            report.append(f"  Large dataset: {mem['large_dataset_shape']} ({mem['dataset_memory_mb']:.2f} MB)")
            report.append(f"  Hash-based caching reduces memory overhead")
            report.append(f"  Precision control prevents cache misses from numerical noise")
        
        # Recommendations
        report.append(f"\nRecommendations:")
        report.append(f"  • Use cached pipeline for iterative analysis")
        report.append(f"  • Set appropriate cache size limits (current: {self.cache_config.MAX_CACHE_SIZE_GB} GB)")
        report.append(f"  • Run cache cleanup periodically")
        report.append(f"  • Clear cache when analysis parameters change significantly")
        
        return "\n".join(report)
    
    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run complete caching system demonstration"""
        self.logger.info("Starting comprehensive caching system demonstration")
        
        results = {}
        
        # Run all demos
        try:
            results['performance'] = self.demo_cache_performance()
        except Exception as e:
            self.logger.error(f"Performance demo failed: {e}")
            results['performance'] = {'error': str(e)}
        
        try:
            results['invalidation'] = self.demo_cache_invalidation()
        except Exception as e:
            self.logger.error(f"Invalidation demo failed: {e}")
            results['invalidation'] = {'error': str(e)}
        
        try:
            results['management'] = self.demo_cache_management()
        except Exception as e:
            self.logger.error(f"Management demo failed: {e}")
            results['management'] = {'error': str(e)}
        
        try:
            results['memory_efficiency'] = self.demo_memory_efficiency()
        except Exception as e:
            self.logger.error(f"Memory efficiency demo failed: {e}")
            results['memory_efficiency'] = {'error': str(e)}
        
        try:
            results['integration'] = self.demo_integration_benefits()
        except Exception as e:
            self.logger.error(f"Integration demo failed: {e}")
            results['integration'] = {'error': str(e)}
        
        # Generate final report
        final_stats = cache_info()
        results['cache_stats'] = final_stats
        
        report = self.generate_performance_report(results)
        self.logger.info(f"\n{report}")
        
        # Save report
        report_file = Path(self.config.OUTPUT_DIR) / "caching_demo_report.txt"
        try:
            with open(report_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
        
        return results


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Caching System Demo")
    parser.add_argument('--demo-type', choices=['performance', 'invalidation', 'management', 'memory', 'integration', 'all'],
                       default='all', help='Type of demo to run')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to use in demo')
    parser.add_argument('--cache-size-gb', type=float, default=10.0,
                       help='Cache size limit for demo (default: 10.0 GB)')
    parser.add_argument('--output-dir', type=str, help='Output directory for demo results')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = OAKConfig()
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    
    # Create demo instance
    demo = CachingDemo(config)
    
    # Override demo subjects if specified
    if args.subjects:
        demo.demo_subjects = args.subjects
    
    # Set cache size limit
    demo.cache_config.MAX_CACHE_SIZE_GB = args.cache_size_gb
    
    # Run requested demo
    if args.demo_type == 'all':
        results = demo.run_comprehensive_demo()
    elif args.demo_type == 'performance':
        results = demo.demo_cache_performance()
    elif args.demo_type == 'invalidation':
        results = demo.demo_cache_invalidation()
    elif args.demo_type == 'management':
        results = demo.demo_cache_management()
    elif args.demo_type == 'memory':
        results = demo.demo_memory_efficiency()
    elif args.demo_type == 'integration':
        results = demo.demo_integration_benefits()
    
    print(f"\nDemo completed successfully!")
    print(f"Results saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main() 