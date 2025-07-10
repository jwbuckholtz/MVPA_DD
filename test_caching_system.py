#!/usr/bin/env python3
"""
Test Script for Caching System
=============================

Simple test script to verify that the caching system works correctly.
Tests basic functionality including cache hits/misses and performance.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Import caching utilities
from caching_utils import (
    ContentHasher, CacheManager, CachedMVPAProcessor, 
    CacheConfig, cache_info, clear_cache
)
from oak_storage_config import OAKConfig


def test_content_hasher():
    """Test ContentHasher functionality"""
    print("Testing ContentHasher...")
    
    # Test array hashing
    X1 = np.random.randn(100, 50)
    X2 = X1.copy()
    X3 = np.random.randn(100, 50)
    
    hash1 = ContentHasher.hash_array(X1)
    hash2 = ContentHasher.hash_array(X2)
    hash3 = ContentHasher.hash_array(X3)
    
    assert hash1 == hash2, "Identical arrays should have same hash"
    assert hash1 != hash3, "Different arrays should have different hashes"
    print("✓ Array hashing works correctly")
    
    # Test DataFrame hashing
    df1 = pd.DataFrame({
        'col1': np.random.randn(50),
        'col2': np.random.randint(0, 2, 50)
    })
    df2 = df1.copy()
    df3 = pd.DataFrame({
        'col1': np.random.randn(50),
        'col2': np.random.randint(0, 2, 50)
    })
    
    hash_df1 = ContentHasher.hash_dataframe(df1)
    hash_df2 = ContentHasher.hash_dataframe(df2)
    hash_df3 = ContentHasher.hash_dataframe(df3)
    
    assert hash_df1 == hash_df2, "Identical DataFrames should have same hash"
    assert hash_df1 != hash_df3, "Different DataFrames should have different hashes"
    print("✓ DataFrame hashing works correctly")
    
    # Test precision control
    X_noisy = X1 + np.random.normal(0, 1e-8, X1.shape)
    hash_precise = ContentHasher.hash_array(X1, precision=6)
    hash_noisy_p6 = ContentHasher.hash_array(X_noisy, precision=6)
    hash_noisy_p10 = ContentHasher.hash_array(X_noisy, precision=10)
    
    assert hash_precise == hash_noisy_p6, "Small differences should not affect hash at precision=6"
    assert hash_precise != hash_noisy_p10, "Small differences should affect hash at precision=10"
    print("✓ Precision control works correctly")


def test_cache_manager():
    """Test CacheManager functionality"""
    print("\nTesting CacheManager...")
    
    # Setup test cache
    test_cache_dir = Path('./test_cache')
    cache_config = CacheConfig()
    cache_config.CACHE_DIR = str(test_cache_dir)
    cache_config.MAX_CACHE_SIZE_GB = 1.0  # Small for testing
    
    cache_manager = CacheManager(config=cache_config)
    
    # Test statistics
    initial_stats = cache_manager.get_stats()
    assert 'cache_hits' in initial_stats
    assert 'cache_misses' in initial_stats
    print("✓ Cache statistics retrieval works")
    
    # Test cache size calculation
    cache_size = cache_manager.get_cache_size()
    assert cache_size >= 0.0
    print(f"✓ Cache size calculation works: {cache_size:.3f} GB")
    
    # Test cache clearing
    clear_result = cache_manager.clear_cache()
    assert 'removed_files' in clear_result
    print("✓ Cache clearing works")
    
    # Cleanup
    import shutil
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)


def test_cached_processor():
    """Test CachedMVPAProcessor functionality"""
    print("\nTesting CachedMVPAProcessor...")
    
    # Setup test configuration
    config = OAKConfig()
    cache_config = CacheConfig()
    cache_config.CACHE_DIR = './test_cache'
    cache_config.MAX_CACHE_SIZE_GB = 1.0
    
    # Clear cache to start fresh
    clear_cache()
    
    # Create processor
    processor = CachedMVPAProcessor(config, cache_config)
    
    # Test cached decoding
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 2, 100)
    
    # First call - should be cache miss
    start_time = time.time()
    result1 = processor.decode_cached(X, y, 'classification', 'test_roi')
    time1 = time.time() - start_time
    
    # Second call - should be cache hit
    start_time = time.time()
    result2 = processor.decode_cached(X, y, 'classification', 'test_roi')
    time2 = time.time() - start_time
    
    # Verify results are identical
    assert result1['accuracy'] == result2['accuracy'], "Cache should return identical results"
    assert time2 < time1, "Cached call should be faster"
    
    print(f"✓ Cached decoding works: {time1:.3f}s -> {time2:.3f}s ({time1/time2:.1f}x speedup)")
    
    # Test different data - should be cache miss
    X_different = np.random.randn(100, 50)
    start_time = time.time()
    result3 = processor.decode_cached(X_different, y, 'classification', 'test_roi')
    time3 = time.time() - start_time
    
    assert time3 > time2, "Different data should cause cache miss"
    print("✓ Cache invalidation works for different data")
    
    # Check cache statistics
    stats = cache_info()
    assert stats['cache_hits'] > 0, "Should have cache hits"
    assert stats['cache_misses'] > 0, "Should have cache misses"
    print(f"✓ Cache statistics: {stats['cache_hits']} hits, {stats['cache_misses']} misses")
    
    # Cleanup
    clear_cache()


def test_cache_performance():
    """Test cache performance improvements"""
    print("\nTesting cache performance...")
    
    # Setup
    config = OAKConfig()
    cache_config = CacheConfig()
    cache_config.CACHE_DIR = './test_cache'
    
    clear_cache()
    
    processor = CachedMVPAProcessor(config, cache_config)
    
    # Create test data
    X = np.random.randn(200, 100)  # Larger for more realistic timing
    y = np.random.randint(0, 2, 200)
    
    # Multiple calls to build cache
    times = []
    for i in range(5):
        start_time = time.time()
        result = processor.decode_cached(X, y, 'classification', f'roi_{i}')
        times.append(time.time() - start_time)
    
    # Call cached versions
    cached_times = []
    for i in range(5):
        start_time = time.time()
        result = processor.decode_cached(X, y, 'classification', f'roi_{i}')
        cached_times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    avg_cached_time = np.mean(cached_times)
    speedup = avg_time / avg_cached_time if avg_cached_time > 0 else float('inf')
    
    print(f"✓ Performance improvement: {avg_time:.3f}s -> {avg_cached_time:.3f}s ({speedup:.1f}x speedup)")
    
    # Check final statistics
    final_stats = cache_info()
    print(f"✓ Final cache stats: {final_stats['cache_hits']} hits, {final_stats['cache_misses']} misses")
    
    # Cleanup
    clear_cache()


def run_all_tests():
    """Run all caching system tests"""
    print("Running Caching System Tests")
    print("=" * 40)
    
    try:
        test_content_hasher()
        test_cache_manager()
        test_cached_processor()
        test_cache_performance()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed!")
        print("Caching system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 Caching system implementation is complete and working!")
        print("\nNext steps:")
        print("1. Run the demo: python demo_caching_system.py")
        print("2. Use cached pipeline: python delay_discounting_mvpa_pipeline_cached.py")
        print("3. Check documentation: CACHING_SYSTEM_README.md")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 