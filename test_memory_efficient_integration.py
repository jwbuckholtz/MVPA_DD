#!/usr/bin/env python3
"""
Test Memory-Efficient Data Loading Integration
============================================

Simple test script to verify that the memory-efficient data loading system
integrates properly with the existing MVPA pipeline.

This script tests:
1. Memory-efficient loader initialization
2. Integration with caching system
3. Memory usage monitoring
4. Cleanup functionality

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_memory_efficient_imports():
    """Test that all memory-efficient modules can be imported"""
    print("Testing imports...")
    
    try:
        from memory_efficient_data import (
            MemoryConfig, MemoryMonitor, MemoryEfficientLoader,
            MemoryEfficientContext, create_memory_efficient_loader
        )
        print("✓ Memory-efficient data module imported successfully")
        
        from logger_utils import PipelineLogger, setup_pipeline_logging
        print("✓ Logger utilities imported successfully")
        
        from caching_utils import create_cached_processor
        print("✓ Caching utilities imported successfully")
        
        from oak_storage_config import OAKConfig
        print("✓ OAK config imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_memory_config():
    """Test MemoryConfig class"""
    print("\nTesting MemoryConfig...")
    
    try:
        from memory_efficient_data import MemoryConfig
        
        # Test default configuration
        config = MemoryConfig()
        assert config.USE_MEMMAP == True
        assert config.MEMMAP_THRESHOLD_GB == 1.0
        assert config.AVAILABLE_MEMORY_BUFFER == 0.2
        
        # Test custom configuration
        config.MEMMAP_THRESHOLD_GB = 0.5
        config.SHARED_MEMORY_PARALLEL = True
        
        print("✓ MemoryConfig works correctly")
        return True
        
    except Exception as e:
        print(f"✗ MemoryConfig test failed: {e}")
        return False

def test_memory_monitor():
    """Test MemoryMonitor class"""
    print("\nTesting MemoryMonitor...")
    
    try:
        from memory_efficient_data import MemoryMonitor, MemoryConfig
        
        config = MemoryConfig()
        monitor = MemoryMonitor(config)
        
        # Test memory info
        memory_info = monitor.get_memory_info()
        assert 'total_gb' in memory_info
        assert 'available_gb' in memory_info
        assert 'process_memory_gb' in memory_info
        
        # Test memory threshold check
        can_allocate = monitor.check_memory_threshold(0.1)  # 0.1 GB
        assert isinstance(can_allocate, bool)
        
        # Test memmap suggestion
        should_memmap = monitor.suggest_memmap(2.0)  # 2 GB
        assert isinstance(should_memmap, bool)
        
        print("✓ MemoryMonitor works correctly")
        return True
        
    except Exception as e:
        print(f"✗ MemoryMonitor test failed: {e}")
        return False

def test_memory_efficient_loader():
    """Test MemoryEfficientLoader class"""
    print("\nTesting MemoryEfficientLoader...")
    
    try:
        from memory_efficient_data import create_memory_efficient_loader, MemoryConfig
        from oak_storage_config import OAKConfig
        
        # Create configurations
        config = OAKConfig()
        memory_config = MemoryConfig()
        memory_config.USE_MEMMAP = True
        memory_config.MEMMAP_THRESHOLD_GB = 0.1  # Force memmap for testing
        
        # Create loader
        loader = create_memory_efficient_loader(config, memory_config)
        
        # Test memory usage report
        report = loader.get_memory_usage_report()
        assert 'system_memory' in report
        assert 'active_memmaps' in report
        
        # Test cleanup
        loader.cleanup()
        
        print("✓ MemoryEfficientLoader works correctly")
        return True
        
    except Exception as e:
        print(f"✗ MemoryEfficientLoader test failed: {e}")
        return False

def test_memory_efficient_context():
    """Test MemoryEfficientContext context manager"""
    print("\nTesting MemoryEfficientContext...")
    
    try:
        from memory_efficient_data import MemoryEfficientContext
        
        # Test context manager
        with MemoryEfficientContext() as loader:
            # Test that loader is available
            assert hasattr(loader, 'load_fmri_memmap')
            assert hasattr(loader, 'extract_roi_timeseries_memmap')
            assert hasattr(loader, 'get_memory_usage_report')
            
            # Test memory usage report
            report = loader.get_memory_usage_report()
            assert isinstance(report, dict)
        
        print("✓ MemoryEfficientContext works correctly")
        return True
        
    except Exception as e:
        print(f"✗ MemoryEfficientContext test failed: {e}")
        return False

def test_caching_integration():
    """Test integration with caching system"""
    print("\nTesting caching integration...")
    
    try:
        from caching_utils import create_cached_processor, CacheConfig
        from memory_efficient_data import MemoryConfig
        from oak_storage_config import OAKConfig
        
        # Create configurations
        config = OAKConfig()
        
        cache_config = CacheConfig()
        cache_config.CACHE_DIR = str(Path(tempfile.gettempdir()) / 'test_cache')
        
        memory_config = MemoryConfig()
        memory_config.USE_MEMMAP = True
        memory_config.MEMMAP_TEMP_DIR = str(Path(tempfile.gettempdir()) / 'test_memmap')
        
        # Create cached processor with memory efficiency
        processor = create_cached_processor(config, cache_config, memory_config)
        
        # Test that processor has memory loader
        assert hasattr(processor, 'memory_loader')
        assert hasattr(processor, 'get_memory_usage_report')
        
        # Test memory usage report
        report = processor.get_memory_usage_report()
        assert 'cache_stats' in report
        assert 'memory_stats' in report
        
        # Cleanup
        processor.cleanup()
        
        print("✓ Caching integration works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Caching integration test failed: {e}")
        return False

def test_logger_integration():
    """Test integration with logger system"""
    print("\nTesting logger integration...")
    
    try:
        from logger_utils import setup_pipeline_logging, PipelineLogger
        
        # Test pipeline logger
        logger = setup_pipeline_logging('test_memory_efficient')
        assert hasattr(logger, 'log_memory_usage')
        assert hasattr(logger, 'log_performance')
        
        # Test PipelineLogger directly
        pipeline_logger = PipelineLogger('test_direct')
        assert hasattr(pipeline_logger, 'logger')
        
        print("✓ Logger integration works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Logger integration test failed: {e}")
        return False

def test_synthetic_data_creation():
    """Test creation of synthetic data for testing"""
    print("\nTesting synthetic data creation...")
    
    try:
        # Create synthetic fMRI data
        shape = (32, 32, 32, 50)  # Small for testing
        data = np.random.randn(*shape).astype(np.float32)
        
        # Test that data has correct properties
        assert data.shape == shape
        assert data.dtype == np.float32
        
        # Create synthetic ROI mask
        roi_mask = np.zeros(shape[:3], dtype=bool)
        center = [s//2 for s in shape[:3]]
        roi_mask[center[0]-2:center[0]+3, center[1]-2:center[1]+3, center[2]-2:center[2]+3] = True
        
        # Test mask properties
        assert roi_mask.shape == shape[:3]
        assert roi_mask.sum() > 0
        
        print("✓ Synthetic data creation works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Synthetic data creation test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("="*60)
    print("MEMORY-EFFICIENT DATA LOADING INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Import Tests", test_memory_efficient_imports),
        ("MemoryConfig Tests", test_memory_config),
        ("MemoryMonitor Tests", test_memory_monitor),
        ("MemoryEfficientLoader Tests", test_memory_efficient_loader),
        ("MemoryEfficientContext Tests", test_memory_efficient_context),
        ("Caching Integration Tests", test_caching_integration),
        ("Logger Integration Tests", test_logger_integration),
        ("Synthetic Data Tests", test_synthetic_data_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Memory-efficient system is ready to use.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 