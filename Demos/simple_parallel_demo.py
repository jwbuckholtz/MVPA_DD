#!/usr/bin/env python3
"""
Simple Parallel Processing Demo for MVPA Pipeline
================================================

This demo shows the core parallel processing concepts that have been added
to the MVPA pipeline, without requiring all the full pipeline dependencies.

Usage:
    python3 simple_parallel_demo.py
"""

import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp
from pathlib import Path

# Mock configuration classes
class MockConfig:
    """Mock configuration for demo purposes"""
    def __init__(self):
        self.ROI_MASKS = {
            'striatum': './masks/striatum_mask.nii.gz',
            'dlpfc': './masks/dlpfc_mask.nii.gz',
            'vmpfc': './masks/vmpfc_mask.nii.gz'
        }
        self.CV_FOLDS = 5
        self.N_PERMUTATIONS = 100  # Reduced for demo
        self.OUTPUT_DIR = './demo_outputs'

class MockParallelConfig:
    """Mock parallel configuration for demo purposes"""
    def __init__(self):
        self.N_JOBS_SUBJECTS = 2
        self.N_JOBS_ROIS = 2
        self.N_JOBS_MVPA = 1
        self.BACKEND = 'loky'
        self.CHUNK_SIZE = 10
        self.MAX_MEMORY_GB = 8
        self.VERBOSE = 1

# Mock processing functions
def mock_process_subject_serial(subject_id, roi_names, processing_time=1.0):
    """Mock serial processing of a single subject"""
    print(f"  Processing subject {subject_id} (serial)")
    
    results = {}
    for roi_name in roi_names:
        print(f"    Processing {roi_name}...")
        time.sleep(processing_time / len(roi_names))  # Simulate processing time
        
        # Mock MVPA results
        results[roi_name] = {
            'success': True,
            'choice_accuracy': np.random.uniform(0.5, 0.8),
            'choice_p_value': np.random.uniform(0.001, 0.05),
            'sv_diff_r2': np.random.uniform(0.1, 0.4),
            'sv_diff_p_value': np.random.uniform(0.001, 0.05)
        }
    
    return {
        'success': True,
        'subject_id': subject_id,
        'mvpa_results': results
    }

def mock_process_subject_parallel(subject_id, roi_names, processing_time=1.0):
    """Mock parallel processing of a single subject"""
    print(f"  Processing subject {subject_id} (parallel)")
    
    def process_roi(roi_name):
        print(f"    Processing {roi_name} (parallel)...")
        time.sleep(processing_time / len(roi_names))  # Simulate processing time
        
        # Mock MVPA results
        return roi_name, {
            'success': True,
            'choice_accuracy': np.random.uniform(0.5, 0.8),
            'choice_p_value': np.random.uniform(0.001, 0.05),
            'sv_diff_r2': np.random.uniform(0.1, 0.4),
            'sv_diff_p_value': np.random.uniform(0.001, 0.05)
        }
    
    # Process ROIs in parallel
    roi_results = Parallel(n_jobs=2)(delayed(process_roi)(roi) for roi in roi_names)
    
    # Convert to dictionary
    results = {}
    for roi_name, result in roi_results:
        results[roi_name] = result
    
    return {
        'success': True,
        'subject_id': subject_id,
        'mvpa_results': results
    }

def demo_serial_vs_parallel():
    """Demo serial vs parallel processing comparison"""
    print("=" * 60)
    print("DEMO: SERIAL vs PARALLEL PROCESSING")
    print("=" * 60)
    
    # Setup
    config = MockConfig()
    subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    roi_names = list(config.ROI_MASKS.keys())
    
    print(f"Processing {len(subjects)} subjects with {len(roi_names)} ROIs each")
    print(f"ROIs: {', '.join(roi_names)}")
    print()
    
    # Serial processing
    print("1. Serial Processing:")
    start_time = time.time()
    
    serial_results = []
    for subject_id in subjects:
        result = mock_process_subject_serial(subject_id, roi_names, processing_time=0.5)
        serial_results.append(result)
    
    serial_time = time.time() - start_time
    print(f"   Serial processing time: {serial_time:.2f} seconds")
    print()
    
    # Parallel processing (subjects in parallel)
    print("2. Parallel Processing (Subjects):")
    start_time = time.time()
    
    # Process subjects in parallel
    parallel_results = Parallel(n_jobs=2)(
        delayed(mock_process_subject_serial)(subject_id, roi_names, processing_time=0.5) 
        for subject_id in subjects
    )
    
    parallel_time = time.time() - start_time
    print(f"   Parallel processing time: {parallel_time:.2f} seconds")
    print(f"   Speedup: {serial_time/parallel_time:.1f}x")
    print()
    
    # Full parallel processing (subjects AND ROIs in parallel)
    print("3. Full Parallel Processing (Subjects + ROIs):")
    start_time = time.time()
    
    # Process subjects in parallel, with ROIs in parallel within each subject
    full_parallel_results = Parallel(n_jobs=2)(
        delayed(mock_process_subject_parallel)(subject_id, roi_names, processing_time=0.5) 
        for subject_id in subjects
    )
    
    full_parallel_time = time.time() - start_time
    print(f"   Full parallel processing time: {full_parallel_time:.2f} seconds")
    print(f"   Speedup: {serial_time/full_parallel_time:.1f}x")
    print()
    
    # Summary
    print("Summary:")
    print(f"  - Serial: {serial_time:.2f}s")
    print(f"  - Subject parallel: {parallel_time:.2f}s ({serial_time/parallel_time:.1f}x)")
    print(f"  - Full parallel: {full_parallel_time:.2f}s ({serial_time/full_parallel_time:.1f}x)")
    
    return {
        'serial_time': serial_time,
        'parallel_time': parallel_time,
        'full_parallel_time': full_parallel_time,
        'serial_results': serial_results,
        'parallel_results': parallel_results,
        'full_parallel_results': full_parallel_results
    }

def demo_resource_optimization():
    """Demo resource optimization and configuration"""
    print("\n" + "=" * 60)
    print("DEMO: RESOURCE OPTIMIZATION")
    print("=" * 60)
    
    # Get system information
    cpu_count = mp.cpu_count()
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        print(f"System Resources:")
        print(f"  - CPU cores: {cpu_count}")
        print(f"  - Total memory: {memory_gb:.1f} GB")
        print(f"  - Available memory: {available_memory_gb:.1f} GB")
    except ImportError:
        print(f"System Resources:")
        print(f"  - CPU cores: {cpu_count}")
        print(f"  - Memory: (psutil not available)")
    
    print()
    
    # Configuration recommendations
    print("Configuration Recommendations:")
    
    # Small dataset
    print("  Small dataset (≤10 subjects):")
    print(f"    - Subject jobs: {min(2, cpu_count // 2)}")
    print(f"    - ROI jobs: {min(3, cpu_count // 2)}")
    print(f"    - MVPA jobs: 1")
    
    # Medium dataset
    print("  Medium dataset (10-50 subjects):")
    print(f"    - Subject jobs: {min(4, cpu_count // 2)}")
    print(f"    - ROI jobs: {min(2, cpu_count // 4)}")
    print(f"    - MVPA jobs: 1")
    
    # Large dataset
    print("  Large dataset (>50 subjects):")
    print(f"    - Subject jobs: {min(8, cpu_count)}")
    print(f"    - ROI jobs: 1")
    print(f"    - MVPA jobs: 1")
    
    return {
        'cpu_count': cpu_count,
        'available_memory_gb': available_memory_gb if 'psutil' in locals() else None
    }

def demo_memory_chunking():
    """Demo memory-efficient chunking for large datasets"""
    print("\n" + "=" * 60)
    print("DEMO: MEMORY-EFFICIENT CHUNKING")
    print("=" * 60)
    
    # Simulate large dataset
    large_subject_list = [f"sub-{i:03d}" for i in range(1, 101)]  # 100 subjects
    
    print(f"Large dataset: {len(large_subject_list)} subjects")
    
    # Different chunk sizes
    chunk_sizes = [10, 20, 50]
    
    for chunk_size in chunk_sizes:
        chunks = [large_subject_list[i:i+chunk_size] 
                 for i in range(0, len(large_subject_list), chunk_size)]
        
        print(f"  Chunk size {chunk_size}:")
        print(f"    - Number of chunks: {len(chunks)}")
        print(f"    - Subjects per chunk: {[len(chunk) for chunk in chunks]}")
        print(f"    - Memory efficiency: {100 * (1 - chunk_size/len(large_subject_list)):.1f}%")
    
    return {
        'total_subjects': len(large_subject_list),
        'chunk_analysis': {
            chunk_size: {
                'n_chunks': len([large_subject_list[i:i+chunk_size] 
                               for i in range(0, len(large_subject_list), chunk_size)]),
                'subjects_per_chunk': [len(chunk) for chunk in 
                                     [large_subject_list[i:i+chunk_size] 
                                      for i in range(0, len(large_subject_list), chunk_size)]]
            }
            for chunk_size in chunk_sizes
        }
    }

def demo_error_handling():
    """Demo error handling and robustness"""
    print("\n" + "=" * 60)
    print("DEMO: ERROR HANDLING AND ROBUSTNESS")
    print("=" * 60)
    
    def mock_process_with_errors(subject_id, error_rate=0.3):
        """Mock processing function that sometimes fails"""
        import random
        
        if random.random() < error_rate:
            raise Exception(f"Mock error for {subject_id}")
        
        time.sleep(0.1)  # Simulate processing
        return {'success': True, 'subject_id': subject_id}
    
    subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005']
    
    print(f"Processing {len(subjects)} subjects with 30% error rate")
    print()
    
    # Process with error handling
    results = []
    for subject_id in subjects:
        try:
            result = mock_process_with_errors(subject_id)
            results.append(result)
            print(f"  ✓ {subject_id}: Success")
        except Exception as e:
            results.append({'success': False, 'subject_id': subject_id, 'error': str(e)})
            print(f"  ✗ {subject_id}: Failed - {e}")
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"\nResults:")
    print(f"  - Successful: {successful}/{len(subjects)}")
    print(f"  - Failed: {failed}/{len(subjects)}")
    print(f"  - Success rate: {successful/len(subjects)*100:.1f}%")
    
    return {
        'results': results,
        'successful': successful,
        'failed': failed,
        'success_rate': successful/len(subjects)
    }

def demo_performance_visualization():
    """Demo performance visualization"""
    print("\n" + "=" * 60)
    print("DEMO: PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    # Create mock performance data
    configurations = [
        {'name': 'Serial', 'subject_jobs': 1, 'roi_jobs': 1, 'time': 100.0},
        {'name': 'Subject Parallel', 'subject_jobs': 2, 'roi_jobs': 1, 'time': 55.0},
        {'name': 'ROI Parallel', 'subject_jobs': 1, 'roi_jobs': 2, 'time': 70.0},
        {'name': 'Full Parallel', 'subject_jobs': 2, 'roi_jobs': 2, 'time': 35.0},
    ]
    
    print("Performance Comparison:")
    print("Configuration          | Time (s) | Speedup | Efficiency")
    print("-" * 55)
    
    baseline_time = configurations[0]['time']
    
    for config in configurations:
        speedup = baseline_time / config['time']
        max_speedup = config['subject_jobs'] * config['roi_jobs']
        efficiency = speedup / max_speedup
        
        print(f"{config['name']:<20} | {config['time']:>7.1f} | {speedup:>6.1f}x | {efficiency:>8.1%}")
    
    return configurations

def main():
    """Main demo function"""
    print("Simple Parallel Processing Demo for MVPA Pipeline")
    print("=" * 60)
    print("This demo shows the core parallel processing concepts")
    print("that have been added to the MVPA pipeline.")
    print()
    
    try:
        # Run demos
        print("Running parallel processing demos...")
        
        # Demo 1: Serial vs parallel comparison
        timing_results = demo_serial_vs_parallel()
        
        # Demo 2: Resource optimization
        resource_info = demo_resource_optimization()
        
        # Demo 3: Memory chunking
        chunking_results = demo_memory_chunking()
        
        # Demo 4: Error handling
        error_results = demo_error_handling()
        
        # Demo 5: Performance visualization
        performance_data = demo_performance_visualization()
        
        # Final summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        
        print(f"Parallel processing capabilities demonstrated:")
        print(f"  - Per-subject parallelization: ✓")
        print(f"  - Per-ROI parallelization: ✓")
        print(f"  - Memory-efficient chunking: ✓")
        print(f"  - Error handling: ✓")
        print(f"  - Resource optimization: ✓")
        
        if timing_results:
            speedup = timing_results['serial_time'] / timing_results['full_parallel_time']
            print(f"\nPerformance improvement: {speedup:.1f}x speedup achieved")
        
        if error_results:
            print(f"Error handling: {error_results['success_rate']:.1%} success rate maintained")
        
        print(f"\n✓ All demos completed successfully!")
        print(f"The parallel processing enhancements are ready for use.")
        
    except Exception as e:
        print(f"Error running demos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 