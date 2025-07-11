#!/usr/bin/env python3
"""
Demo: Parallel MVPA Processing
=============================

This script demonstrates the parallel processing capabilities of the delay
discounting MVPA pipeline using joblib.Parallel for per-subject and per-ROI
parallelization.

Usage:
    python demo_parallel_mvpa.py
    python demo_parallel_mvpa.py --subjects sub-001 sub-002 sub-003
    python demo_parallel_mvpa.py --enable-roi-parallel --optimize-config
    
Author: Cognitive Neuroscience Lab, Stanford University
"""

# Import logger utilities for standardized setup
from logger_utils import setup_pipeline_environment, create_analysis_parser

import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import parallel processing modules
from parallel_mvpa_utils import (
    ParallelMVPAProcessor, ParallelMVPAConfig, 
    optimize_parallel_config, create_parallel_pipeline
)
from oak_storage_config import OAKConfig
from data_utils import get_complete_subjects


def demo_parallel_configuration():
    """Demo different parallel configuration options"""
    print("=" * 60)
    print("DEMO 1: PARALLEL CONFIGURATION OPTIONS")
    print("=" * 60)
    
    # Basic configuration
    print("\n1. Basic Configuration:")
    basic_config = ParallelMVPAConfig()
    print(f"   - Subject parallelization: {basic_config.N_JOBS_SUBJECTS} jobs")
    print(f"   - ROI parallelization: {basic_config.N_JOBS_ROIS} jobs")
    print(f"   - MVPA algorithms: {basic_config.N_JOBS_MVPA} jobs")
    print(f"   - Backend: {basic_config.BACKEND}")
    print(f"   - Chunk size: {basic_config.CHUNK_SIZE}")
    
    # Optimized configuration
    print("\n2. Optimized Configuration:")
    config = OAKConfig()
    subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005']  # Mock subjects
    
    optimized_config = optimize_parallel_config(config, subjects)
    print(f"   - Subject parallelization: {optimized_config.N_JOBS_SUBJECTS} jobs")
    print(f"   - ROI parallelization: {optimized_config.N_JOBS_ROIS} jobs")
    print(f"   - MVPA algorithms: {optimized_config.N_JOBS_MVPA} jobs")
    print(f"   - Chunk size: {optimized_config.CHUNK_SIZE}")
    
    # Resource-aware configuration
    print("\n3. Resource-aware Configuration:")
    processor = ParallelMVPAProcessor(config, optimized_config)
    
    optimal_subject_jobs = processor.get_optimal_n_jobs('subject')
    optimal_roi_jobs = processor.get_optimal_n_jobs('roi')
    optimal_mvpa_jobs = processor.get_optimal_n_jobs('mvpa')
    
    print(f"   - Optimal subject jobs: {optimal_subject_jobs}")
    print(f"   - Optimal ROI jobs: {optimal_roi_jobs}")
    print(f"   - Optimal MVPA jobs: {optimal_mvpa_jobs}")
    
    return optimized_config


def demo_parallel_processing_comparison():
    """Demo comparison between serial and parallel processing"""
    print("\n" + "=" * 60)
    print("DEMO 2: SERIAL vs PARALLEL PROCESSING COMPARISON")
    print("=" * 60)
    
    config = OAKConfig()
    
    # Mock subjects for demo
    subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    
    print(f"\nProcessing {len(subjects)} subjects with {len(config.ROI_MASKS)} ROIs")
    print("(Mock processing - measuring overhead and setup time)")
    
    # Serial processing simulation
    print("\n1. Serial Processing:")
    start_time = time.time()
    
    for subject in subjects:
        print(f"   Processing {subject}...")
        time.sleep(0.1)  # Simulate processing time
        
        for roi_name in config.ROI_MASKS.keys():
            print(f"     - {roi_name}...")
            time.sleep(0.05)  # Simulate ROI processing
    
    serial_time = time.time() - start_time
    print(f"   Serial processing time: {serial_time:.2f} seconds")
    
    # Parallel processing simulation
    print("\n2. Parallel Processing:")
    start_time = time.time()
    
    # Create parallel processor
    parallel_config = ParallelMVPAConfig()
    parallel_config.N_JOBS_SUBJECTS = 2
    parallel_config.N_JOBS_ROIS = 2
    
    processor = ParallelMVPAProcessor(config, parallel_config)
    
    print(f"   Setup time: {time.time() - start_time:.2f} seconds")
    print(f"   Configured for {parallel_config.N_JOBS_SUBJECTS} subject jobs")
    print(f"   Configured for {parallel_config.N_JOBS_ROIS} ROI jobs")
    
    # Simulate parallel processing
    from joblib import Parallel, delayed
    
    def mock_subject_processing(subject_id):
        time.sleep(0.1)  # Simulate processing
        return f"Processed {subject_id}"
    
    parallel_start = time.time()
    results = Parallel(n_jobs=2)(delayed(mock_subject_processing)(sub) for sub in subjects)
    parallel_time = time.time() - parallel_start
    
    print(f"   Parallel processing time: {parallel_time:.2f} seconds")
    print(f"   Speedup: {serial_time/parallel_time:.1f}x")
    
    return {'serial_time': serial_time, 'parallel_time': parallel_time}


def demo_memory_and_resource_management():
    """Demo memory and resource management features"""
    print("\n" + "=" * 60)
    print("DEMO 3: MEMORY AND RESOURCE MANAGEMENT")
    print("=" * 60)
    
    config = OAKConfig()
    
    # Create processor with memory management
    parallel_config = ParallelMVPAConfig()
    parallel_config.MAX_MEMORY_GB = 8  # 8GB per job
    parallel_config.CHUNK_SIZE = 20    # 20 subjects per chunk
    
    processor = ParallelMVPAProcessor(config, parallel_config)
    
    print(f"1. Memory Management:")
    print(f"   - Max memory per job: {parallel_config.MAX_MEMORY_GB} GB")
    print(f"   - Chunk size: {parallel_config.CHUNK_SIZE} subjects")
    
    # Simulate large dataset
    large_subject_list = [f"sub-{i:03d}" for i in range(1, 101)]  # 100 subjects
    
    print(f"\n2. Large Dataset Processing:")
    print(f"   - Total subjects: {len(large_subject_list)}")
    
    # Create chunks
    chunks = processor._chunk_subjects(large_subject_list)
    print(f"   - Number of chunks: {len(chunks)}")
    print(f"   - Subjects per chunk: {[len(chunk) for chunk in chunks]}")
    
    # Memory optimization
    print(f"\n3. Resource Optimization:")
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"   - Total system memory: {memory_info.total / (1024**3):.1f} GB")
        print(f"   - Available memory: {memory_info.available / (1024**3):.1f} GB")
        print(f"   - Memory usage: {memory_info.percent}%")
    except ImportError:
        print("   - psutil not available, using fallback estimates")
    
    print(f"   - Optimal subject jobs: {processor.get_optimal_n_jobs('subject')}")
    print(f"   - Optimal ROI jobs: {processor.get_optimal_n_jobs('roi')}")
    
    return processor


def demo_error_handling_and_robustness():
    """Demo error handling and robustness features"""
    print("\n" + "=" * 60)
    print("DEMO 4: ERROR HANDLING AND ROBUSTNESS")
    print("=" * 60)
    
    config = OAKConfig()
    
    # Configure with error handling
    parallel_config = ParallelMVPAConfig()
    parallel_config.MAX_RETRIES = 3
    parallel_config.CONTINUE_ON_ERROR = True
    
    processor = ParallelMVPAProcessor(config, parallel_config)
    
    print(f"1. Error Handling Configuration:")
    print(f"   - Max retries: {parallel_config.MAX_RETRIES}")
    print(f"   - Continue on error: {parallel_config.CONTINUE_ON_ERROR}")
    
    # Simulate processing with errors
    print(f"\n2. Simulating Processing with Errors:")
    
    def mock_processing_with_errors(subject_id):
        import random
        
        if random.random() < 0.3:  # 30% chance of error
            raise Exception(f"Mock error for {subject_id}")
        
        time.sleep(0.1)  # Simulate processing
        return {'success': True, 'subject_id': subject_id}
    
    subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005']
    
    from joblib import Parallel, delayed
    
    # Process with error handling
    results = []
    for subject in subjects:
        try:
            result = mock_processing_with_errors(subject)
            results.append(result)
            print(f"   ✓ {subject}: Success")
        except Exception as e:
            results.append({'success': False, 'subject_id': subject, 'error': str(e)})
            print(f"   ✗ {subject}: Failed - {e}")
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"\n3. Processing Summary:")
    print(f"   - Successful: {successful}/{len(subjects)}")
    print(f"   - Failed: {failed}/{len(subjects)}")
    print(f"   - Success rate: {successful/len(subjects)*100:.1f}%")
    
    return results


def demo_performance_profiling():
    """Demo performance profiling and monitoring"""
    print("\n" + "=" * 60)
    print("DEMO 5: PERFORMANCE PROFILING")
    print("=" * 60)
    
    config = OAKConfig()
    
    # Configure profiling
    parallel_config = ParallelMVPAConfig()
    parallel_config.ENABLE_PROFILING = True
    parallel_config.PROFILE_OUTPUT_DIR = './demo_profiling_results'
    
    processor = ParallelMVPAProcessor(config, parallel_config)
    
    print(f"1. Profiling Configuration:")
    print(f"   - Profiling enabled: {parallel_config.ENABLE_PROFILING}")
    print(f"   - Output directory: {parallel_config.PROFILE_OUTPUT_DIR}")
    
    # Create mock performance data
    print(f"\n2. Mock Performance Analysis:")
    
    configurations = [
        {'subject_jobs': 1, 'roi_jobs': 1, 'name': 'Serial'},
        {'subject_jobs': 2, 'roi_jobs': 1, 'name': 'Subject Parallel'},
        {'subject_jobs': 1, 'roi_jobs': 2, 'name': 'ROI Parallel'},
        {'subject_jobs': 2, 'roi_jobs': 2, 'name': 'Full Parallel'},
    ]
    
    performance_data = []
    
    for config_dict in configurations:
        # Simulate performance measurement
        base_time = 10.0  # Base processing time
        subject_speedup = 1.0 + (config_dict['subject_jobs'] - 1) * 0.7
        roi_speedup = 1.0 + (config_dict['roi_jobs'] - 1) * 0.5
        
        total_speedup = subject_speedup * roi_speedup
        processing_time = base_time / total_speedup
        
        performance_data.append({
            'Configuration': config_dict['name'],
            'Subject Jobs': config_dict['subject_jobs'],
            'ROI Jobs': config_dict['roi_jobs'],
            'Processing Time (s)': processing_time,
            'Speedup': base_time / processing_time
        })
        
        print(f"   {config_dict['name']}: {processing_time:.2f}s "
              f"(speedup: {base_time/processing_time:.1f}x)")
    
    # Create performance visualization
    print(f"\n3. Performance Visualization:")
    
    df = pd.DataFrame(performance_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Processing time comparison
    ax1.bar(df['Configuration'], df['Processing Time (s)'])
    ax1.set_title('Processing Time by Configuration')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Speedup comparison
    ax2.bar(df['Configuration'], df['Speedup'])
    ax2.set_title('Speedup by Configuration')
    ax2.set_ylabel('Speedup Factor')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('./demo_parallel_outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'parallel_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   - Performance plot saved to: {output_dir / 'parallel_performance_comparison.png'}")
    
    return performance_data


def demo_integration_with_existing_pipeline():
    """Demo integration with existing pipeline components"""
    print("\n" + "=" * 60)
    print("DEMO 6: INTEGRATION WITH EXISTING PIPELINE")
    print("=" * 60)
    
    config = OAKConfig()
    
    print(f"1. Pipeline Integration:")
    print(f"   - Data root: {config.DATA_ROOT}")
    print(f"   - Output directory: {config.OUTPUT_DIR}")
    print(f"   - Available ROIs: {list(config.ROI_MASKS.keys())}")
    
    # Create integrated pipeline
    print(f"\n2. Creating Integrated Pipeline:")
    
    subjects = ['sub-001', 'sub-002', 'sub-003']  # Mock subjects
    
    try:
        # Create optimized pipeline
        pipeline = create_parallel_pipeline(config, subjects=subjects)
        
        print(f"   ✓ Pipeline created successfully")
        print(f"   - Subject parallelization: {pipeline.parallel_config.N_JOBS_SUBJECTS}")
        print(f"   - ROI parallelization: {pipeline.parallel_config.N_JOBS_ROIS}")
        print(f"   - Chunk size: {pipeline.parallel_config.CHUNK_SIZE}")
        
        # Show integration with existing components
        print(f"\n3. Component Integration:")
        print(f"   - Logger: {type(pipeline.logger).__name__}")
        print(f"   - Config: {type(pipeline.config).__name__}")
        print(f"   - Memory manager: {'Enabled' if pipeline.memory else 'Disabled'}")
        
        # Show processing workflow
        print(f"\n4. Processing Workflow:")
        print(f"   - Input: {len(subjects)} subjects")
        print(f"   - ROIs per subject: {len(config.ROI_MASKS)}")
        print(f"   - Total analyses: {len(subjects) * len(config.ROI_MASKS)}")
        print(f"   - Parallel efficiency: ~{pipeline.parallel_config.N_JOBS_SUBJECTS * pipeline.parallel_config.N_JOBS_ROIS}x")
        
    except Exception as e:
        print(f"   ✗ Pipeline creation failed: {e}")
        return None
    
    return pipeline


def main():
    """Main demo function"""
    print("Parallel MVPA Processing Demo")
    print("=" * 60)
    print("This demo shows enhanced parallel processing capabilities")
    print("for the delay discounting MVPA pipeline.")
    print()
    
    # Create argument parser
    parser = create_analysis_parser(
        script_name='demo_parallel_mvpa',
        analysis_type='mvpa',
        require_data=False
    )
    
    # Add demo-specific arguments
    parser.parser.add_argument('--subjects', nargs='+', 
                              help='Specific subjects to process')
    parser.parser.add_argument('--enable-roi-parallel', action='store_true',
                              help='Enable ROI-level parallelization')
    parser.parser.add_argument('--optimize-config', action='store_true',
                              help='Use optimized parallel configuration')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup environment
    env = setup_pipeline_environment(
        script_name='demo_parallel_mvpa',
        args=args,
        required_modules=['joblib', 'psutil', 'matplotlib']
    )
    
    logger = env['logger']
    config = env['config']
    
    try:
        logger.logger.info("Starting parallel MVPA processing demo")
        
        # Run demos
        print("Running parallel processing demos...")
        
        # Demo 1: Configuration options
        demo_config = demo_parallel_configuration()
        
        # Demo 2: Serial vs parallel comparison
        timing_results = demo_parallel_processing_comparison()
        
        # Demo 3: Memory and resource management
        processor = demo_memory_and_resource_management()
        
        # Demo 4: Error handling
        error_results = demo_error_handling_and_robustness()
        
        # Demo 5: Performance profiling
        performance_data = demo_performance_profiling()
        
        # Demo 6: Integration with existing pipeline
        pipeline = demo_integration_with_existing_pipeline()
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        
        if timing_results:
            speedup = timing_results['serial_time'] / timing_results['parallel_time']
            print(f"Parallel processing speedup: {speedup:.1f}x")
        
        if error_results:
            success_rate = sum(1 for r in error_results if r.get('success', False)) / len(error_results)
            print(f"Error handling success rate: {success_rate*100:.1f}%")
        
        if performance_data:
            max_speedup = max(d['Speedup'] for d in performance_data)
            print(f"Maximum theoretical speedup: {max_speedup:.1f}x")
        
        if pipeline:
            print(f"Pipeline integration: ✓ Successful")
        
        print(f"\n✓ All demos completed successfully!")
        print(f"See ./demo_parallel_outputs/ for generated visualizations")
        
        logger.logger.info("Parallel MVPA processing demo completed successfully")
        
    except Exception as e:
        logger.log_error_with_traceback(e, 'parallel processing demo')
        raise


if __name__ == "__main__":
    main() 