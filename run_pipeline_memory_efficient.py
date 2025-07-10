#!/usr/bin/env python3
"""
Memory-Efficient MVPA Pipeline Runner
====================================

Command-line interface for running the delay discounting MVPA pipeline
with memory-efficient data loading options.

Usage Examples:
  # Run with memory efficiency enabled
  python run_pipeline_memory_efficient.py --memory-efficient

  # Run with custom memory threshold
  python run_pipeline_memory_efficient.py --memory-efficient --memory-threshold 0.5

  # Run with parallel processing and memory efficiency
  python run_pipeline_memory_efficient.py --memory-efficient --parallel --n-jobs 8

  # Run cached pipeline with memory efficiency
  python run_pipeline_memory_efficient.py --cached --memory-efficient

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Run delay discounting MVPA pipeline with memory-efficient options",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Pipeline options
    parser.add_argument('--cached', action='store_true',
                       help='Use cached pipeline (recommended)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--standard', action='store_true',
                       help='Use standard (non-cached) pipeline')
    
    # Memory efficiency options
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Enable memory-efficient data loading')
    parser.add_argument('--memory-threshold', type=float, default=1.0,
                       help='Memory mapping threshold in GB (default: 1.0)')
    parser.add_argument('--memory-buffer', type=float, default=0.2,
                       help='Memory buffer fraction (default: 0.2)')
    parser.add_argument('--max-memory-per-process', type=float, default=8.0,
                       help='Max memory per process in GB (default: 8.0)')
    parser.add_argument('--force-memmap', action='store_true',
                       help='Force memory mapping for all data')
    parser.add_argument('--disable-shared-memory', action='store_true',
                       help='Disable shared memory for parallel processing')
    
    # Processing options
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1)')
    parser.add_argument('--subjects', nargs='+',
                       help='Specific subjects to process (default: all)')
    parser.add_argument('--rois', nargs='+',
                       help='Specific ROIs to analyze (default: all)')
    
    # Cache options
    parser.add_argument('--cache-dir', type=str,
                       help='Cache directory (default: ./cache)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache before running')
    parser.add_argument('--cache-stats', action='store_true',
                       help='Show cache statistics')
    
    # Output options
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: from config)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-memory', action='store_true',
                       help='Enable memory usage logging')
    
    # HPC options
    parser.add_argument('--slurm', action='store_true',
                       help='Configure for SLURM environment')
    parser.add_argument('--auto-configure', action='store_true',
                       help='Auto-configure based on system resources')
    
    return parser

def configure_memory_efficiency(args):
    """Configure memory efficiency settings"""
    from memory_efficient_data import MemoryConfig
    
    memory_config = MemoryConfig()
    
    if args.memory_efficient:
        memory_config.USE_MEMMAP = True
        memory_config.MEMMAP_THRESHOLD_GB = args.memory_threshold
        memory_config.AVAILABLE_MEMORY_BUFFER = args.memory_buffer
        memory_config.MAX_MEMORY_PER_PROCESS_GB = args.max_memory_per_process
        memory_config.SHARED_MEMORY_PARALLEL = not args.disable_shared_memory
        memory_config.MONITOR_MEMORY_USAGE = args.log_memory
        
        if args.force_memmap:
            memory_config.MEMMAP_THRESHOLD_GB = 0.0
        
        print(f"🚀 Memory-efficient loading configured:")
        print(f"   Threshold: {memory_config.MEMMAP_THRESHOLD_GB} GB")
        print(f"   Memory buffer: {memory_config.AVAILABLE_MEMORY_BUFFER * 100:.0f}%")
        print(f"   Max memory per process: {memory_config.MAX_MEMORY_PER_PROCESS_GB} GB")
        print(f"   Shared memory: {'enabled' if memory_config.SHARED_MEMORY_PARALLEL else 'disabled'}")
    else:
        memory_config.USE_MEMMAP = False
        print("📊 Standard data loading enabled")
    
    return memory_config

def configure_for_slurm(memory_config):
    """Configure for SLURM environment"""
    print("🖥️ Configuring for SLURM environment...")
    
    # Get SLURM environment variables
    mem_per_node = int(os.environ.get('SLURM_MEM_PER_NODE', '32000'))  # MB
    n_tasks = int(os.environ.get('SLURM_NTASKS_PER_NODE', '16'))
    job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
    
    print(f"   SLURM Job ID: {job_id}")
    print(f"   Memory per node: {mem_per_node} MB")
    print(f"   Tasks per node: {n_tasks}")
    
    # Calculate memory per process
    mem_per_process_gb = (mem_per_node / 1024) / n_tasks * 0.8  # 80% utilization
    
    # Configure memory settings
    memory_config.MAX_MEMORY_PER_PROCESS_GB = mem_per_process_gb
    memory_config.MEMMAP_THRESHOLD_GB = min(1.0, mem_per_process_gb / 4)
    memory_config.SHARED_MEMORY_PARALLEL = True
    
    print(f"   Configured memory per process: {mem_per_process_gb:.1f} GB")
    print(f"   Memory mapping threshold: {memory_config.MEMMAP_THRESHOLD_GB} GB")
    
    return memory_config

def auto_configure_resources(memory_config, n_jobs):
    """Auto-configure based on system resources"""
    import psutil
    
    print("🔧 Auto-configuring based on system resources...")
    
    # Get system info
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"   System memory: {system_memory_gb:.1f} GB")
    print(f"   Available memory: {available_memory_gb:.1f} GB")
    print(f"   CPU cores: {cpu_count}")
    
    # Configure based on available resources
    if system_memory_gb < 16:
        # Low memory system
        memory_config.MEMMAP_THRESHOLD_GB = 0.5
        memory_config.AVAILABLE_MEMORY_BUFFER = 0.3
        memory_config.MAX_MEMORY_PER_PROCESS_GB = min(4.0, available_memory_gb / max(n_jobs, 2))
        print("   Low memory system detected - aggressive memory mapping")
    elif system_memory_gb < 64:
        # Medium memory system
        memory_config.MEMMAP_THRESHOLD_GB = 1.0
        memory_config.AVAILABLE_MEMORY_BUFFER = 0.2
        memory_config.MAX_MEMORY_PER_PROCESS_GB = min(8.0, available_memory_gb / max(n_jobs, 2))
        print("   Medium memory system detected - moderate memory mapping")
    else:
        # High memory system
        memory_config.MEMMAP_THRESHOLD_GB = 2.0
        memory_config.AVAILABLE_MEMORY_BUFFER = 0.15
        memory_config.MAX_MEMORY_PER_PROCESS_GB = min(16.0, available_memory_gb / max(n_jobs, 2))
        print("   High memory system detected - conservative memory mapping")
    
    return memory_config

def run_standard_pipeline(args, memory_config):
    """Run standard (non-cached) pipeline"""
    print("\n📊 Running standard MVPA pipeline...")
    
    from delay_discounting_mvpa_pipeline import main
    
    # Run with memory efficiency option
    results = main(
        enable_memory_efficient=args.memory_efficient,
        memory_config=memory_config
    )
    
    return results

def run_cached_pipeline(args, memory_config):
    """Run cached pipeline"""
    print("\n💾 Running cached MVPA pipeline...")
    
    from delay_discounting_mvpa_pipeline_cached import CachedMVPAPipeline
    from caching_utils import CacheConfig
    from oak_storage_config import OAKConfig
    
    # Setup configurations
    config = OAKConfig()
    
    cache_config = CacheConfig()
    if args.cache_dir:
        cache_config.CACHE_DIR = args.cache_dir
    if args.clear_cache:
        cache_config.CLEAR_ON_START = True
    
    # Create cached pipeline
    pipeline = CachedMVPAPipeline(
        config=config,
        cache_config=cache_config,
        memory_config=memory_config
    )
    
    # Show cache stats if requested
    if args.cache_stats:
        print("\n📈 Cache Statistics:")
        stats = pipeline.cached_processor.cache_manager.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Run analysis
    if args.subjects:
        subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in args.subjects]
        results = pipeline.run_analysis(subjects)
    else:
        results = pipeline.run_analysis()
    
    # Show memory usage report
    if args.log_memory:
        print("\n🔧 Memory Usage Report:")
        memory_report = pipeline.cached_processor.get_memory_usage_report()
        for key, value in memory_report.items():
            print(f"   {key}: {value}")
    
    return results

def run_parallel_pipeline(args, memory_config):
    """Run parallel pipeline"""
    print(f"\n⚡ Running parallel MVPA pipeline with {args.n_jobs} workers...")
    
    from parallel_mvpa_utils import ParallelMVPAProcessor, ParallelMVPAConfig
    from oak_storage_config import OAKConfig
    
    # Setup configurations
    config = OAKConfig()
    
    parallel_config = ParallelMVPAConfig()
    parallel_config.N_JOBS_SUBJECTS = args.n_jobs
    parallel_config.N_JOBS_ROIS = min(4, args.n_jobs)  # Conservative for ROIs
    parallel_config.ENABLE_NESTED_PARALLEL = True
    
    # Create parallel processor
    processor = ParallelMVPAProcessor(config, parallel_config)
    
    # Get subjects
    if args.subjects:
        subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in args.subjects]
    else:
        from data_utils import get_complete_subjects
        subjects = get_complete_subjects(config)
    
    # Run parallel processing
    results = processor.process_subjects_parallel(subjects)
    
    return results

def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("="*70)
    print("MEMORY-EFFICIENT MVPA PIPELINE")
    print("="*70)
    
    # Configure memory efficiency
    memory_config = configure_memory_efficiency(args)
    
    # Auto-configure if requested
    if args.auto_configure:
        memory_config = auto_configure_resources(memory_config, args.n_jobs)
    
    # Configure for SLURM if requested
    if args.slurm:
        memory_config = configure_for_slurm(memory_config)
    
    # Run appropriate pipeline
    try:
        if args.parallel:
            results = run_parallel_pipeline(args, memory_config)
        elif args.cached:
            results = run_cached_pipeline(args, memory_config)
        else:
            results = run_standard_pipeline(args, memory_config)
        
        print("\n✅ Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 