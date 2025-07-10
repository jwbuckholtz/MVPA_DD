#!/usr/bin/env python3
"""
Demo: Memory-Efficient Data Loading for fMRI Analysis
===================================================

This demo shows how memory-efficient data loading with numpy.memmap
can dramatically reduce memory usage during fMRI analysis, especially
during parallel processing.

Key Features Demonstrated:
- Memory mapping vs regular loading comparison
- Parallel processing memory benefits
- Integration with caching system
- ROI extraction with memory efficiency
- Performance benchmarks
- Memory usage monitoring

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

# Pipeline imports
from oak_storage_config import OAKConfig
from data_utils import get_complete_subjects, load_mask
from memory_efficient_data import (
    MemoryEfficientLoader, MemoryConfig, MemoryEfficientContext,
    create_memory_efficient_loader, MemoryMonitor
)
from caching_utils import create_cached_processor, CacheConfig
from logger_utils import setup_pipeline_logging, PerformanceTimer


class MemoryBenchmark:
    """Benchmark memory usage for different loading strategies"""
    
    def __init__(self, config: OAKConfig = None):
        self.config = config or OAKConfig()
        self.logger = setup_pipeline_logging('memory_benchmark')
        self.results = []
        
        # Setup temp directory for demo
        self.temp_dir = Path(tempfile.gettempdir()) / 'memory_demo'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory monitor
        self.monitor = MemoryMonitor()
        
        print("\n" + "="*70)
        print("MEMORY-EFFICIENT DATA LOADING DEMO")
        print("="*70)
        print(f"System Memory: {self.monitor.get_memory_info()['total_gb']:.1f} GB")
        print(f"Available Memory: {self.monitor.get_memory_info()['available_gb']:.1f} GB")
    
    def create_synthetic_fmri_data(self, subject_id: str, 
                                  shape: Tuple[int, int, int, int] = (64, 64, 64, 300),
                                  save_nifti: bool = True) -> str:
        """Create synthetic fMRI data for testing"""
        print(f"\nCreating synthetic fMRI data for {subject_id}")
        print(f"Shape: {shape} ({np.prod(shape) * 4 / (1024**3):.2f} GB)")
        
        # Generate synthetic data
        data = np.random.randn(*shape).astype(np.float32)
        
        # Add some structure (realistic fMRI patterns)
        for t in range(shape[3]):
            # Add temporal correlation
            if t > 0:
                data[..., t] = 0.3 * data[..., t-1] + 0.7 * data[..., t]
            
            # Add spatial structure
            center = [s//2 for s in shape[:3]]
            for i in range(3):
                coords = np.arange(shape[i])
                gaussian = np.exp(-0.001 * (coords - center[i])**2)
                data[..., t] *= gaussian[None, :, None] if i == 1 else gaussian
        
        if save_nifti:
            import nibabel as nib
            
            # Create NIfTI file
            nifti_file = self.temp_dir / f'{subject_id}_synthetic_fmri.nii.gz'
            
            # Create affine transformation
            affine = np.diag([2.0, 2.0, 2.0, 0.68])  # 2mm voxels, 0.68s TR
            
            # Save as NIfTI
            img = nib.Nifti1Image(data, affine)
            nib.save(img, str(nifti_file))
            
            print(f"Saved synthetic fMRI data: {nifti_file}")
            return str(nifti_file)
        
        return data
    
    def create_synthetic_roi_mask(self, shape: Tuple[int, int, int] = (64, 64, 64),
                                 roi_name: str = 'test_roi') -> str:
        """Create synthetic ROI mask"""
        print(f"\nCreating synthetic ROI mask: {roi_name}")
        
        # Create spherical ROI
        center = [s//2 for s in shape]
        radius = min(shape) // 6
        
        mask = np.zeros(shape, dtype=bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                    if dist <= radius:
                        mask[i, j, k] = True
        
        # Save mask
        import nibabel as nib
        mask_file = self.temp_dir / f'{roi_name}_mask.nii.gz'
        img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
        nib.save(img, str(mask_file))
        
        print(f"ROI mask created: {mask.sum()} voxels ({mask.sum()/np.prod(shape)*100:.1f}%)")
        return str(mask_file)
    
    def benchmark_loading_strategies(self, nifti_file: str, 
                                   n_repetitions: int = 3) -> Dict[str, Any]:
        """Benchmark different loading strategies"""
        print(f"\n{'='*50}")
        print("BENCHMARKING LOADING STRATEGIES")
        print(f"{'='*50}")
        
        import nibabel as nib
        
        results = {}
        
        # 1. Standard loading (baseline)
        print("\n1. Standard Loading (nibabel)")
        memory_before = self.monitor.get_memory_info()['process_memory_gb']
        
        times = []
        for i in range(n_repetitions):
            start_time = time.time()
            
            # Load data into memory
            img = nib.load(nifti_file)
            data = img.get_fdata()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Clean up
            del data, img
        
        memory_after = self.monitor.get_memory_info()['process_memory_gb']
        
        results['standard'] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'memory_used_gb': memory_after - memory_before,
            'strategy': 'Standard nibabel loading'
        }
        
        print(f"  Time: {results['standard']['mean_time']:.2f} ± {results['standard']['std_time']:.2f}s")
        print(f"  Memory: {results['standard']['memory_used_gb']:.2f} GB")
        
        # 2. Memory-mapped loading
        print("\n2. Memory-Mapped Loading")
        memory_before = self.monitor.get_memory_info()['process_memory_gb']
        
        memory_config = MemoryConfig()
        memory_config.USE_MEMMAP = True
        memory_config.MEMMAP_THRESHOLD_GB = 0.0  # Force memmap
        
        loader = create_memory_efficient_loader(self.config, memory_config)
        
        times = []
        for i in range(n_repetitions):
            start_time = time.time()
            
            # Create memory map
            memmap_array = loader.create_memmap_from_nifti(nifti_file, f'test_subject_{i}')
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Clean up
            memmap_array.cleanup()
        
        memory_after = self.monitor.get_memory_info()['process_memory_gb']
        
        results['memmap'] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'memory_used_gb': memory_after - memory_before,
            'strategy': 'Memory-mapped loading'
        }
        
        print(f"  Time: {results['memmap']['mean_time']:.2f} ± {results['memmap']['std_time']:.2f}s")
        print(f"  Memory: {results['memmap']['memory_used_gb']:.2f} GB")
        
        # 3. Memory-mapped with ROI extraction
        print("\n3. Memory-Mapped ROI Extraction")
        
        # Create ROI mask
        roi_mask_file = self.create_synthetic_roi_mask()
        roi_mask = nib.load(roi_mask_file).get_fdata().astype(bool)
        
        memory_before = self.monitor.get_memory_info()['process_memory_gb']
        
        times = []
        for i in range(n_repetitions):
            start_time = time.time()
            
            # Extract ROI time series using memory mapping
            timeseries = loader.extract_roi_timeseries_memmap(
                f'test_subject_{i}', roi_mask, standardize=True
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Clean up
            del timeseries
        
        memory_after = self.monitor.get_memory_info()['process_memory_gb']
        
        results['memmap_roi'] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'memory_used_gb': memory_after - memory_before,
            'strategy': 'Memory-mapped ROI extraction'
        }
        
        print(f"  Time: {results['memmap_roi']['mean_time']:.2f} ± {results['memmap_roi']['std_time']:.2f}s")
        print(f"  Memory: {results['memmap_roi']['memory_used_gb']:.2f} GB")
        
        # Cleanup
        loader.cleanup()
        
        return results
    
    def demonstrate_parallel_benefits(self, nifti_files: List[str], 
                                    n_workers: int = 4) -> Dict[str, Any]:
        """Demonstrate memory benefits during parallel processing"""
        print(f"\n{'='*50}")
        print("PARALLEL PROCESSING MEMORY BENEFITS")
        print(f"{'='*50}")
        
        results = {}
        
        # 1. Serial processing with standard loading
        print(f"\n1. Serial Processing (Standard Loading)")
        memory_before = self.monitor.get_memory_info()['process_memory_gb']
        
        start_time = time.time()
        
        for i, nifti_file in enumerate(nifti_files):
            print(f"  Processing file {i+1}/{len(nifti_files)}")
            
            # Simulate standard loading
            import nibabel as nib
            img = nib.load(nifti_file)
            data = img.get_fdata()
            
            # Simulate some processing
            mean_signal = np.mean(data)
            
            # Clean up
            del data, img
        
        end_time = time.time()
        memory_after = self.monitor.get_memory_info()['process_memory_gb']
        
        results['serial_standard'] = {
            'time': end_time - start_time,
            'memory_used_gb': memory_after - memory_before,
            'strategy': 'Serial standard loading'
        }
        
        print(f"  Time: {results['serial_standard']['time']:.2f}s")
        print(f"  Memory: {results['serial_standard']['memory_used_gb']:.2f} GB")
        
        # 2. Parallel processing with memory mapping
        print(f"\n2. Parallel Processing (Memory-Mapped)")
        memory_before = self.monitor.get_memory_info()['process_memory_gb']
        
        def process_file_memmap(nifti_file: str) -> Dict[str, Any]:
            """Process single file with memory mapping"""
            # Create memory-efficient loader
            memory_config = MemoryConfig()
            memory_config.USE_MEMMAP = True
            loader = create_memory_efficient_loader(self.config, memory_config)
            
            # Load with memory mapping
            memmap_array = loader.create_memmap_from_nifti(nifti_file, f'parallel_test')
            
            # Simulate processing
            mean_signal = np.mean(memmap_array.array[..., 0])
            
            # Get memory usage
            process_memory = psutil.Process().memory_info().rss / (1024**3)
            
            # Cleanup
            memmap_array.cleanup()
            loader.cleanup()
            
            return {
                'mean_signal': mean_signal,
                'process_memory_gb': process_memory,
                'file': nifti_file
            }
        
        start_time = time.time()
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_file_memmap, nifti_file) 
                      for nifti_file in nifti_files]
            
            parallel_results = []
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result)
                print(f"  Processed file: {Path(result['file']).name}")
        
        end_time = time.time()
        memory_after = self.monitor.get_memory_info()['process_memory_gb']
        
        results['parallel_memmap'] = {
            'time': end_time - start_time,
            'memory_used_gb': memory_after - memory_before,
            'strategy': 'Parallel memory-mapped loading',
            'n_workers': n_workers,
            'max_worker_memory_gb': max(r['process_memory_gb'] for r in parallel_results)
        }
        
        print(f"  Time: {results['parallel_memmap']['time']:.2f}s")
        print(f"  Memory: {results['parallel_memmap']['memory_used_gb']:.2f} GB")
        print(f"  Max worker memory: {results['parallel_memmap']['max_worker_memory_gb']:.2f} GB")
        
        # Calculate speedup
        if results['serial_standard']['time'] > 0:
            speedup = results['serial_standard']['time'] / results['parallel_memmap']['time']
            print(f"  Speedup: {speedup:.2f}x")
            results['parallel_memmap']['speedup'] = speedup
        
        return results
    
    def demonstrate_caching_integration(self, nifti_files: List[str]) -> Dict[str, Any]:
        """Demonstrate integration with caching system"""
        print(f"\n{'='*50}")
        print("CACHING SYSTEM INTEGRATION")
        print(f"{'='*50}")
        
        # Setup caching with memory efficiency
        cache_config = CacheConfig()
        cache_config.CACHE_DIR = str(self.temp_dir / 'cache')
        cache_config.MAX_CACHE_SIZE_GB = 10.0
        
        memory_config = MemoryConfig()
        memory_config.USE_MEMMAP = True
        memory_config.MEMMAP_TEMP_DIR = str(self.temp_dir / 'memmap')
        
        # Create cached processor
        cached_processor = create_cached_processor(
            self.config, cache_config, memory_config
        )
        
        results = {}
        
        # Test with synthetic behavioral data
        behavioral_data = pd.DataFrame({
            'onset': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            'duration': [2] * 10,
            'choice': ['smaller_sooner', 'larger_later'] * 5,
            'large_amount': [30, 40, 50, 60, 70] * 2,
            'later_delay': [7, 14, 30, 90, 180] * 2,
            'response_time': np.random.uniform(0.5, 3.0, 10)
        })
        
        print("\n1. First Run (No Cache)")
        start_time = time.time()
        
        # Process first file
        test_file = nifti_files[0]
        subject_id = 'test_subject_cache'
        
        # Create temporary behavioral data file
        behavioral_file = self.temp_dir / f'{subject_id}_behavioral.csv'
        behavioral_data.to_csv(behavioral_file, index=False)
        
        # Mock the path to point to our synthetic data
        original_fmri_path = test_file
        
        try:
            # This would normally call the cached processor
            # For demo, we'll simulate the memory-efficient extraction
            loader = cached_processor.memory_loader
            memmap_array = loader.create_memmap_from_nifti(original_fmri_path, subject_id)
            
            # Simulate ROI extraction
            roi_mask = self.create_synthetic_roi_mask()
            roi_mask_data = nib.load(roi_mask).get_fdata().astype(bool)
            
            timeseries = loader.extract_roi_timeseries_memmap(
                subject_id, roi_mask_data, standardize=True
            )
            
            end_time = time.time()
            
            results['first_run'] = {
                'time': end_time - start_time,
                'memory_used_gb': self.monitor.get_memory_info()['process_memory_gb'],
                'cache_hit': False,
                'timeseries_shape': timeseries.shape
            }
            
            print(f"  Time: {results['first_run']['time']:.2f}s")
            print(f"  Memory: {results['first_run']['memory_used_gb']:.2f} GB")
            print(f"  Timeseries shape: {results['first_run']['timeseries_shape']}")
            
        except Exception as e:
            print(f"  Error in first run: {e}")
            results['first_run'] = {'error': str(e)}
        
        # Get memory usage report
        memory_report = cached_processor.get_memory_usage_report()
        
        print("\n2. Memory Usage Report")
        print(f"  Active memory maps: {memory_report['memory_stats']['active_memmaps']}")
        print(f"  Memory map usage: {memory_report['memory_stats']['memmap_usage_gb']:.2f} GB")
        print(f"  Temp directory size: {memory_report['memory_stats']['temp_dir_size_gb']:.2f} GB")
        
        # Cleanup
        cached_processor.cleanup()
        
        results['memory_report'] = memory_report
        
        return results
    
    def plot_benchmark_results(self, results: Dict[str, Any], 
                             save_path: Optional[str] = None) -> None:
        """Plot benchmark results"""
        print(f"\n{'='*50}")
        print("PLOTTING BENCHMARK RESULTS")
        print(f"{'='*50}")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Loading strategy comparison
        if 'loading_benchmark' in results:
            strategies = []
            times = []
            memory_usage = []
            
            for key, data in results['loading_benchmark'].items():
                strategies.append(data['strategy'])
                times.append(data['mean_time'])
                memory_usage.append(data['memory_used_gb'])
            
            # Time comparison
            ax1.bar(strategies, times, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax1.set_title('Loading Time Comparison')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Memory comparison
            ax2.bar(strategies, memory_usage, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax2.set_title('Memory Usage Comparison')
            ax2.set_ylabel('Memory (GB)')
            ax2.tick_params(axis='x', rotation=45)
        
        # 2. Parallel processing benefits
        if 'parallel_benchmark' in results:
            categories = []
            times = []
            memory_usage = []
            
            for key, data in results['parallel_benchmark'].items():
                categories.append(data['strategy'])
                times.append(data['time'])
                memory_usage.append(data['memory_used_gb'])
            
            # Time comparison
            ax3.bar(categories, times, color=['orange', 'green'])
            ax3.set_title('Serial vs Parallel Processing')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Memory comparison
            ax4.bar(categories, memory_usage, color=['orange', 'green'])
            ax4.set_title('Memory Usage: Serial vs Parallel')
            ax4.set_ylabel('Memory (GB)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Plot saved: {save_path}")
        
        plt.show()
    
    def run_complete_demo(self, n_synthetic_files: int = 4) -> Dict[str, Any]:
        """Run complete memory efficiency demo"""
        print(f"\n{'='*70}")
        print("RUNNING COMPLETE MEMORY EFFICIENCY DEMO")
        print(f"{'='*70}")
        
        all_results = {}
        
        # 1. Create synthetic data
        print("\n🔄 Creating synthetic fMRI data...")
        nifti_files = []
        
        for i in range(n_synthetic_files):
            subject_id = f'synthetic_sub_{i:03d}'
            
            # Vary data sizes for realistic testing
            if i == 0:
                shape = (48, 48, 40, 200)  # Smaller dataset
            elif i == 1:
                shape = (64, 64, 50, 250)  # Medium dataset
            else:
                shape = (80, 80, 60, 300)  # Larger dataset
            
            nifti_file = self.create_synthetic_fmri_data(subject_id, shape)
            nifti_files.append(nifti_file)
        
        # 2. Benchmark loading strategies
        print("\n🧪 Benchmarking loading strategies...")
        loading_results = self.benchmark_loading_strategies(nifti_files[0])
        all_results['loading_benchmark'] = loading_results
        
        # 3. Demonstrate parallel benefits
        print("\n⚡ Demonstrating parallel processing benefits...")
        parallel_results = self.demonstrate_parallel_benefits(nifti_files)
        all_results['parallel_benchmark'] = parallel_results
        
        # 4. Show caching integration
        print("\n💾 Demonstrating caching integration...")
        caching_results = self.demonstrate_caching_integration(nifti_files[:2])
        all_results['caching_integration'] = caching_results
        
        # 5. Generate summary report
        print("\n📊 Generating summary report...")
        summary = self.generate_summary_report(all_results)
        all_results['summary'] = summary
        
        # 6. Plot results
        print("\n📈 Plotting results...")
        plot_path = self.temp_dir / 'memory_efficiency_benchmark.png'
        self.plot_benchmark_results(all_results, str(plot_path))
        
        # 7. Cleanup
        print("\n🧹 Cleaning up...")
        self.cleanup()
        
        return all_results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        print(f"\n{'='*50}")
        print("SUMMARY REPORT")
        print(f"{'='*50}")
        
        summary = {
            'system_info': self.monitor.get_memory_info(),
            'recommendations': [],
            'key_findings': []
        }
        
        # Analyze loading strategies
        if 'loading_benchmark' in results:
            loading_data = results['loading_benchmark']
            
            # Find most efficient strategy
            best_memory = min(loading_data.values(), key=lambda x: x['memory_used_gb'])
            best_time = min(loading_data.values(), key=lambda x: x['mean_time'])
            
            summary['key_findings'].append(
                f"Memory mapping reduces memory usage by "
                f"{(loading_data['standard']['memory_used_gb'] - loading_data['memmap']['memory_used_gb']):.2f} GB"
            )
            
            if loading_data['memmap']['mean_time'] < loading_data['standard']['mean_time']:
                summary['key_findings'].append(
                    f"Memory mapping is also faster by "
                    f"{(loading_data['standard']['mean_time'] - loading_data['memmap']['mean_time']):.2f}s"
                )
        
        # Analyze parallel processing
        if 'parallel_benchmark' in results:
            parallel_data = results['parallel_benchmark']
            
            if 'speedup' in parallel_data.get('parallel_memmap', {}):
                speedup = parallel_data['parallel_memmap']['speedup']
                summary['key_findings'].append(
                    f"Parallel processing with memory mapping achieves {speedup:.2f}x speedup"
                )
        
        # Generate recommendations
        system_memory = summary['system_info']['total_gb']
        
        if system_memory < 16:
            summary['recommendations'].append(
                "Low memory system detected - use aggressive memory mapping (threshold: 0.5GB)"
            )
        elif system_memory < 64:
            summary['recommendations'].append(
                "Medium memory system - use moderate memory mapping (threshold: 1.0GB)"
            )
        else:
            summary['recommendations'].append(
                "High memory system - use conservative memory mapping (threshold: 2.0GB)"
            )
        
        summary['recommendations'].extend([
            "Use memory mapping for datasets larger than 1GB",
            "Enable memory mapping for all parallel processing",
            "Combine with caching for maximum efficiency",
            "Monitor memory usage during long-running analyses"
        ])
        
        # Print summary
        print("\n📋 Key Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
        
        return summary
    
    def cleanup(self):
        """Clean up temporary files and directories"""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"  Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"  Warning: Could not clean up {self.temp_dir}: {e}")


def main():
    """Run the memory efficiency demonstration"""
    print("Memory-Efficient fMRI Data Loading Demo")
    print("=" * 70)
    
    # Initialize configuration
    config = OAKConfig()
    
    # Create benchmark instance
    benchmark = MemoryBenchmark(config)
    
    # Run complete demo
    try:
        results = benchmark.run_complete_demo(n_synthetic_files=4)
        
        print("\n" + "="*70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Save results
        import pickle
        results_file = Path('memory_efficiency_demo_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Ensure cleanup
        benchmark.cleanup()


if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n✅ Demo completed successfully!")
        print("Key benefits of memory-efficient loading:")
        print("  • Reduced memory usage during fMRI processing")
        print("  • Faster parallel processing with shared memory maps")
        print("  • Integration with caching for maximum efficiency")
        print("  • Scalable to large datasets and HPC environments")
    else:
        print("\n❌ Demo failed - check error messages above") 