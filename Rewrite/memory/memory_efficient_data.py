#!/usr/bin/env python3
"""
Memory-Efficient Data Loading for MVPA Pipeline
===============================================

This module provides memory-efficient data loading using numpy.memmap to handle
large fMRI datasets without causing memory spikes during parallel processing.

Key Features:
- Memory-mapped fMRI data loading
- Shared memory for parallel processing
- Automatic memory usage monitoring
- Integration with caching system
- Configurable memory management strategies

Author: Cognitive Neuroscience Lab, Stanford University
"""

import hashlib
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import psutil
from nibabel.arrayproxy import ArrayProxy

# Import data utilities
from data_utils import get_fmri_path, load_behavioral_data
from logger_utils import PipelineLogger

# Import configuration and logging
from oak_storage_config import OAKConfig


class MemoryConfigOLD:
    """Configuration for memory-efficient data loading"""

    # Memory mapping settings
    USE_MEMMAP = True
    MEMMAP_MODE = 'r'  # Read-only by default
    MEMMAP_TEMP_DIR = None  # Use system temp if None

    # Memory thresholds (in GB)
    MEMMAP_THRESHOLD_GB = 1.0  # Use memmap for data larger than 1GB
    AVAILABLE_MEMORY_BUFFER = 0.2  # Keep 20% memory free

    # Parallel processing
    SHARED_MEMORY_PARALLEL = True  # Share memory-mapped files between processes
    MAX_MEMORY_PER_PROCESS_GB = 8.0  # Maximum memory per parallel process

    # Caching integration
    CACHE_MEMMAP_FILES = True  # Cache memory-mapped files
    CLEANUP_TEMP_FILES = True  # Cleanup temporary files

    # Performance monitoring
    MONITOR_MEMORY_USAGE = True
    LOG_MEMORY_STATS = True


from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """
    Configuration for memory-efficient data loading.

    Fields:
        use_memmap: Whether to use memory mapping.
        memmap_mode: Mode for numpy.memmap (e.g., 'r', 'r+', etc.).
        memmap_temp_dir: Directory for temporary memmap files.
        memmap_threshold_gb: File size (in GB) above which memmap is used.
        available_memory_buffer: Fraction of system memory to keep free.
        shared_memory_parallel: Whether to share memmaps between parallel processes.
        max_memory_per_process_gb: Cap memory use per process in GB.
        cache_memmap_files: Whether to keep memmap files cached.
        cleanup_temp_files: Whether to delete temp files after use.
        monitor_memory_usage: Enable memory monitoring.
        log_memory_stats: Log memory usage to logger.
    """

    use_memmap: bool = True
    memmap_mode: str = 'r'
    memmap_temp_dir: Optional[str] = None

    memmap_threshold_gb: float = 1.0
    available_memory_buffer: float = 0.2

    shared_memory_parallel: bool = True
    max_memory_per_process_gb: float = 8.0

    cache_memmap_files: bool = True
    cleanup_temp_files: bool = True

    monitor_memory_usage: bool = True
    log_memory_stats: bool = True


class MemoryMonitor:
    """Monitor system memory usage"""

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.logger = PipelineLogger('memory_monitor').logger
        self.initial_memory = self.get_memory_info()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        process = psutil.Process()

        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'process_memory_gb': process.memory_info().rss / (1024**3),
            'process_percent': process.memory_percent(),
        }

    def check_memory_threshold(self, required_gb: float) -> bool:
        """Check if we have enough memory available"""
        memory_info = self.get_memory_info()
        available_with_buffer = memory_info['available_gb'] * (
            1 - self.config.AVAILABLE_MEMORY_BUFFER
        )

        return required_gb <= available_with_buffer

    def suggest_memmap(self, data_size_gb: float) -> bool:
        """Suggest whether to use memory mapping"""
        # Use memmap if data is large or memory is limited
        return (
            data_size_gb > self.config.MEMMAP_THRESHOLD_GB
            or not self.check_memory_threshold(data_size_gb)
        )

    def log_memory_usage(self, operation: str = ''):
        """Log current memory usage"""
        if not self.config.LOG_MEMORY_STATS:
            return

        memory_info = self.get_memory_info()
        self.logger.info(
            f'Memory usage {operation}: {memory_info["process_memory_gb"]:.2f} GB '
            f'({memory_info["process_percent"]:.1f}% of system), '
            f'System: {memory_info["used_gb"]:.2f}/{memory_info["total_gb"]:.2f} GB '
            f'({memory_info["percent"]:.1f}%)'
        )


class MemoryMappedArray:
    """Wrapper for memory-mapped arrays with metadata"""

    def __init__(
        self,
        file_path: Union[str, Path],
        shape: Tuple[int, ...],
        dtype: np.dtype,
        mode: str = 'r',
        temp_file: bool = False,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize memory-mapped array

        Parameters:
        -----------
        file_path : str or Path
            Path to memory-mapped file
        shape : tuple
            Array shape
        dtype : np.dtype
            Array data type
        mode : str
            File access mode ('r', 'r+', 'w+', 'c')
        temp_file : bool
            Whether this is a temporary file
        metadata : dict
            Additional metadata
        """
        self.file_path = Path(file_path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.temp_file = temp_file
        self.metadata = metadata or {}

        # Create memory-mapped array
        self._memmap = np.memmap(
            str(self.file_path), dtype=dtype, mode=mode, shape=shape
        )

    @property
    def array(self) -> np.memmap:
        """Get the memory-mapped array"""
        return self._memmap

    @property
    def size_gb(self) -> float:
        """Get size in GB"""
        return self._memmap.nbytes / (1024**3)

    def __getitem__(self, key):
        """Array indexing"""
        return self._memmap[key]

    def __setitem__(self, key, value):
        """Array assignment"""
        if self.mode in ['r+', 'w+', 'c']:
            self._memmap[key] = value
        else:
            raise ValueError(f'Cannot write to read-only memory map (mode={self.mode})')

    def flush(self):
        """Flush changes to disk"""
        if hasattr(self._memmap, 'flush'):
            self._memmap.flush()

    def close(self):
        """Close memory-mapped file"""
        if hasattr(self._memmap, '_mmap'):
            self._memmap._mmap.close()
        del self._memmap

    def cleanup(self):
        """Cleanup temporary files"""
        if self.temp_file and self.file_path.exists():
            try:
                self.close()
                self.file_path.unlink()
            except Exception as e:
                PipelineLogger('memory_mapped_array').logger.warning(
                    f'Failed to cleanup temp file {self.file_path}: {e}'
                )


class MemoryEfficientLoader:
    """Memory-efficient data loader using memory mapping"""

    def __init__(self, config: OAKConfig = None, memory_config: MemoryConfig = None):
        """Initialize memory-efficient loader"""
        self.config = config or OAKConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.logger = PipelineLogger('memory_efficient_loader').logger
        self.monitor = MemoryMonitor(self.memory_config)

        # Setup temp directory for memory-mapped files
        if self.memory_config.MEMMAP_TEMP_DIR:
            self.temp_dir = Path(self.memory_config.MEMMAP_TEMP_DIR)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / 'mvpa_memmap'

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Track active memory maps for cleanup
        self.active_memmaps = []

        self.logger.info(
            f'Memory-efficient loader initialized. Temp dir: {self.temp_dir}'
        )
        self.monitor.log_memory_usage('initialization')

    def estimate_fmri_size(self, subject_id: str) -> float:
        """Estimate fMRI data size in GB"""
        try:
            fmri_path = get_fmri_path(subject_id, self.config)
            img = nib.load(fmri_path)
            # Estimate without loading data
            if hasattr(img, 'header'):
                shape = img.header.get_data_shape()
                dtype = img.header.get_data_dtype()
                size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
                return size_bytes / (1024**3)
            # Fallback: assume typical size
            return 2.0  # 2GB estimate
        except Exception as e:
            self.logger.warning(f'Could not estimate size for {subject_id}: {e}')
            return 2.0  # Default estimate

    def create_memmap_from_nifti(
        self, nifti_path: Union[str, Path], subject_id: str = None
    ) -> MemoryMappedArray:
        """
        Create memory-mapped array from NIfTI file

        Parameters:
        -----------
        nifti_path : str or Path
            Path to NIfTI file
        subject_id : str, optional
            Subject identifier for naming

        Returns:
        --------
        MemoryMappedArray : Memory-mapped array wrapper
        """
        nifti_path = Path(nifti_path)

        # Load NIfTI header to get shape and dtype
        img = nib.load(str(nifti_path))
        shape = img.header.get_data_shape()
        dtype = img.header.get_data_dtype()

        # Estimate size
        size_gb = np.prod(shape) * np.dtype(dtype).itemsize / (1024**3)

        # Create unique filename for memory map
        if subject_id:
            filename = f'fmri_{subject_id}_{hashlib.md5(str(nifti_path).encode()).hexdigest()[:8]}.dat'
        else:
            filename = (
                f'fmri_{hashlib.md5(str(nifti_path).encode()).hexdigest()[:8]}.dat'
            )

        memmap_path = self.temp_dir / filename

        # Check if memory map already exists
        if memmap_path.exists():
            self.logger.info(f'Using existing memory map: {memmap_path}')
            memmap_array = MemoryMappedArray(
                memmap_path,
                shape,
                dtype,
                mode='r',
                temp_file=True,
                metadata={'source': str(nifti_path), 'subject_id': subject_id},
            )
        else:
            self.logger.info(f'Creating memory map for {nifti_path} ({size_gb:.2f} GB)')

            # Create memory-mapped file
            memmap_array = MemoryMappedArray(
                memmap_path,
                shape,
                dtype,
                mode='w+',
                temp_file=True,
                metadata={'source': str(nifti_path), 'subject_id': subject_id},
            )

            # Copy data from NIfTI to memory map in chunks
            self._copy_nifti_to_memmap(img, memmap_array)

            # Switch to read-only mode
            memmap_array.flush()
            memmap_array.close()
            memmap_array = MemoryMappedArray(
                memmap_path,
                shape,
                dtype,
                mode='r',
                temp_file=True,
                metadata={'source': str(nifti_path), 'subject_id': subject_id},
            )

        # Track for cleanup
        self.active_memmaps.append(memmap_array)

        self.monitor.log_memory_usage(f'after creating memmap for {nifti_path.name}')

        return memmap_array

    def _copy_nifti_to_memmap(
        self,
        nifti_img: nib.Nifti1Image,
        memmap_array: MemoryMappedArray,
        chunk_size: int = 1000000,
    ):  # 1M voxels per chunk
        """Copy NIfTI data to memory map in chunks"""
        shape = nifti_img.header.get_data_shape()

        if len(shape) == 4:
            # 4D data - process volume by volume
            for t in range(shape[3]):
                volume = nifti_img.dataobj[..., t]
                memmap_array.array[..., t] = volume

                if t % 10 == 0:  # Log progress every 10 volumes
                    self.logger.debug(f'Copied volume {t + 1}/{shape[3]}')
        else:
            # 3D data - copy directly
            memmap_array.array[:] = nifti_img.get_fdata()

        memmap_array.flush()

    def load_fmri_memmap(
        self, subject_id: str, force_memmap: bool = False
    ) -> Union[MemoryMappedArray, np.ndarray]:
        """
        Load fMRI data using memory mapping when beneficial

        Parameters:
        -----------
        subject_id : str
            Subject identifier
        force_memmap : bool
            Force use of memory mapping regardless of size

        Returns:
        --------
        Union[MemoryMappedArray, np.ndarray] : fMRI data
        """
        # Get fMRI file path
        fmri_path = get_fmri_path(subject_id, self.config)

        # Estimate data size
        size_gb = self.estimate_fmri_size(subject_id)

        # Decide whether to use memory mapping
        use_memmap = (
            force_memmap
            or self.memory_config.USE_MEMMAP
            and self.monitor.suggest_memmap(size_gb)
        )

        if use_memmap:
            self.logger.info(
                f'Loading {subject_id} using memory mapping ({size_gb:.2f} GB)'
            )
            return self.create_memmap_from_nifti(fmri_path, subject_id)
        self.logger.info(f'Loading {subject_id} into memory ({size_gb:.2f} GB)')
        img = nib.load(fmri_path)
        self.monitor.log_memory_usage(f'after loading {subject_id}')
        return img.get_fdata()

    def extract_roi_timeseries_memmap(
        self, subject_id: str, roi_mask: np.ndarray, standardize: bool = True
    ) -> np.ndarray:
        """
        Extract ROI time series from memory-mapped fMRI data

        Parameters:
        -----------
        subject_id : str
            Subject identifier
        roi_mask : np.ndarray
            ROI mask (3D boolean array)
        standardize : bool
            Whether to standardize time series

        Returns:
        --------
        np.ndarray : ROI time series (n_timepoints x n_voxels)
        """
        # Load fMRI data (memory-mapped if beneficial)
        fmri_data = self.load_fmri_memmap(subject_id)

        if isinstance(fmri_data, MemoryMappedArray):
            # Extract ROI data from memory map without loading entire volume
            if len(fmri_data.shape) == 4:
                # Apply mask to get ROI indices
                roi_indices = np.where(roi_mask.flatten())[0]
                n_timepoints = fmri_data.shape[3]
                n_voxels = len(roi_indices)

                # Extract time series for ROI voxels
                timeseries = np.zeros((n_timepoints, n_voxels), dtype=fmri_data.dtype)

                for t in range(n_timepoints):
                    volume = fmri_data.array[..., t].flatten()
                    timeseries[t, :] = volume[roi_indices]
            else:
                # 3D data
                roi_data = fmri_data.array[roi_mask]
                timeseries = roi_data.reshape(1, -1)
        else:
            # Regular array
            if len(fmri_data.shape) == 4:
                roi_data = fmri_data[roi_mask, :]
                timeseries = roi_data.T
            else:
                roi_data = fmri_data[roi_mask]
                timeseries = roi_data.reshape(1, -1)

        # Standardize if requested
        if standardize and timeseries.shape[0] > 1:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            timeseries = scaler.fit_transform(timeseries)

        self.monitor.log_memory_usage(f'after extracting ROI for {subject_id}')

        return timeseries

    def create_shared_memmap(
        self, data: np.ndarray, identifier: str
    ) -> MemoryMappedArray:
        """
        Create shared memory map for parallel processing

        Parameters:
        -----------
        data : np.ndarray
            Data to store in shared memory
        identifier : str
            Unique identifier for the shared memory

        Returns:
        --------
        MemoryMappedArray : Shared memory-mapped array
        """
        # Create unique filename
        filename = f'shared_{identifier}_{hashlib.md5(identifier.encode()).hexdigest()[:8]}.dat'
        memmap_path = self.temp_dir / filename

        # Create memory map
        memmap_array = MemoryMappedArray(
            memmap_path,
            data.shape,
            data.dtype,
            mode='w+',
            temp_file=True,
            metadata={'identifier': identifier, 'shared': True},
        )

        # Copy data
        memmap_array.array[:] = data
        memmap_array.flush()

        # Switch to read-only for sharing
        memmap_array.close()
        memmap_array = MemoryMappedArray(
            memmap_path,
            data.shape,
            data.dtype,
            mode='r',
            temp_file=True,
            metadata={'identifier': identifier, 'shared': True},
        )

        self.active_memmaps.append(memmap_array)

        self.logger.info(
            f'Created shared memory map: {identifier} ({memmap_array.size_gb:.2f} GB)'
        )

        return memmap_array

    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        memory_info = self.monitor.get_memory_info()

        # Calculate memory usage by active memory maps
        memmap_usage = sum(mmap.size_gb for mmap in self.active_memmaps)

        return {
            'system_memory': memory_info,
            'active_memmaps': len(self.active_memmaps),
            'memmap_usage_gb': memmap_usage,
            'temp_dir': str(self.temp_dir),
            'temp_dir_size_gb': self._get_dir_size_gb(self.temp_dir),
        }

    def _get_dir_size_gb(self, directory: Path) -> float:
        """Get directory size in GB"""
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob('*') if f.is_file()
            )
            return total_size / (1024**3)
        except Exception:
            return 0.0

    def cleanup(self):
        """Cleanup all memory maps and temporary files"""
        self.logger.info('Cleaning up memory-mapped files...')

        # Close and cleanup active memory maps
        for memmap_array in self.active_memmaps:
            try:
                memmap_array.cleanup()
            except Exception as e:
                self.logger.warning(f'Failed to cleanup memory map: {e}')

        self.active_memmaps.clear()

        # Clean up temp directory if requested
        if self.memory_config.CLEANUP_TEMP_FILES:
            try:
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    self.logger.info(f'Cleaned up temp directory: {self.temp_dir}')
            except Exception as e:
                self.logger.warning(f'Failed to cleanup temp directory: {e}')

    def __del__(self):
        """Destructor - cleanup memory maps"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction


def create_memory_efficient_loader(
    config: OAKConfig = None, memory_config: MemoryConfig = None
) -> MemoryEfficientLoader:
    """Create memory-efficient loader with default configuration"""
    if config is None:
        config = OAKConfig()

    if memory_config is None:
        memory_config = MemoryConfig()

        # Auto-configure based on system memory
        system_memory_gb = psutil.virtual_memory().total / (1024**3)

        if system_memory_gb < 16:
            # Low memory system - aggressive memory mapping
            memory_config.MEMMAP_THRESHOLD_GB = 0.5
            memory_config.AVAILABLE_MEMORY_BUFFER = 0.3
            memory_config.MAX_MEMORY_PER_PROCESS_GB = 4.0
        elif system_memory_gb < 64:
            # Medium memory system - moderate memory mapping
            memory_config.MEMMAP_THRESHOLD_GB = 1.0
            memory_config.AVAILABLE_MEMORY_BUFFER = 0.2
            memory_config.MAX_MEMORY_PER_PROCESS_GB = 8.0
        else:
            # High memory system - conservative memory mapping
            memory_config.MEMMAP_THRESHOLD_GB = 2.0
            memory_config.AVAILABLE_MEMORY_BUFFER = 0.15
            memory_config.MAX_MEMORY_PER_PROCESS_GB = 16.0

    return MemoryEfficientLoader(config, memory_config)


# Context manager for memory-efficient processing
class MemoryEfficientContext:
    """Context manager for memory-efficient data processing"""

    def __init__(self, config: OAKConfig = None, memory_config: MemoryConfig = None):
        self.loader = create_memory_efficient_loader(config, memory_config)
        self.initial_memory = None

    def __enter__(self) -> MemoryEfficientLoader:
        self.initial_memory = self.loader.monitor.get_memory_info()
        self.loader.logger.info('Starting memory-efficient processing context')
        self.loader.monitor.log_memory_usage('context start')
        return self.loader

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get final memory usage
        final_memory = self.loader.monitor.get_memory_info()

        # Log memory usage change
        memory_change = (
            final_memory['process_memory_gb'] - self.initial_memory['process_memory_gb']
        )
        self.loader.logger.info(f'Memory usage change: {memory_change:+.2f} GB')

        # Get usage report
        report = self.loader.get_memory_usage_report()
        self.loader.logger.info(f'Memory usage report: {report}')

        # Cleanup
        self.loader.cleanup()
        self.loader.logger.info('Memory-efficient processing context ended')
