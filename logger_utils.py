#!/usr/bin/env python3
"""
Logger Utilities for MVPA Pipeline
=================================

Provides consistent logging across the pipeline with memory usage tracking
and performance monitoring capabilities.

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)


class PipelineLogger:
    """Enhanced logger with memory and performance tracking"""
    
    def __init__(self, name: str, level: int = logging.INFO, 
                 log_file: Optional[str] = None,
                 include_memory: bool = True):
        """
        Initialize pipeline logger
        
        Parameters:
        -----------
        name : str
            Logger name
        level : int
            Logging level
        log_file : str, optional
            Log file path (if None, uses name-based filename)
        include_memory : bool
            Whether to include memory usage in logs
        """
        self.name = name
        self.include_memory = include_memory
        self.start_time = time.time()
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = LOGS_DIR / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Log initialization
        self.logger.info(f"Logger initialized: {name}")
        if self.include_memory:
            self._log_system_info()
    
    def _log_system_info(self):
        """Log system information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            self.logger.info(f"System info - Memory: {memory.total / (1024**3):.1f} GB, "
                           f"CPUs: {cpu_count}, Available: {memory.available / (1024**3):.1f} GB")
        except ImportError:
            self.logger.warning("psutil not available - memory tracking disabled")
    
    def log_memory_usage(self, operation: str = "", include_system: bool = False):
        """Log current memory usage"""
        if not self.include_memory:
            return
            
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            message = f"Memory usage"
            if operation:
                message += f" {operation}"
            
            message += f": {memory_info.rss / (1024**3):.2f} GB"
            
            if include_system:
                system_memory = psutil.virtual_memory()
                message += f" (System: {system_memory.used / (1024**3):.1f}/"
                message += f"{system_memory.total / (1024**3):.1f} GB, "
                message += f"{system_memory.percent:.1f}%)"
            
            self.logger.info(message)
            
        except ImportError:
            pass  # Skip if psutil not available
    
    def log_performance(self, operation: str, start_time: float, 
                       items_processed: int = None):
        """Log performance metrics"""
        duration = time.time() - start_time
        
        message = f"Performance - {operation}: {duration:.2f}s"
        
        if items_processed is not None:
            rate = items_processed / duration if duration > 0 else 0
            message += f" ({items_processed} items, {rate:.1f} items/s)"
        
        self.logger.info(message)
    
    def log_step(self, step_name: str, step_number: int = None, 
                total_steps: int = None):
        """Log processing step"""
        message = f"Step"
        if step_number is not None:
            message += f" {step_number}"
            if total_steps is not None:
                message += f"/{total_steps}"
        
        message += f": {step_name}"
        
        self.logger.info(message)
        
        if self.include_memory:
            self.log_memory_usage(f"after {step_name}")
    
    def log_results(self, results: Dict[str, Any], prefix: str = "Results"):
        """Log results dictionary"""
        self.logger.info(f"{prefix}:")
        for key, value in results.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        message = f"Error"
        if context:
            message += f" in {context}"
        message += f": {type(error).__name__}: {str(error)}"
        
        self.logger.error(message)
    
    def get_runtime(self) -> float:
        """Get total runtime since logger creation"""
        return time.time() - self.start_time


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: PipelineLogger, operation: str, 
                 log_start: bool = True, log_end: bool = True):
        self.logger = logger
        self.operation = operation
        self.log_start = log_start
        self.log_end = log_end
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.log_start:
            self.logger.logger.info(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_end:
            duration = time.time() - self.start_time
            self.logger.logger.info(f"Completed {self.operation} in {duration:.2f}s")
    
    def get_duration(self) -> float:
        """Get current duration"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


def setup_pipeline_logging(name: str = "mvpa_pipeline", 
                         level: int = logging.INFO,
                         log_dir: Optional[str] = None) -> PipelineLogger:
    """Setup pipeline logging with standard configuration"""
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    else:
        log_file = None
    
    return PipelineLogger(name, level, log_file)


def get_logger(name: str) -> logging.Logger:
    """Get standard logger"""
    return logging.getLogger(name) 