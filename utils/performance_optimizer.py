"""
Performance optimization utilities for the Belief Propagation Simulator.

This module provides helper functions and classes for monitoring and optimizing
performance in the belief propagation simulator.
"""

import time
import numpy as np
from functools import wraps
from typing import Dict, Any, Optional, List
import logging
import os

# Try to import psutil, fall back to basic functionality if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Lightweight performance profiler for BP operations."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
        
    def profile_method(self, method_name: str):
        """Decorator to profile method execution time and memory usage."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Record initial memory (if psutil available)
                initial_memory = 0
                if HAS_PSUTIL:
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time the execution
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                # Record final memory (if psutil available)
                final_memory = 0
                memory_delta = 0
                if HAS_PSUTIL:
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = final_memory - initial_memory
                
                # Store results
                execution_time = end_time - start_time
                
                if method_name not in self.timings:
                    self.timings[method_name] = []
                    self.memory_usage[method_name] = []
                
                self.timings[method_name].append(execution_time)
                self.memory_usage[method_name].append(memory_delta)
                
                if execution_time > 0.1:  # Log slow operations
                    logger.debug(f"{method_name} took {execution_time:.3f}s, memory delta: {memory_delta:.1f}MB")
                
                return result
            return wrapper
        return decorator
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        for method_name in self.timings:
            times = self.timings[method_name]
            memory_deltas = self.memory_usage[method_name]
            
            summary[method_name] = {
                'call_count': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'max_time': max(times) if times else 0,
                'avg_memory_delta': np.mean(memory_deltas),
                'max_memory_delta': max(memory_deltas) if memory_deltas else 0
            }
        
        return summary
    
    def print_summary(self):
        """Print a formatted performance summary."""
        summary = self.get_summary()
        print("\n=== Performance Summary ===")
        for method_name, stats in summary.items():
            print(f"\n{method_name}:")
            print(f"  Calls: {stats['call_count']}")
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"  Avg time: {stats['avg_time']:.3f}s")
            print(f"  Max time: {stats['max_time']:.3f}s")
            print(f"  Avg memory delta: {stats['avg_memory_delta']:.1f}MB")
            print(f"  Max memory delta: {stats['max_memory_delta']:.1f}MB")


def optimize_numpy_settings():
    """Configure NumPy for optimal performance."""
    # Set optimal thread counts for NumPy operations
    import os
    
    # Limit threading to prevent oversubscription in multiprocessing
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Enable faster math operations (if available)
    try:
        np.seterr(all='ignore')  # Ignore overflow warnings for speed
    except:
        pass


def estimate_memory_usage(factor_graph, engine_configs: Dict) -> float:
    """Estimate memory usage for a simulation."""
    # Rough estimation based on graph size and configuration
    num_vars = len(factor_graph.variables) if hasattr(factor_graph, 'variables') else 50
    num_factors = len(factor_graph.factors) if hasattr(factor_graph, 'factors') else 50
    
    # Estimate based on typical message sizes and history
    estimated_mb = (num_vars * num_factors * 4 * 4) / (1024 * 1024)  # 4 bytes per float, domain size ~4
    estimated_mb *= len(engine_configs)  # Multiple engines
    estimated_mb *= 2  # Safety margin
    
    return estimated_mb


def get_optimal_batch_size(total_simulations: int, available_memory_gb: float) -> int:
    """Calculate optimal batch size based on available memory."""
    # Conservative estimation: each simulation uses ~10MB
    memory_per_sim_mb = 10
    available_memory_mb = available_memory_gb * 1024 * 0.8  # Use 80% of available memory
    
    max_batch_from_memory = int(available_memory_mb / memory_per_sim_mb)
    
    # Get CPU cores (fallback to reasonable default if psutil not available)
    if HAS_PSUTIL:
        cpu_cores = psutil.cpu_count()
    else:
        cpu_cores = os.cpu_count() or 4  # fallback to 4 cores
    
    # Balance between memory constraints and CPU utilization
    optimal_batch = min(
        total_simulations,
        max_batch_from_memory,
        cpu_cores * 8  # 8 simulations per core
    )
    
    return max(1, optimal_batch)


def cleanup_memory():
    """Force garbage collection and memory cleanup."""
    import gc
    gc.collect()
    
    # Try to release unused memory back to OS (Linux/Mac)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass  # Not available on all systems


class MemoryMonitor:
    """Monitor memory usage during execution."""
    
    def __init__(self, threshold_mb: float = 1000):
        self.threshold_mb = threshold_mb
        self.peak_memory = 0
        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None
        
    def check_memory(self, context: str = ""):
        """Check current memory usage and warn if threshold exceeded."""
        if not HAS_PSUTIL:
            return False  # Can't check memory without psutil
            
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if current_memory > self.threshold_mb:
            logger.warning(
                f"High memory usage detected{' in ' + context if context else ''}: "
                f"{current_memory:.1f}MB (threshold: {self.threshold_mb}MB)"
            )
            return True
        return False
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        if not HAS_PSUTIL:
            return {
                'current_mb': 0.0,
                'peak_mb': self.peak_memory,
                'virtual_mb': 0.0
            }
            
        memory_info = self.process.memory_info()
        return {
            'current_mb': memory_info.rss / 1024 / 1024,
            'peak_mb': self.peak_memory,
            'virtual_mb': memory_info.vms / 1024 / 1024
        }


# Global performance profiler instance
global_profiler = PerformanceProfiler(enabled=False)  # Disabled by default

def enable_profiling():
    """Enable global performance profiling."""
    global_profiler.enabled = True

def disable_profiling():
    """Disable global performance profiling."""
    global_profiler.enabled = False

def get_profiling_summary():
    """Get global profiling summary."""
    return global_profiler.get_summary()

def print_profiling_summary():
    """Print global profiling summary."""
    global_profiler.print_summary()