"""A collection of general-purpose utility functions and decorators.

This module contains miscellaneous helper functions that are used across
various parts of the PropFlow library, such as a dummy function for placeholders
and a profiling decorator for performance analysis.
"""
from functools import lru_cache, wraps
from typing import Callable, Any


def dummy_func(*args: Any, **kwargs: Any) -> None:
    """A dummy function that does nothing and returns nothing.

    This function serves as a placeholder for callbacks or functions that are
    optional or not yet implemented, allowing the main logic to proceed
    without raising errors.

    Args:
        *args: Accepts any positional arguments.
        **kwargs: Accepts any keyword arguments.
    """
    pass


def profiling(func: Callable) -> Callable:
    """A decorator to profile a function's execution time using `cProfile`.

    When a function decorated with `@profiling` is called, this decorator will
    wrap the call with `cProfile`, execute the function, and then print a
    performance report to the console. The report lists the top 10 functions
    sorted by cumulative time spent.

    Example:
        @profiling
        def my_expensive_function():
            # ... function logic ...

    Args:
        func: The function to be profiled.

    Returns:
        The wrapped function with profiling enabled.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative").print_stats(10)
        return result

    return wrapper
