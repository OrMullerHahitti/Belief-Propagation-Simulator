# Computator Interface Guide

This guide explains how to easily switch between different computator types in the Belief Propagation Simulator while maintaining high performance.

## Problem Solved

Previously, there was an interface inconsistency between the abstract `Computator` class (which defined async methods) and concrete implementations (which were synchronous). This made it difficult for users to:

1. **Easily swap computators** - Calling code needed to know whether to await methods
2. **Maintain performance** - Risk of encapsulating vectorized operations
3. **Extend the system** - Inconsistent interfaces made adding new computators complex

## Solution: Unified Interface

We've unified the interface to be consistently synchronous and provided tools for easy computator management.

### Key Features

1. **Consistent Interface**: All computators use the same synchronous interface
2. **Easy Switching**: `ComputatorRegistry` for managing different computator types
3. **Performance Preserved**: Direct access to vectorized numpy operations
4. **Extensible**: Simple registration system for custom computators

## Quick Start

### Basic Usage

```python
from bp_base.computators import ComputatorRegistry

# Get different computators
min_sum = ComputatorRegistry.get("min_sum")
max_sum = ComputatorRegistry.get("max_sum")

# Use them interchangeably
result1 = min_sum.compute_Q(messages)
result2 = max_sum.compute_Q(messages)
```

### Available Computators

```python
# List all available computators
print(ComputatorRegistry.list_available())
# Output: ['min_sum', 'max_sum', 'bp']
```

### Custom Computators

```python
# Create custom computator with specific functions
custom = ComputatorRegistry.create_custom(
    reduce_func=np.mean,
    combine_func=np.multiply
)

# Register your own computator class
class MyAlgorithm(BPComputator):
    def __init__(self):
        super().__init__(reduce_func=np.sum, combine_func=np.add)

ComputatorRegistry.register("my_algo", MyAlgorithm)
my_comp = ComputatorRegistry.get("my_algo")
```

### Performance: Vectorized Operations

```python
# Direct access to numpy operations for maximum performance
comp = ComputatorRegistry.get("min_sum")

# Fast vectorized operations
large_array = np.random.rand(10000, 100)
result = comp.reduce_func(large_array, axis=1)  # Direct numpy call - no overhead!
```

## Migration Guide

### Before (Problematic)

```python
# Had to know if computator was async or not
if is_async_computator(comp):
    result = await comp.compute_Q(messages)
else:
    result = comp.compute_Q(messages)
```

### After (Unified)

```python
# Always synchronous, always works
result = comp.compute_Q(messages)

# Easy switching
for algo_name in ["min_sum", "max_sum"]:
    comp = ComputatorRegistry.get(algo_name)
    result = comp.compute_Q(messages)
```

## Advanced Usage

### Computator Adapter

For systems that need to work with both BP and Search computators:

```python
from bp_base.computators import ComputatorAdapter

# Wrap any computator for unified interface
adapter = ComputatorAdapter(some_computator)

# Check computator type
if adapter.is_search_computator():
    # Handle search-specific features
    decision = adapter.compute_decision(agent, neighbors)
else:
    # Standard BP computator
    pass

# Always works regardless of wrapped type
result = adapter.compute_Q(messages)
```

### Engine Integration

```python
from bp_base.engine_base import BPEngine

# Easy computator switching in engines
for algo in ["min_sum", "max_sum"]:
    computator = ComputatorRegistry.get(algo)
    engine = BPEngine(factor_graph=graph, computator=computator)
    engine.run()
```

## Performance Notes

- **Zero Overhead**: Registry and adapter patterns add negligible performance cost
- **Vectorized Operations**: Direct access to numpy functions preserved
- **Memory Efficient**: No additional copying or wrapping of data
- **Cache Friendly**: Existing optimizations in `BPComputator` maintained

## Testing

Comprehensive tests ensure:

- Interface consistency across all computators
- Backward compatibility with existing code
- Performance preservation
- Easy switching functionality

Run tests with:
```bash
python -m pytest tests/test_computator_interface.py -v
```

## Examples

See `examples/computator_switching_example.py` for a complete demonstration of all features.