# Code Refactoring: Bottom-Up Programming Implementation

This document describes the comprehensive refactoring undertaken to implement bottom-up programming principles by encapsulating complex control structures (if/for loops) into focused, reusable functions.

## Overview

The refactoring follows the **Single Responsibility Principle** and **Extract Method** pattern to transform complex, nested control structures into clean, modular functions. This approach improves:

- **Readability**: Each function has a clear, focused purpose
- **Maintainability**: Smaller functions are easier to modify and debug
- **Testability**: Individual components can be tested in isolation
- **Reusability**: Common patterns are extracted into utility functions

## Refactoring Strategy

### 1. Identify Complex Control Structures
We identified areas with:
- Nested for loops with complex logic
- Long if-else chains
- Multiple responsibilities in single methods
- Repetitive patterns across modules

### 2. Extract Methods Pattern
For each complex method, we:
1. **Identified logical units** within the method
2. **Extracted each unit** into a focused helper method
3. **Named methods descriptively** to convey intent
4. **Maintained original public interface** for backward compatibility

### 3. Create Utility Functions
Common patterns were extracted into reusable utilities in `base_models/utils.py`.

## Changes by Module

### simulator.py

#### Original Issues:
- `plot_results()`: 45-line method with nested loops and complex conditionals
- `run_simulations()`: Complex simulation management with mixed concerns
- `_run_in_batches()`: Batch processing with error handling mixed in

#### Refactored Solution:

**Plotting Logic (6 methods)**
```python
def plot_results(self, max_iter=5000, verbose=False):
    # Orchestration method - clear high-level flow
    for idx, (engine_name, costs_list) in enumerate(self.results.items()):
        processed_costs = self._validate_and_process_costs(engine_name, costs_list, max_iter)
        if processed_costs is None:
            continue
        color = colors[idx % len(colors)]
        self._plot_engine_results(engine_name, processed_costs, color, verbose)
    self._finalize_plot(verbose)

# Helper methods:
# - _validate_and_process_costs() - Data validation and preprocessing  
# - _plot_engine_results() - Main plotting logic per engine
# - _plot_individual_runs() - Individual run visualization
# - _plot_uncertainty_bands() - Standard deviation bands
# - _finalize_plot() - Plot styling and display
```

**Simulation Management (3 methods)**
```python
def run_simulations(self, graphs, max_iter=5000):
    # Clear separation of concerns
    simulation_args = self._prepare_simulation_arguments(graphs, max_iter)
    all_results = self._execute_simulations(simulation_args)
    self._process_simulation_results(all_results, simulation_args)
    return self.results
```

**Batch Processing (5 methods)**
```python
def _run_in_batches(self, simulation_args, batch_size=None, max_workers=None):
    # Configuration and orchestration separated
    batch_config = self._configure_batch_processing(batch_size, max_workers)
    batches = self._create_batches(simulation_args, batch_config['batch_size'])
    
    for batch_num, batch in enumerate(batches, 1):
        batch_results = self._process_single_batch(batch, batch_num, len(batches), batch_config['max_workers'])
        all_results.extend(batch_results)
```

### bp_base/computators.py

#### Original Issues:
- `compute_R()`: Dense 25-line method with vectorized operations and complex loops

#### Refactored Solution:

**Message Computation (4 methods)**
```python
def compute_R(self, cost_table: np.ndarray, incoming_messages: List[Message]):
    # Clear pipeline of operations
    computation_data = self._prepare_r_computation(cost_table, incoming_messages)
    aggregated_costs = self._aggregate_costs_and_messages(cost_table, computation_data['b_msgs'])
    return self._build_r_messages(aggregated_costs, computation_data, incoming_messages)

# Helper methods:
# - _prepare_r_computation() - Data structure preparation
# - _aggregate_costs_and_messages() - Cost aggregation
# - _build_r_messages() - Message construction
```

### bp_base/factor_graph.py

#### Original Issues:
- `global_cost`: Complex nested loops with validation mixed in
- `__setstate__()`: Graph reconstruction with nested structure building

#### Refactored Solution:

**Global Cost Calculation (5 methods)**
```python
@property
def global_cost(self) -> int | float:
    # High-level flow
    var_name_assignments = self._get_variable_assignments()
    return self._compute_total_factor_cost(var_name_assignments)

# Helper methods handle specific responsibilities:
# - _get_variable_assignments() - Create assignment mapping
# - _compute_total_factor_cost() - Iterate and sum factor costs
# - _compute_single_factor_cost() - Handle individual factor logic
# - _build_factor_indices() - Complex index construction logic
```

**Graph Reconstruction (4 methods)**
```python
def __setstate__(self, state):
    self.__dict__.update(state)
    if not hasattr(self, "G") or self.G is None:
        self._reconstruct_graph()  # Clear delegation

# Helper methods:
# - _reconstruct_graph() - Orchestration
# - _add_graph_nodes() - Node addition logic
# - _rebuild_graph_edges() - Edge reconstruction logic  
# - _add_factor_edges() - Factor-specific edge logic
```

### utils/examples.py

#### Original Issues:
- `create_factor_graph()`: Repetitive if-else logic for graph types

#### Refactored Solution:

**Strategy Pattern Implementation**
```python
def create_factor_graph(graph_type="cycle", ...):
    ct_params = _get_default_ct_params(ct_params)
    ct_factory_fn = CT_FACTORIES[ct_factory]
    variables, factors, edges = _build_graph_by_type(
        graph_type, num_vars, domain_size, ct_factory_fn, ct_params, density
    )
    return FactorGraph(variable_li=variables, factor_li=factors, edges=edges)

def _build_graph_by_type(graph_type, ...):
    # Dictionary-based dispatch instead of if-else chain
    graph_builders = {
        "cycle": lambda: build_cycle_graph(...),
        "random": lambda: build_random_graph(...)
    }
    if graph_type not in graph_builders:
        raise ValueError(f"Unknown graph type: {graph_type}")
    return graph_builders[graph_type]()
```

### base_models/components.py

#### Original Issues:
- `receive_messages()`: Mixed single/multiple message handling
- Complex inline logic for pruning and validation

#### Refactored Solution:

**Message Handling (4 methods)**
```python
def receive_messages(self, messages: Message | list[Message]):
    # Clear dispatch based on type
    if isinstance(messages, list):
        self._receive_multiple_messages(messages)
        return
    self._receive_single_message(messages)

# Helper methods:
# - _receive_multiple_messages() - Handle list iteration
# - _receive_single_message() - Handle individual message
# - _should_prune_message() - Pruning policy logic
# - _populate_inbox_from_messages() - Inbox population logic
```

### base_models/utils.py (New Module)

Created comprehensive utility functions for common patterns:

**Data Processing Utilities**
- `process_items_with_validation()` - Validation + processing pipeline
- `filter_and_transform()` - Functional filter + transform
- `batch_process()` - Generic batch processing

**Data Structure Utilities**  
- `safe_dict_lookup()` - Safe dictionary operations
- `validate_and_extract()` - Data validation patterns
- `create_index_mapping()` - Index mapping creation
- `build_nested_structure()` - Structure building from flat data

**Conditional Processing**
- `conditional_aggregate()` - Conditional numpy operations

## Benefits Achieved

### Readability Improvements
- **Before**: 45-line `plot_results()` method with mixed concerns
- **After**: 6-line orchestration method + 5 focused helpers

### Maintainability Improvements  
- **Separation of Concerns**: Each method has one responsibility
- **Focused Testing**: Individual components can be unit tested
- **Easier Debugging**: Smaller methods are easier to trace

### Reusability Improvements
- **Common Patterns**: Extracted into `base_models/utils.py`
- **Strategy Pattern**: Graph creation uses dispatch table
- **Modular Components**: Methods can be reused across classes

### Performance Maintained
- **Same Algorithms**: Core computational logic unchanged
- **Same Interfaces**: Public APIs preserved for compatibility  
- **Optimizations Preserved**: Vectorized operations maintained

## Testing Strategy

The refactoring maintains all existing public interfaces, so existing tests should continue to pass without modification. The extracted methods can also be tested individually for better coverage.

## Future Improvements

1. **Extract More Utilities**: Additional common patterns can be moved to utils
2. **Add Type Hints**: Improve type safety for extracted methods
3. **Performance Profiling**: Ensure no performance regression from method calls
4. **Documentation**: Add docstrings to all extracted methods

## Conclusion

This refactoring successfully implements bottom-up programming by:
- **Encapsulating complex control structures** into focused functions
- **Eliminating nested loops** from public methods
- **Creating reusable utility functions** for common patterns
- **Improving code organization** through clear separation of concerns

The result is a more maintainable, testable, and readable codebase that follows modern software engineering best practices.