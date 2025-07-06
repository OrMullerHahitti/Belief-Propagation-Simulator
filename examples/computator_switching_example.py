"""
Example: Easy Computator Switching

This example demonstrates how users can easily switch between different 
computator types while maintaining high performance through exposed 
vectorized operations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bp_base.computators import ComputatorRegistry, ComputatorAdapter
from base_models.components import Message
from base_models.agents import VariableAgent, FactorAgent


def main():
    print("=== Computator Interface Unification Example ===\n")
    
    # Show available computators
    print("Available computators:", ComputatorRegistry.list_available())
    
    # Create test data
    var1 = VariableAgent("var1", domain=3)
    factor1 = FactorAgent("factor1", domain=3, ct_creation_func=lambda: None)
    
    messages = [
        Message(data=np.array([1.0, 2.0, 3.0]), sender=factor1, recipient=var1),
        Message(data=np.array([0.5, 1.5, 2.5]), sender=factor1, recipient=var1)
    ]
    
    print(f"\nTest messages: {len(messages)} messages")
    print(f"Message 1 data: {messages[0].data}")
    print(f"Message 2 data: {messages[1].data}")
    
    # Easy switching between computators
    print("\n=== Easy Computator Switching ===")
    
    computator_types = ["min_sum", "max_sum", "bp"]
    
    for comp_type in computator_types:
        print(f"\n--- Using {comp_type.upper()} computator ---")
        
        # Get computator from registry
        computator = ComputatorRegistry.get(comp_type)
        
        # Compute Q messages
        result = computator.compute_Q(messages)
        
        print(f"Computator type: {type(computator).__name__}")
        print(f"Reduce function: {computator.reduce_func.__name__}")
        print(f"Combine function: {computator.combine_func.__name__}")
        print(f"Result: {len(result)} messages")
        print(f"First result data: {result[0].data}")
    
    # Custom computator creation
    print("\n=== Custom Computator Creation ===")
    
    # Create a custom computator with different functions
    custom_comp = ComputatorRegistry.create_custom(
        reduce_func=np.mean,
        combine_func=np.multiply
    )
    
    print(f"Custom computator reduce function: {custom_comp.reduce_func.__name__}")
    print(f"Custom computator combine function: {custom_comp.combine_func.__name__}")
    
    # Register a custom class
    class CustomAlgorithm(ComputatorRegistry._computators["bp"]):
        def __init__(self):
            super().__init__(reduce_func=np.sum, combine_func=np.add)
    
    ComputatorRegistry.register("custom_algo", CustomAlgorithm)
    custom_algo = ComputatorRegistry.get("custom_algo")
    
    print(f"Registered custom algorithm: {type(custom_algo).__name__}")
    
    # Using the adapter for unified interface
    print("\n=== Computator Adapter Usage ===")
    
    min_sum_comp = ComputatorRegistry.get("min_sum")
    adapter = ComputatorAdapter(min_sum_comp)
    
    print(f"Adapter wraps: {type(adapter.computator).__name__}")
    print(f"Is search computator: {adapter.is_search_computator()}")
    
    # Adapter provides the same interface
    adapter_result = adapter.compute_Q(messages)
    print(f"Adapter result: {len(adapter_result)} messages")
    
    # Performance demonstration - vectorized operations are preserved
    print("\n=== Vectorized Operations Preserved ===")
    
    comp = ComputatorRegistry.get("min_sum")
    
    # Direct access to numpy operations
    test_array = np.random.rand(1000, 100)
    
    # Measure time for vectorized operation
    import time
    start_time = time.time()
    result = comp.reduce_func(test_array, axis=1)
    end_time = time.time()
    
    print(f"Vectorized operation on {test_array.shape} array")
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Result shape: {result.shape}")
    print("✓ Vectorized operations are fast and accessible!")
    
    print("\n=== Summary ===")
    print("✓ Easy switching between computator types")
    print("✓ Consistent interface across all computators")
    print("✓ Vectorized operations remain exposed for performance")
    print("✓ Registry system for managing computators")
    print("✓ Adapter pattern for unified interface")
    print("✓ Custom computator support")


if __name__ == "__main__":
    main()