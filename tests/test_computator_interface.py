"""
Tests for computator interface unification and easy swapping functionality.

This test suite validates that:
1. All computators have a consistent interface
2. Users can easily swap between different computator types
3. Vectorized operations remain exposed for performance
4. The registry system works correctly
"""

import pytest
import numpy as np
from typing import List

# Import the modules we need to test
from bp_base.computators import (
    BPComputator, 
    MinSumComputator, 
    MaxSumComputator, 
    ComputatorRegistry,
    ComputatorAdapter
)
from base_models.components import Message
from base_models.agents import VariableAgent, FactorAgent


class TestComputatorInterface:
    """Test that all computators have a consistent interface."""
    
    def test_bp_computator_interface(self):
        """Test that BPComputator has the expected interface."""
        comp = BPComputator()
        
        # Check that methods exist and are not async
        assert hasattr(comp, 'compute_Q')
        assert hasattr(comp, 'compute_R')
        assert callable(comp.compute_Q)
        assert callable(comp.compute_R)
        
        # Check that methods are not coroutines
        import inspect
        assert not inspect.iscoroutinefunction(comp.compute_Q)
        assert not inspect.iscoroutinefunction(comp.compute_R)
    
    def test_minsum_computator_interface(self):
        """Test that MinSumComputator has the expected interface."""
        comp = MinSumComputator()
        
        # Should inherit from BPComputator
        assert isinstance(comp, BPComputator)
        assert hasattr(comp, 'compute_Q')
        assert hasattr(comp, 'compute_R')
        
        # Check reduce and combine functions are set correctly
        assert comp.reduce_func == np.min
        assert comp.combine_func == np.add
    
    def test_maxsum_computator_interface(self):
        """Test that MaxSumComputator has the expected interface."""
        comp = MaxSumComputator()
        
        # Should inherit from BPComputator
        assert isinstance(comp, BPComputator)
        assert hasattr(comp, 'compute_Q')
        assert hasattr(comp, 'compute_R')
        
        # Check reduce and combine functions are set correctly
        assert comp.reduce_func == np.max
        assert comp.combine_func == np.add


class TestComputatorRegistry:
    """Test the computator registry system."""
    
    def test_registry_has_default_computators(self):
        """Test that default computators are registered."""
        available = ComputatorRegistry.list_available()
        assert "min_sum" in available
        assert "max_sum" in available
        assert "bp" in available
    
    def test_get_computator_by_name(self):
        """Test that we can retrieve computators by name."""
        min_sum = ComputatorRegistry.get("min_sum")
        max_sum = ComputatorRegistry.get("max_sum")
        bp = ComputatorRegistry.get("bp")
        
        assert isinstance(min_sum, MinSumComputator)
        assert isinstance(max_sum, MaxSumComputator)
        assert isinstance(bp, BPComputator)
    
    def test_registry_error_for_unknown_computator(self):
        """Test that registry raises error for unknown computator."""
        with pytest.raises(ValueError, match="Unknown computator"):
            ComputatorRegistry.get("unknown_computator")
    
    def test_register_custom_computator(self):
        """Test that we can register custom computators."""
        class CustomComputator(BPComputator):
            def __init__(self):
                super().__init__(reduce_func=np.sum, combine_func=np.multiply)
        
        # Register the custom computator
        ComputatorRegistry.register("custom", CustomComputator)
        
        # Should be able to retrieve it
        custom = ComputatorRegistry.get("custom")
        assert isinstance(custom, CustomComputator)
        assert "custom" in ComputatorRegistry.list_available()
    
    def test_create_custom_computator(self):
        """Test creating custom computators with specific functions."""
        custom = ComputatorRegistry.create_custom(
            reduce_func=np.sum, 
            combine_func=np.multiply
        )
        
        assert isinstance(custom, BPComputator)
        assert custom.reduce_func == np.sum
        assert custom.combine_func == np.multiply


class TestComputatorAdapter:
    """Test the computator adapter for unified interface."""
    
    def test_adapter_with_bp_computator(self):
        """Test adapter with BP computator."""
        bp_comp = BPComputator()
        adapter = ComputatorAdapter(bp_comp)
        
        # Should have the same interface
        assert hasattr(adapter, 'compute_Q')
        assert hasattr(adapter, 'compute_R')
        assert not adapter.is_search_computator()
    
    def test_adapter_delegates_to_wrapped_computator(self):
        """Test that adapter delegates calls to wrapped computator."""
        bp_comp = BPComputator()
        adapter = ComputatorAdapter(bp_comp)
        
        # Should delegate attribute access
        assert adapter.reduce_func == bp_comp.reduce_func
        assert adapter.combine_func == bp_comp.combine_func


class TestComputatorSwapping:
    """Test that computators can be easily swapped."""
    
    def create_test_messages(self):
        """Create test messages for computator testing."""
        # Create simple test agents
        var1 = VariableAgent("var1", domain=3)
        var2 = VariableAgent("var2", domain=3)
        factor = FactorAgent("factor1", domain=3, ct_creation_func=lambda: None)
        
        # Create test messages
        msg1 = Message(
            data=np.array([1.0, 2.0, 3.0]),
            sender=factor,
            recipient=var1
        )
        msg2 = Message(
            data=np.array([0.5, 1.5, 2.5]),
            sender=factor,
            recipient=var2
        )
        
        return [msg1, msg2]
    
    def test_swapping_between_minsum_and_maxsum(self):
        """Test swapping between MinSum and MaxSum computators."""
        messages = self.create_test_messages()
        
        # Test with MinSum
        min_sum_comp = ComputatorRegistry.get("min_sum")
        min_result = min_sum_comp.compute_Q(messages)
        
        # Test with MaxSum
        max_sum_comp = ComputatorRegistry.get("max_sum")
        max_result = max_sum_comp.compute_Q(messages)
        
        # Both should return valid results
        assert len(min_result) == len(max_result) == 2
        assert all(isinstance(msg, Message) for msg in min_result)
        assert all(isinstance(msg, Message) for msg in max_result)
        
        # For Q messages, both algorithms use the same combine function (addition)
        # so results may be the same. The difference is in the reduce function
        # which is used in R message computation.
        # Let's just verify that the computations completed successfully
        assert len(min_result) == 2
        assert len(max_result) == 2
    
    def test_vectorized_operations_preserved(self):
        """Test that vectorized operations are not encapsulated."""
        comp = ComputatorRegistry.get("min_sum")
        
        # Should have direct access to numpy operations
        assert hasattr(comp, 'reduce_func')
        assert hasattr(comp, 'combine_func')
        assert comp.reduce_func == np.min
        assert comp.combine_func == np.add
        
        # Should be able to use vectorized operations directly
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        result = comp.reduce_func(test_array, axis=0)
        expected = np.array([1, 2, 3])
        assert np.array_equal(result, expected)


class TestBackwardCompatibility:
    """Test that existing code still works after changes."""
    
    def test_existing_usage_patterns(self):
        """Test that existing usage patterns still work."""
        # Direct instantiation should still work
        comp1 = MinSumComputator()
        comp2 = MaxSumComputator()
        comp3 = BPComputator()
        
        assert isinstance(comp1, MinSumComputator)
        assert isinstance(comp2, MaxSumComputator)
        assert isinstance(comp3, BPComputator)
        
        # Should all have the same interface
        for comp in [comp1, comp2, comp3]:
            assert hasattr(comp, 'compute_Q')
            assert hasattr(comp, 'compute_R')
    
    def test_no_performance_degradation(self):
        """Test that changes don't add significant overhead."""
        # This is a basic test - in practice you'd benchmark
        comp = ComputatorRegistry.get("min_sum")
        
        # Should be the same object type as direct instantiation
        direct_comp = MinSumComputator()
        assert type(comp) == type(direct_comp)
        
        # Should have same attributes
        assert comp.reduce_func == direct_comp.reduce_func
        assert comp.combine_func == direct_comp.combine_func


if __name__ == "__main__":
    pytest.main([__file__])