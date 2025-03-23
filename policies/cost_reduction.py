#cost reduction policy for the factor node
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, Any, List, Union, Optional, Tuple, cast

import numpy as np

from DCOP_base import Agent
from policies.abstract import FactorPolicy
from utils.kappa_calculations import Envelope, Line

Iteration: TypeAlias = Any #for now only a place holder
'''
this module is mostly for config!!
'''
class CostReductionPolicy(FactorPolicy, ABC):
    def __init__(self, stopping_critiria: Optional[Callable[[Iteration], bool]] = None,
                 applying_critiria: Optional[Callable[[Iteration], bool]] = None):
        self.calc_k = stopping_critiria
        self.applying_critiria = applying_critiria


    def should_apply(self, iteration: Iteration) -> bool:
        if not self.applying_critiria:
            return False
        return self.applying_critiria(iteration)


    @abstractmethod
    def get_K(self, agent: Optional[Agent] = None) -> float:
        """
        Return the cost-reduction multiplier (K) for the given iteration.
        """
        pass

class EveryTimeCostReduction(CostReductionPolicy):

    def should_apply(self, iteration: int) -> bool:
        return True
    
    def get_K(self) -> float:
        return 0.5

class MinMaxEnvelopeCostReduction(CostReductionPolicy):

    def should_apply(self, iteration: int) -> bool:
        return iteration % 2 == 0

    def get_K(self) -> float:
        return 0.5

class EnvelopeBasedCostReduction(CostReductionPolicy):
    """
    A cost reduction policy that uses envelope calculations to determine
    the optimal K value dynamically.
    """
    def __init__(self, 
                 applying_critiria: Optional[Callable[[int], bool]] = None,
                 min_k: float = 0.0, 
                 max_k: float = 1.0,
                 num_samples: int = 50):
        """
        Initialize the envelope-based cost reduction policy.
        
        Args:
            applying_critiria: Function that determines when to apply the cost reduction
            min_k: Minimum allowed value for K (default: 0.0)
            max_k: Maximum allowed value for K (default: 1.0)
            num_samples: Number of samples to use when calculating the envelope (default: 50)
        """
        super().__init__(None, applying_critiria)
        self.min_k = min_k
        self.max_k = max_k
        self.num_samples = num_samples
        self.last_k = max_k  # Default to max_k initially

    def should_apply(self, iteration: int) -> bool:
        """
        Determine if cost reduction should be applied for this iteration.
        If no criteria is specified, applies at every iteration.
        """
        if not self.applying_critiria:
            return True
        return self.applying_critiria(iteration)

    def get_K(self, cost_table: Optional[np.ndarray] = None, 
              message_data: Optional[np.ndarray] = None,
              agent: Optional[Agent] = None) -> float:
        """
        Calculate the optimal K value based on envelope calculations.
        
        Args:
            cost_table: The cost table for which to calculate the envelope
            message_data: The message data (Q) to use in the envelope calculation
            agent: Optional agent reference (not used in this implementation)
            
        Returns:
            The calculated K value
        """
        if cost_table is None or message_data is None:
            return self.last_k  # Return the last calculated K if data is missing
        
        # Create envelope instance
        envelope = Envelope(cost_table, message_data, k=self.max_k)
        
        # Calculate envelope and find the k value that minimizes the envelope
        k_values, envelope_values = envelope.calculate_envelope(num_points=self.num_samples)
        
        # Find the k value that gives the minimum envelope value
        min_idx = np.argmin(envelope_values)
        optimal_k = k_values[min_idx]
        
        # Ensure K is within bounds
        optimal_k = max(self.min_k, min(self.max_k, optimal_k))
        
        # Store the calculated K for future reference
        self.last_k = optimal_k
        
        return optimal_k

    def get_optimal_k_segments(self, cost_table: np.ndarray, 
                              message_data: np.ndarray) -> List[Tuple[float, float, int, int]]:
        """
        Get the segments of the envelope where different entries in the cost table are minimal.
        Useful for analyzing which transitions occur at different K values.
        
        Returns:
            List of (k_start, k_end, row_index, col_index) tuples
        """
        envelope = Envelope(cost_table, message_data, k=self.max_k)
        return envelope.calculate_minimum_envelope_segments(num_points=self.num_samples)
