from typing import Dict, List, Optional, Union
import numpy as np

from bp_base.bp_engine import BeliefPropagation
from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent

class MinSumBP(BeliefPropagation):
    """
    Concrete implementation of BeliefPropagation using the Min-Sum algorithm.
    This algorithm minimizes the total cost, which is useful for DCOP problems.
    """
    
    def __init__(self, factor_graph: FactorGraph):
        """
        Initialize the Min-Sum Belief Propagation engine.
        
        Args:
            factor_graph: The factor graph to run belief propagation on.
        """
        super().__init__(factor_graph)
        # Optional damping factor (0.0-1.0) to help with convergence
        self.damping_factor = 0.5
        
    def get_beliefs(self) -> Dict[str, np.ndarray]:
        """
        Compute the beliefs (local marginals) for each variable node.
        In min-sum, this represents the cost (or energy) for each possible value.
        
        Returns:
            A dictionary mapping variable names to belief vectors.
        """
        beliefs = {}
        
        # For each variable node, compute its belief by summing incoming messages
        for node in self.graph.G.nodes():
            if isinstance(node, VariableAgent):
                # Initialize to zeros
                belief = np.zeros(node.domain)
                
                # Sum up messages from all connected factor nodes
                for message in node.mailbox:
                    belief += message.data
                
                # Store the belief
                beliefs[node.name] = belief
                
                # Also update the variable's internal belief
                node.final_belief = belief
                
        return beliefs
    
    def get_map_estimate(self) -> Dict[str, int]:
        """
        Return the MAP (Maximum A Posteriori) estimate for each variable.
        In min-sum, this means finding the value with the MINIMUM cost.
        
        Returns:
            A dictionary mapping variable names to their most likely values.
        """
        # First, compute the beliefs for all variables
        beliefs = self.get_beliefs()
        
        # For each variable, find the value that minimizes its cost
        map_estimate = {}
        for var_name, belief in beliefs.items():
            # In min-sum, we want the minimum value
            map_estimate[var_name] = int(np.argmin(belief))
            
        return map_estimate