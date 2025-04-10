from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from bp_base.agents import BPAgent, VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.factor_graph import FactorGraph
from bp_base.iteration import Iteration

class BeliefPropagation(ABC):
    """
    Abstract engine for belief propagation.
    """
    def __init__(self, factor_graph: FactorGraph):
        self.graph = factor_graph
        self.iterations = {}  # Dictionary mapping iteration number to Iteration object
        self.current_iteration = 0
        self.converged = False
        self.residual_threshold = 1e-4  # Default threshold for convergence
    
    def step(self):
        """Run a single step of the belief propagation algorithm."""
        # Compute messages to send and put them in the mailbox
        for agent in self.graph.G.nodes():
            agent.messages_to_send = agent.compute_messages(agent.mailbox)
            agent.mailbox = []  # Empty mailbox after computing messages
            
        # Send the messages to the right nodes
        for agent in self.graph.G.nodes():
            for message in agent.messages_to_send:
                message.recipient.receive_message(message)

    def cycle(self):
        """Run a complete cycle of message passing (diameter of the graph)."""
        # Create a new iteration object
        iteration = Iteration()
        iteration.number = self.current_iteration
        iteration.max_iterations = self.max_iterations if hasattr(self, 'max_iterations') else 1000
        
        # Run a message passing cycle
        for i in range(self.graph.diameter):
            self.step()
        
        # Collect messages for analysis
        q_messages = {}
        r_messages = {}
        
        # Collect variable-to-factor messages (Q)
        for variable in [n for n in self.graph.G.nodes() if isinstance(n, VariableAgent)]:
            for message in variable.messages_to_send:
                q_messages[(variable.name, message.recipient.name)] = message.data
        
        # Collect factor-to-variable messages (R)
        for factor in [n for n in self.graph.G.nodes() if isinstance(n, FactorAgent)]:
            for message in factor.messages_to_send:
                r_messages[(factor.name, message.recipient.name)] = message.data
        
        # Update the iteration object with the new messages
        iteration.update_messages(q_messages, r_messages)
        
        # Calculate residual if not the first iteration
        if self.current_iteration > 0:
            residual = iteration.calculate_residual()
            self.converged = residual < self.residual_threshold
        
        # Complete the iteration
        iteration.complete()
        
        # Store the iteration
        self.iterations[self.current_iteration] = iteration
        self.current_iteration += 1
        
        return iteration

    def run(self, max_iter: int = 1000) -> None:
        """
        Run the belief propagation algorithm for a maximum number of iterations.
        :param max_iter: Maximum number of iterations to run.
        """
        self.max_iterations = max_iter
        for i in range(max_iter):
            iteration = self.cycle()
            if self.converged:
                print(f"Converged after {i+1} iterations with residual {iteration.message_residual}")
                break
            
        if not self.converged:
            print(f"Did not converge after {max_iter} iterations.")
            
        return self.get_beliefs()

    @abstractmethod
    def get_beliefs(self) -> Dict[str, np.ndarray]:
        """
        Return the beliefs of all variable nodes in the factor graph.
        :return: A dictionary mapping variable names to belief vectors.
        """
        pass

    @abstractmethod
    def get_map_estimate(self) -> Dict[str, int]:
        """
        Return the MAP (Maximum A Posteriori) estimate for each variable.
        :return: A dictionary mapping variable names to their most likely values.
        """
        pass
        
    def is_converged(self) -> bool:
        """
        Check if the belief propagation algorithm has converged.
        :return: True if converged, False otherwise.
        """
        return self.converged
        
    def get_iterations_data(self) -> Dict[int, Iteration]:
        """
        Get the data for all iterations.
        :return: Dictionary mapping iteration numbers to Iteration objects.
        """
        return self.iterations
        
    def get_last_iteration(self) -> Optional[Iteration]:
        """
        Get the data for the last completed iteration.
        :return: The last Iteration object or None if no iterations have been run.
        """
        if self.current_iteration > 0:
            return self.iterations.get(self.current_iteration - 1)
        return None

# Concrete implementation classes can be defined in bp_variations