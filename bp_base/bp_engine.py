from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from bp_base.agents import BPAgent
from bp_base.components import Message
from bp_base.factor_graph import FactorGraph

class BeliefPropagation(ABC):
    """
    Abstract engine for belief propagation.
    """
    def __init__(self, factor_graph: FactorGraph):
        self.graph = factor_graph
        self.iterations = Dict[int, List[BPAgent]]  # Store history of beliefs

    def step(self):
        """Run the factor graph algorithm."""

        # compute messages to send and put them in the mailbox
        for agent in self.graph.G.nodes():
            agent.messages_to_send = agent.compute_messages(agent.mailbox)
            agent.empty_mailbox()
        # send the messages to the right nodes
        for agent in self.graph.G.nodes():
            for message in agent.messages_to_send:
                message.sender.send_message(message.recipient, message)

    def cycle(self):
        for i in range(self.graph.G.diameter):
            self.step()

    def run(self, max_iter: int = 1000) -> None:
        """
        Run the factor graph algorithm for a maximum number of iterations.
        :param max_iter: Maximum number of iterations to run.
        """
        for i in range(max_iter):
            self.cycle()
            if self.is_converged():
                break

    def get_beliefs(self) -> Dict[str, np.ndarray]:
        ''' Return the beliefs of the factor graph.
        :return:
        :param: A dictionary mapping variable names to belief vectors.'''
        pass


    def get_map_estimate(self) -> Dict[str, int]:
        pass
    def is_converged(self) -> bool:
        if self.cycle[-1] == self.cycle[-2]:
            return True
        else:
            return False