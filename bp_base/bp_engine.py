
from abc import ABC, abstractmethod
from typing import Dict, List, Callable
import numpy as np

from bp_base.agents import BPAgent
from bp_base.components import Message
from bp_base.computators import MaxSumComputator
from bp_base.factor_graph import FactorGraph
from DCOP_base import Computator
class BeliefPropagation(ABC):
    """
    Abstract engine for belief propagation.
    """
    def __init__(self, factor_graph: FactorGraph,computator:Computator = MaxSumComputator(),policies:Dict[str, List[Callable]] = None):
        self.graph = factor_graph
        self.iterations = Dict[int, List[BPAgent]]
        self.graph.set_computator(computator)# Store history of beliefs
        self.policies = policies # Store policies - with all different kinds - message , cost table, stopping critiria, etc.

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


__doc__=""" in this module we will implement the belief propagation with various policies with factor graph configs
most of which are implemented in the factor graph module and will be max-sum with different policies and different structures
we will start with the usual 3-cycle and then move to more complex structures"""