import typing
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Tuple, Any
import numpy as np
import networkx as nx
from bp_base.agents import BPAgent, VariableAgent
from bp_base.components import Message
from bp_base.computators import MaxSumComputator
from bp_base.factor_graph import FactorGraph
from DCOP_base import Computator, Agent
from functools import reduce
from bp_base.typing import Policy, PolicyType
from dataclasses import dataclass, field
T = typing.TypeVar('T')
""" in this module we will implement the belief propagation with various policies with factor graph configs
most of which are implemented in the factor graph module and will be max-sum with different policies and different structures
we will start with the usual 3-cycle and then move to more complex structures"""

@dataclass
class Step:
    """
    A class to represent a step in the factor graph.
    """
    num: int = 0
    messages: Dict[str, List[Message]] = field(default_factory=dict)


    def add(self, agent:Agent, messages: List[Message]):
        """
        Add a List of messages for each agent per step.
        :param agent: Agent who will send the messages next step
        :param messages: the messages to be sent
        :return:
        """
        if agent.name not in self.messages:
            self.messages[agent.name] = []
        self.messages[agent.name].extend(messages)


@dataclass
class Cycle:
    """
    A class to represent a cycle in the factor graph.
    """

    number: int
    steps : List[Step] = field(default_factory=list)

    def add(self, step: Step):
        """
        Add a step to the cycle.
        """
        self.steps.append(step)
    def __eq__(self, other: 'Cycle'):
        """
        Check if two cycles are equal.
        """
        if len(self.steps) != len(other.steps):
            return False
        for step1, step2 in zip(self.steps, other.steps):
            if step1.messages != step2.messages:
                return False
        return True

class History:
    def __init__(self,**kwargs):
        self.config = dict(kwargs)
        self.cycles : Dict[int,Cycle] = {}
    def __setitem__(self, key:int, value:Cycle):
        self.cycles[key] = value
    def __getitem__(self, key:int):
        return self.cycles[key]
    def compare_last_two_cycles(self):
        if len(self.cycles) < 2:
            return False
        last_cycle = list(self.cycles.values())[-1]
        second_last_cycle = list(self.cycles.values())[-2]
        return last_cycle == second_last_cycle

### TODO: create a wrapper to config everything beforehand
### TODO: add a class to handle the policies and the history of the beliefs
### begining running the algorithm
class BPEngine:
    """
    Abstract engine for belief propagation.
    """
    def __init__(self, factor_graph:FactorGraph, computator:Computator = MaxSumComputator(), policies:Dict[PolicyType, List[Policy]]|None=None):
        """
        Initialize the belief propagation engine.
        :param factor_graph:
        :param computator:
        :param policies:
        """
        self.graph = factor_graph
        self.graph.set_computator(computator)# Store history of beliefs
        self.policies = policies # Store policies - with all different kinds - message , cost table, stopping critiria, etc.
        self.history = History(computator = computator,policies = policies,factor_graph = factor_graph) # Store history of beliefs

    # TODO:maybe apply here cost table policies too? or after a cycle? should think this over will open issue
    def step(self,i:int = 0) -> Step:
        """Run the factor graph algorithm."""
        step = Step(i)
        # compute messages to send and put them in the mailbox
        for agent in self.graph.G.nodes():
            agent.messages_to_send = agent.compute_messages(agent.mailbox)
            step.add(agent, agent.messages_to_send)

            #apply message policies
            for message in agent.messages_to_send:
                    # apply policies to the message
                if isinstance(message, Message) and self.policies["message"]:
                    message.data = self._apply_policies(self.policies["message"], message)
                # add the message to the mailbox

            agent.empty_mailbox()

        # send the messages to the right nodes
        for agent in self.graph.G.nodes():
            for message in agent.messages_to_send:
                message.sender.send_message(message.recipient, message)

        return step

    def cycle(self,j) -> Cycle:
        cy=Cycle(j)
        #TODO add diameter in the factorgraph class
        for i in range(nx.diameter(self.graph.G)):
            cy.add(self.step(i))
        return cy

    def run(self, max_iter: int = 1000) -> None:
        """
        Run the factor graph algorithm for a maximum number of iterations.
        :param max_iter: Maximum number of iterations to run.
        """
        for i in range(max_iter):
            self.history[i]=self.cycle(i)
            if self._is_converged():
                break
### -------------------------------------------------------------------####
    #from here we will implement the getters/properties for the beliefs and the assignments
    def get_beliefs(self) -> Dict[str, np.ndarray]:
        ''' Return the beliefs of the factor graph.
        :return:
        :param: A dictionary mapping variable names to belief vectors.'''
        pass
    #TODO : make it modular with the policies (i.e. policy stopping criteria)
    def _is_converged(self) -> bool:
        return self.history.compare_last_two_cycles()

    @property
    def assignments(self) -> Dict[str, int|float]:
        """
        Get the assignments of the factor graph.
        Get the assignments of the factor graph.
        :return: A dictionary mapping variable names to their assignments.
        """
        return {node.name: node.curr_assignment for node in self.graph.G.nodes() if isinstance(node, VariableAgent)}

    @staticmethod
    def _apply_policies(policies:List[Policy],data:T)->T:
        """
        Apply the policies to the factor graph.
        """

        return reduce(lambda acc, policy: policy(acc), policies, data)


