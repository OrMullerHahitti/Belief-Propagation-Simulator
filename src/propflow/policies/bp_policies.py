"""Abstract Base Classes for Belief Propagation Policies.

This module defines the foundational abstract classes for various policies that
can be applied to modify the behavior of belief propagation algorithms. These
policies can operate on different components of the factor graph, such as
variables, factors, messages, or the entire graph structure.
"""
from abc import abstractmethod
from enum import Enum
from typing import List, Dict

from ..core.agents import VariableAgent, FactorAgent
from ..core.components import Message
from ..bp.factor_graph import FactorGraph


class PolicyType(int, Enum):
    """An enumeration for the different types of policies.

    This helps categorize policies based on the component they operate on.
    """

    def __new__(cls, value: int, p_type: str):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.type = p_type
        return obj

    FACTOR = (10, "factor")
    VARIABLE = (20, "variable")
    MESSAGE = (30, "message")
    GRAPH = (40, "graph")

    def __str__(self) -> str:
        return self.type


class Policy:
    """An abstract base class for all policies.

    Attributes:
        policy_type (PolicyType): The type of the policy, indicating which
            part of the graph it applies to.
    """

    def __init__(self, policy_type: PolicyType):
        """Initializes the Policy.

        Args:
            policy_type: The type of the policy.
        """
        self.policy_type = policy_type

    def __call__(self, *args, **kwargs):
        """Executes the policy. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")


class DampingPolicy(Policy):
    """An abstract base class for message damping policies.

    Damping policies are applied to variable nodes to stabilize message updates
    by blending new messages with old ones.
    """

    def __init__(self):
        """Initializes the damping policy, setting its type to VARIABLE."""
        super().__init__(PolicyType.VARIABLE)

    def __call__(self, var: VariableAgent) -> List[Message]:
        """Applies damping to the incoming messages for a variable.

        Args:
            var: The `VariableAgent` to apply the policy to.

        Returns:
            A list of the new, damped messages.
        """
        k = self._get_damping()
        return [
            last_message * (1 - k) + curr_message * k
            for last_message, curr_message in sorted(
                zip(var.last_iteration, var.mailer.inbox), key=lambda x: x[0].recipient
            )
        ]

    @abstractmethod
    def _get_damping(self) -> float:
        """Returns the damping factor. Must be implemented by subclasses."""
        pass


class CostReductionPolicy(Policy):
    """An abstract base class for cost reduction policies.

    These policies are applied to factor nodes to modify their cost tables.
    """

    def __init__(self, factor_graph: FactorGraph):
        """Initializes the cost reduction policy.

        Args:
            factor_graph: The factor graph the policy will operate on.
        """
        super().__init__(PolicyType.FACTOR)
        self.factor = factor_graph

    def __call__(self) -> None:
        """Applies the cost reduction to the relevant factors."""
        mapping = self._get_reduction()
        for k, factor in mapping.items():
            factor.update_cost_table = factor.cost_table * k

    @abstractmethod
    def _get_reduction(self) -> Dict[float, FactorAgent]:
        """Returns a mapping from reduction factors to factor agents.

        Must be implemented by subclasses.
        """
        pass


class SplittingPolicy(Policy):
    """An abstract base class for factor splitting policies.

    These policies modify the graph structure itself, typically by splitting
    factor nodes.
    """

    def __init__(self, factor_graph: FactorGraph):
        """Initializes the splitting policy.

        Args:
            factor_graph: The factor graph the policy will operate on.
        """
        super().__init__(PolicyType.GRAPH)
        self.factor = factor_graph

    def __call__(self) -> Dict[float, FactorAgent]:
        """Applies the splitting policy to the graph.

        Returns:
            A dictionary mapping splitting factors to the affected factor agents.
        """
        mapping = self._get_splitting()
        for k, factor in mapping.items():
            factor.update_cost_table = factor.cost_table * k
        return mapping

    @abstractmethod
    def _get_splitting(self) -> Dict[float, FactorAgent]:
        """Returns a mapping from splitting factors to factor agents.

        Must be implemented by subclasses.
        """
        pass
