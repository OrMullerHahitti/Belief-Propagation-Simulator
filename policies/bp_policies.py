# protocols
from abc import abstractmethod
from enum import Enum
from typing import List, Tuple, Dict

from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.factor_graph import FactorGraph


# This file is part of the BpPolicies package.

class PolicyType(Enum):
    FACTOR = "factor"
    VARIABLE = "variable"
    MESSAGE = "message"

class Policy:
    """
    Abstract base class for policies.
    """
    def __init__(self, policy_type: PolicyType):
        self.policy_type = policy_type

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class DampingPolicy(Policy):
    """
    Abstract base class for damping policies.
    """
    def __init__(self):
        super().__init__(PolicyType.VARIABLE)


    def __call__(self, var:VariableAgent)->List[Message]:
        k = self._get_damping()
        return [last_message * (1 - k) + curr_message * k for last_message,curr_message in sorted(zip(var.last_iteration, var.mailer.inbox), key = lambda x:x[0].recipient)]

    @abstractmethod
    def _get_damping(self)->float:
        pass

class CostReductionPolicy(Policy):
    """
    Abstract base class for cost reduction policies.
    """
    def __init__(self, factor_graph: FactorGraph):
        super().__init__(PolicyType.FACTOR)
        self.factor = factor_graph


    def __call__(self)->None:
        mapping = self._get_reduction()
        for k, factor in mapping.items():
            factor.update_cost_table = factor.cost_table * k
    @abstractmethod
    def _get_reduction(self)->Dict[float,FactorAgent]:
        pass


