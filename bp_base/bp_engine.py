from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Tuple, Any, Optional
import numpy as np
import networkx as nx
import json
import os
import logging
from bp_base.agents import BPAgent, VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.computators import MaxSumComputator
from bp_base.factor_graph import FactorGraph
from DCOP_base import Computator, Agent
from functools import reduce
from bp_base.typing import Policy, PolicyType
from dataclasses import dataclass, field

""" in this module we will implement the belief propagation with various policies with factor graph configs
most of which are implemented in the factor graph module and will be max-sum with different policies and different structures
we will start with the usual 3-cycle and then move to more complex structures"""

# Add logger configuration
logger = logging.getLogger(__name__)
# Basic configuration for demonstration. You might want to configure this more globally.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class Step:
    """
    A class to represent a step in the factor graph.
    """

    num: int = 0
    messages: Dict[str, List[Message]] = field(default_factory=dict)

    def add(self, agent: Agent, message: Message):
        """
        Adds a message to the list of messages to be sent by the specified agent.

        These messages are typically processed or sent in a subsequent phase of the step
        or the beginning of the next step.

        :param agent: The agent for whom the message is being recorded.
        :param message: The message to add.
        """
        if agent.name not in self.messages:
            self.messages[agent.name] = []
        self.messages[agent.name].append(message)


@dataclass
class Cycle:
    """
    A class to represent a cycle in the factor graph.
    A cycle consists of a sequence of steps, typically up to the graph diameter.
    """

    number: int
    steps: List[Step] = field(default_factory=list)
    global_cost: Optional[float] = None  # Store global cost at the end of this cycle

    def add(self, step: Step):
        """
        Add a step to the cycle.
        """
        self.steps.append(step)

    def __eq__(self, other: object):
        """
        Check if two cycles are equal based on their steps.
        """
        if not isinstance(other, Cycle):
            return NotImplemented
        if len(self.steps) != len(other.steps):
            return False
        for step1, step2 in zip(self.steps, other.steps):
            if step1.messages != step2.messages:
                return False
        return True


class History:
    def __init__(self, **kwargs):
        self.config = dict(kwargs)
        self.iterations: Dict[int, Step] = {}  # Stores each step; iteration_number -> Step object
        self.cycles: Dict[int, Cycle] = {}  # Stores cycles; cycle_number -> Cycle object
        self.beliefs: Dict[int, Dict[str, np.ndarray]] = {}  # Beliefs per iteration (step)
        self.assignments: Dict[int, Dict[str, int | float]] = {}  # Assignments per iteration (step)

    def __setitem__(self, iteration_key: int, step_value: Step):
        """Allows setting a step for a given iteration number."""
        self.iterations[iteration_key] = step_value

    def __getitem__(self, iteration_key: int) -> Step:
        """Allows getting a step for a given iteration number."""
        return self.iterations[iteration_key]

    def compare_last_two_iterations(self) -> bool:
        """Compares assignments of the last two recorded iterations (steps)."""
        if len(self.assignments) < 2:
            return False
        sorted_assignment_keys = sorted(self.assignments.keys())
        last_iteration_assignments = self.assignments[sorted_assignment_keys[-1]]
        second_last_iteration_assignments = self.assignments[sorted_assignment_keys[-2]]
        return list(last_iteration_assignments.values()) == list(second_last_iteration_assignments.values())

    @staticmethod
    def _normalize_for_json(obj: Any) -> Any:
        """Helper function to normalize objects for JSON serialization."""
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            # For dataclasses or other objects, convert to dict using vars()
            # Ensure recursive normalization for nested objects
            return {key: History._normalize_for_json(value) for key, value in vars(obj).items()}
        if isinstance(obj, dict):
            return {key: History._normalize_for_json(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [History._normalize_for_json(value) for value in obj]
        return obj

    @property
    def name(self):
        factor_graph_name = self.config.get('factor_graph_name', 'UnknownFactorGraph')
        computator_name = str(self.config.get('computator', 'UnknownComputator'))
        policies_repr = self.config.get('policies_repr', '')
        return f"{factor_graph_name}_{computator_name}_{policies_repr}"

    def save_results(self, filename: str = None) -> str:
        """
        Save history (iterations, cycles, beliefs, assignments) as pure-Python JSON.
        """
        if filename is None:
            base_name = self.name
            num_iterations = len(self.iterations)
            filename = f"{base_name}_details_iter_{num_iterations}.json"

        serializable_config = {}
        for k, v in self.config.items():
            if k == 'factor_graph':
                serializable_config[k] = v.name if hasattr(v, 'name') else str(v)
            elif k == 'computator':
                serializable_config[k] = v.__class__.__name__ if hasattr(v, '__class__') else str(v)
            elif k == 'policies':
                serializable_config[k] = {ptype.value if hasattr(ptype, 'value') else str(ptype): [p.__class__.__name__ for p in pol_list] for ptype, pol_list in v.items()} if v else None
            else:
                serializable_config[k] = v

        raw = {
            "name": self.name,
            "config_summary": serializable_config,
            "iterations_data": self.iterations,
            "cycles_data": self.cycles,
            "beliefs_per_iteration": self.beliefs,
            "assignments_per_iteration": self.assignments,
        }

        data = History._normalize_for_json(raw)

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"History saved to {filename}")
        return filename


### TODO: create a wrapper to config everything beforehand
### TODO: add a class to handle the policies and the history of the beliefs
### begining running the algorithm
class BPEngine:
    """
    Engine for belief propagation. Iterations are steps. Cycles group steps.
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        computator: Computator = MaxSumComputator(),
        policies: Dict[PolicyType, List[Policy]] | None = None,
        stopping_criteria_policies: List[Policy] | None = None,
    ):
        self.graph = factor_graph
        self.graph.set_computator(computator)
        self.policies = policies if policies else {}
        self.stopping_criteria_policies = stopping_criteria_policies if stopping_criteria_policies else []

        policies_str = self._get_policies_repr(self.policies)

        self.history = History(
            factor_graph_name=self.graph.name if hasattr(self.graph, 'name') else 'UnnamedGraph',
            computator=computator,
            policies=self.policies,
            policies_repr=policies_str,
            factor_graph=factor_graph
        )

        self.current_cycle_number = 0
        self.steps_in_current_cycle = 0
        self.cycle_length = max(1, self.graph.diameter if hasattr(self.graph, 'diameter') else 1)
        if not hasattr(self.graph, 'diameter'):
            logger.warning("FactorGraph does not have a 'diameter' attribute. Defaulting cycle_length to 1.")

    def _get_policies_repr(self, policies: Dict[PolicyType, List[Policy]]) -> str:
        """Generates a string representation for policies."""
        if not policies:
            return "NoPolicies"
        
        policies_repr_parts = []
        for ptype, pol_list in sorted(policies.items()):
            # Use ptype.value if it's an Enum, otherwise convert to string
            ptype_str = ptype.value if hasattr(ptype, 'value') else str(ptype)
            pol_names = ",".join(p.__class__.__name__ for p in pol_list)
            policies_repr_parts.append(f"{ptype_str}-{pol_names}")
        return "_".join(policies_repr_parts)

    def step(self, i: int = 0) -> Step:
        step_obj = Step(i)

        for var_agent in self.graph.get_variable_agents():
            var_agent.compute_messages()

        for var_agent in self.graph.get_variable_agents():
            var_agent.mailer.send()
            var_agent.mailer.prepare()
            var_agent.empty_mailbox()

        for factor_agent in self.graph.get_factor_agents():
            factor_agent.compute_messages()

        for factor_agent in self.graph.get_factor_agents():
            factor_agent.mailer.send()
            for message in factor_agent.mailer.outbox:
                step_obj.add(message.recipient, message)
            factor_agent.mailer.prepare()
            factor_agent.empty_mailbox()

        if PolicyType.MESSAGE in self.policies:
            for agent in self.graph.G.nodes():
                pass  # TODO: Implement message policy application logic here
        
        return step_obj

    def run(
        self, max_iter: int = 1000, save_json: bool = True, filename: str = None
    ) -> None:
        logger.info(f"Starting BPEngine.run: max_iter (steps) = {max_iter}, cycle_length = {self.cycle_length}")

        for i in range(max_iter):
            current_step_obj = self.step(i)
            self.history[i] = current_step_obj
            self.history.beliefs[i] = self.get_beliefs()
            self.history.assignments[i] = self.assignments

            self.steps_in_current_cycle += 1

            if self.current_cycle_number not in self.history.cycles:
                self.history.cycles[self.current_cycle_number] = Cycle(number=self.current_cycle_number)

            self.history.cycles[self.current_cycle_number].add(current_step_obj)

            if self._is_converged():
                logger.info(f"Converged after {i + 1} steps (iteration {i}).")
                if self.history.cycles[self.current_cycle_number].global_cost is None:
                    final_cost = self.calculate_global_cost()
                    self.history.cycles[self.current_cycle_number].global_cost = final_cost
                    logger.info(f"Global cost at convergence (cycle {self.current_cycle_number}, step {i+1}): {final_cost}")
                break

            if self.steps_in_current_cycle == self.cycle_length:
                current_global_cost = self.calculate_global_cost()
                self.history.cycles[self.current_cycle_number].global_cost = current_global_cost
                logger.info(f"Cycle {self.current_cycle_number} completed at step {i + 1}. Global cost: {current_global_cost}")

                self.current_cycle_number += 1
                self.steps_in_current_cycle = 0
        else:
            logger.info(f"Reached max_iter = {max_iter} steps without convergence.")
            if self.steps_in_current_cycle > 0 and self.history.cycles[self.current_cycle_number].global_cost is None:
                final_cost = self.calculate_global_cost()
                self.history.cycles[self.current_cycle_number].global_cost = final_cost
                logger.info(f"Global cost at max_iter (partial cycle {self.current_cycle_number}, {self.steps_in_current_cycle} steps): {final_cost}")

        if save_json:
            self.history.save_results(filename)
        return None

    def get_beliefs(self) -> Dict[str, np.ndarray]:
        beliefs = {}
        for node in self.graph.G.nodes():
            if isinstance(node, VariableAgent):
                beliefs[node.name] = getattr(node, "belief", None)
        return beliefs

    def _is_converged(self) -> bool:
        converged_by_assignment = self.history.compare_last_two_iterations()
        if converged_by_assignment:
            return True

        if self.stopping_criteria_policies:
            for policy in self.stopping_criteria_policies:
                if policy(self):
                    logger.info(f"Converged due to policy: {policy.__class__.__name__}")
                    return True
        return False

    @property
    def assignments(self) -> Dict[str, int | float]:
        return {
            node.name: node.curr_assignment
            for node in self.graph.G.nodes()
            if isinstance(node, VariableAgent)
        }

    def calculate_global_cost(self) -> float:
        total_cost = 0.0
        current_assignments = self.assignments  # Get current assignments for all variables

        for factor_node in self.graph.get_factor_agents():
            if factor_node.cost_table is None:
                logger.warning(f"Factor {factor_node.name} has no cost table. Skipping in global cost calculation.")
                continue

            try:
                if not hasattr(factor_node, 'connection_map') or not factor_node.connection_map:
                    logger.error(f"Factor {factor_node.name} does not have a valid 'connection_map'. Cannot calculate its contribution to global cost.")
                    continue

                num_dims = factor_node.cost_table.ndim
                indices = [0] * num_dims 
                
                found_all_indices = True
                for var_agent, dim_idx in factor_node.connection_map.items():
                    if var_agent.name not in current_assignments:
                        logger.warning(f"Variable {var_agent.name} (connected to factor {factor_node.name}) not in current assignments. Skipping factor cost.")
                        found_all_indices = False
                        break
                    assignment_value = current_assignments[var_agent.name]
                    
                    if not isinstance(assignment_value, int):
                        try:
                            assignment_value = int(assignment_value)
                        except (ValueError, TypeError):
                            logger.error(f"Assignment for variable {var_agent.name} is not a valid integer index: {assignment_value}. Skipping factor cost.")
                            found_all_indices = False
                            break
                    
                    if dim_idx >= num_dims:
                        logger.error(f"Dimension index {dim_idx} for variable {var_agent.name} in factor {factor_node.name} is out of bounds for cost table with {num_dims} dimensions.")
                        found_all_indices = False
                        break
                    indices[dim_idx] = assignment_value
                
                if not found_all_indices:
                    continue

                cost_for_factor = factor_node.cost_table[tuple(indices)]
                total_cost += cost_for_factor
            except IndexError as e:
                logger.error(f"IndexError accessing cost table for factor {factor_node.name} with assignments {indices}: {e}")
            except Exception as e:
                logger.error(f"Error calculating cost for factor {factor_node.name}: {e}")
        
        return total_cost
