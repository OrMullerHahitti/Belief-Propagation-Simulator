"""Tools for creating and analyzing Backtrack Cost Trees (BCTs).

This module provides the `BCTCreator` class, which can take the history of a
simulation run and construct a BCT. A BCT is a tree structure that visualizes
how costs or beliefs from earlier iterations contribute to the final belief of
a specific variable, making it a powerful tool for debugging and understanding
the dynamics of belief propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class BCTNode:
    """Represents a single node in a Backtrack Cost Tree (BCT).

    Attributes:
        name: The name of the node, typically identifying the source agent.
        iteration: The simulation iteration this node corresponds to.
        cost: The belief or message value at this node.
        node_type: The type of the node (e.g., 'variable', 'factor').
        children: A list of child `BCTNode` objects.
        coefficient: The damping coefficient applied to this node's cost.
    """
    name: str
    iteration: int
    cost: float
    node_type: str  # 'variable', 'factor', 'cost'
    children: List["BCTNode"] = field(default_factory=list)
    coefficient: float = 1.0


class BCTCreator:
    """Creates, analyzes, and visualizes Backtrack Cost Trees (BCTs).

    This class uses the data collected in a `History` object to trace the
    dependencies of a variable's final belief back through the message-passing
    history of the simulation.
    """

    def __init__(self, factor_graph: Any, history: Any, damping_factor: float = 0.0):
        """Initializes the BCTCreator.

        Args:
            factor_graph: The `FactorGraph` object from the simulation.
            history: The `History` object containing the simulation run data.
            damping_factor: The damping factor used in the simulation, for
                correctly calculating cost contributions.
        """
        self.factor_graph = factor_graph
        self.history = history
        self.damping_factor = damping_factor
        self.bct_data = history.get_bct_data()
        self.bcts: Dict[str, BCTNode] = {}  # Cache for built BCTs

        print("BCTCreator initialized:")
        print(f"  - BCT mode: {history.use_bct_history}")
        print(f"  - Variables tracked: {len(self.bct_data.get('beliefs', {}))}")
        print(f"  - Message flows: {len(self.bct_data.get('messages', {}))}")
        print(f"  - Total steps: {self.bct_data.get('metadata', {}).get('total_steps', 0)}")

    def create_bct(self, variable_name: str, final_iteration: int = -1) -> BCTNode:
        """Creates a BCT for a specific variable, tracing its belief back in time.

        Args:
            variable_name: The name of the variable to create the BCT for.
            final_iteration: The final iteration to analyze. Defaults to -1 (the last one).

        Returns:
            The root `BCTNode` of the constructed tree.

        Raises:
            ValueError: If no data is found for the specified variable.
        """
        if variable_name not in self.bct_data.get("beliefs", {}):
            raise ValueError(f"Variable {variable_name} not found in history data")

        beliefs = self.bct_data["beliefs"][variable_name]
        if not beliefs:
            raise ValueError(f"No belief data found for {variable_name}")

        if final_iteration == -1 or final_iteration >= len(beliefs):
            final_iteration = len(beliefs) - 1
        final_belief = beliefs[final_iteration]

        root = BCTNode(
            name=f"{variable_name}_belief", iteration=final_iteration, cost=final_belief,
            node_type="variable", coefficient=self._get_damping_coefficient(final_iteration)
        )
        self._build_bct_recursive(root, variable_name, final_iteration)
        self.bcts[f"{variable_name}_{final_iteration}"] = root
        return root

    def _build_bct_recursive(self, node: BCTNode, variable_name: str, iteration: int) -> None:
        """Recursively builds the BCT by backtracking through message history."""
        if iteration <= 0:
            return

        messages = self.bct_data.get("messages", {})
        for flow_key, msg_values in messages.items():
            if "->" in flow_key:
                sender, recipient = flow_key.split("->")
                if recipient == variable_name and iteration - 1 < len(msg_values):
                    msg_cost = msg_values[iteration - 1]
                    sender_type = "factor" if "f" in sender.lower() else "variable"
                    child = BCTNode(
                        name=f"{sender}->msg", iteration=iteration - 1, cost=msg_cost,
                        node_type=sender_type, coefficient=self._get_damping_coefficient(iteration - 1)
                    )
                    child.cost *= child.coefficient
                    node.children.append(child)
                    self._build_bct_recursive(child, sender, iteration - 1)

    def _get_damping_coefficient(self, iteration: int) -> float:
        """Calculates the damping coefficient for a given iteration."""
        if self.damping_factor == 0.0:
            return 1.0
        return (1 - self.damping_factor) * (self.damping_factor ** max(0, iteration - 1))

    def analyze_convergence(self, variable_name: str) -> Dict[str, Any]:
        """Analyzes the convergence pattern for a single variable.

        Args:
            variable_name: The name of the variable to analyze.

        Returns:
            A dictionary containing convergence metrics like total change,
            average change, and whether the variable has stabilized.
        """
        if variable_name not in self.bct_data.get("beliefs", {}):
            return {"error": f"Variable {variable_name} not found"}
        beliefs = self.bct_data["beliefs"][variable_name]
        assignments = self.bct_data.get("assignments", {}).get(variable_name, [])
        if not beliefs: return {"error": "No belief data available"}

        changes = [abs(beliefs[i] - beliefs[i - 1]) for i in range(1, len(beliefs))]
        converged = len(changes) >= 3 and all(c < 0.001 for c in changes[-3:])
        convergence_iter = len(beliefs) - 3 if converged else -1
        assignment_stable = len(assignments) >= 3 and len(set(assignments[-3:])) == 1

        return {
            "variable": variable_name, "total_iterations": len(beliefs),
            "initial_belief": beliefs[0], "final_belief": beliefs[-1],
            "total_change": abs(beliefs[-1] - beliefs[0]) if len(beliefs) >= 2 else 0.0,
            "max_change": max(changes) if changes else 0.0,
            "average_change": sum(changes) / len(changes) if changes else 0.0,
            "converged": converged, "convergence_iteration": convergence_iter,
            "assignment_stable": assignment_stable, "belief_evolution": beliefs,
            "assignment_evolution": assignments, "changes": changes,
        }

    def visualize_bct(self, variable_name: str, iteration: int = -1, save_path: Optional[str] = None) -> plt.Figure:
        """Generates and optionally saves a visualization of a BCT.

        Args:
            variable_name: The name of the variable to visualize the BCT for.
            iteration: The final iteration to trace back from. Defaults to -1 (last).
            save_path: An optional path to save the generated figure.

        Returns:
            The `matplotlib.Figure` object of the visualization.
        """
        cache_key = f"{variable_name}_{iteration}"
        root = self.bcts.get(cache_key) or self.create_bct(variable_name, iteration)

        G = nx.DiGraph()
        pos, labels, colors = {}, {}, []
        queue = deque([(root, 0, 0)])
        node_map = {root: f"{root.name}_{root.iteration}_{id(root)}"}
        G.add_node(node_map[root])

        while queue:
            node, x, level = queue.popleft()
            node_id = node_map[node]
            pos[node_id] = (x, -level)
            label = f"{node.name}\niter:{node.iteration}\ncost:{node.cost:.3f}"
            if node.coefficient != 1.0: label += f"\ncoeff:{node.coefficient:.3f}"
            labels[node_id] = label
            colors.append("lightblue" if node.node_type == "variable" else "lightcoral" if node.node_type == "factor" else "lightgreen")

            num_children = len(node.children)
            start_x = x - (num_children - 1) / 2
            for i, child in enumerate(node.children):
                child_id = f"{child.name}_{child.iteration}_{id(child)}"
                node_map[child] = child_id
                G.add_edge(node_id, child_id)
                queue.append((child, start_x + i, level + 1))

        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, with_labels=False, node_color=colors, node_size=2500, arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, pos, labels, font_size=7)
        plt.title(f"BCT for {variable_name} at iteration {iteration}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"BCT visualization saved to: {save_path}")
        return plt.gcf()

    def compare_variables(self, variable_names: List[str]) -> Dict[str, Any]:
        """Performs a comparative analysis of convergence across multiple variables.

        Args:
            variable_names: A list of variable names to compare.

        Returns:
            A dictionary containing the analysis for each variable and a summary.
        """
        comparison = {"variables": variable_names, "analyses": {}, "summary": {}}
        for var_name in variable_names:
            if var_name in self.bct_data.get("beliefs", {}):
                comparison["analyses"][var_name] = self.analyze_convergence(var_name)
        if comparison["analyses"]:
            all_analyses = list(comparison["analyses"].values())
            comparison["summary"] = {
                "all_converged": all(a.get("converged", False) for a in all_analyses),
                "convergence_rates": [a.get("convergence_iteration", -1) for a in all_analyses],
                "final_beliefs": [a.get("final_belief", 0.0) for a in all_analyses],
                "total_changes": [a.get("total_change", 0.0) for a in all_analyses],
            }
        return comparison

    def export_analysis(self, output_file: str) -> None:
        """Exports the complete analysis for all variables to a JSON file.

        Args:
            output_file: The path where the JSON file will be saved.
        """
        analysis_data = {
            "metadata": {
                "damping_factor": self.damping_factor, "bct_mode": self.history.use_bct_history,
                "total_variables": len(self.bct_data.get("beliefs", {})),
                "total_steps": self.bct_data.get("metadata", {}).get("total_steps", 0),
            },
            "variable_analyses": {var: self.analyze_convergence(var) for var in self.bct_data.get("beliefs", {})},
            "global_data": {"costs": self.bct_data.get("costs", []), "message_flows": list(self.bct_data.get("messages", {}).keys())},
        }
        with open(output_file, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
        print(f"Complete analysis exported to: {output_file}")

    def print_summary(self) -> None:
        """Prints a human-readable summary of the BCT analysis to the console."""
        print("\n=== BCT Analysis Summary ===")
        print(f"History mode: {'BCT (step-by-step)' if self.history.use_bct_history else 'Legacy (cycle-based)'}")
        print(f"Damping factor: {self.damping_factor}")
        beliefs = self.bct_data.get("beliefs", {})
        print(f"Variables: {len(beliefs)}")
        print(f"Message flows: {len(self.bct_data.get('messages', {}))}")
        if beliefs:
            print("\nPer-variable analysis:")
            for var_name in beliefs:
                analysis = self.analyze_convergence(var_name)
                print(f"  {var_name}:")
                print(f"    Iterations: {analysis.get('total_iterations', 0)}")
                print(f"    Final belief: {analysis.get('final_belief', 0):.4f}")
                print(f"    Total change: {analysis.get('total_change', 0):.4f}")
                print(f"    Converged: {analysis.get('converged', False)}")
        costs = self.bct_data.get("costs", [])
        if costs:
            print("\nGlobal costs:")
            print(f"  Initial: {costs[0]:.2f}")
            print(f"  Final: {costs[-1]:.2f}")
            print(f"  Improvement: {costs[0] - costs[-1]:.2f}")


def quick_bct_analysis(factor_graph: Any, history: Any, variable_name: str, damping_factor: float = 0.0) -> Dict[str, Any]:
    """A convenience function for performing a quick BCT analysis on a variable.

    Args:
        factor_graph: The `FactorGraph` object.
        history: The `History` object from a simulation run.
        variable_name: The name of the variable to analyze.
        damping_factor: The damping factor used in the simulation.

    Returns:
        A dictionary containing the convergence analysis for the variable.
    """
    creator = BCTCreator(factor_graph, history, damping_factor)
    return creator.analyze_convergence(variable_name)


def quick_bct_visualization(factor_graph: Any, history: Any, variable_name: str, save_path: Optional[str] = None, damping_factor: float = 0.0) -> plt.Figure:
    """A convenience function for generating a quick BCT visualization.

    Args:
        factor_graph: The `FactorGraph` object.
        history: The `History` object from a simulation run.
        variable_name: The name of the variable to visualize.
        save_path: An optional path to save the generated figure.
        damping_factor: The damping factor used in the simulation.

    Returns:
        The `matplotlib.Figure` object of the visualization.
    """
    creator = BCTCreator(factor_graph, history, damping_factor)
    return creator.visualize_bct(variable_name, save_path=save_path)


def example_usage() -> None:
    """Provides an example of how to use the `BCTCreator` class."""
    print("BCTCreator ready for use! See the function's docstring for an example.")
    # Example usage pattern:
    #
    # # Assuming `my_graph` and `engine` (after a run) exist
    # creator = BCTCreator(my_graph, engine.history, damping_factor=0.2)
    # analysis = creator.analyze_convergence('x1')
    # print(f"Variable x1 converged: {analysis['converged']}")
    # creator.visualize_bct('x1', save_path='x1_bct.png')
    # comparison = creator.compare_variables(['x1', 'x2', 'x3'])
    # creator.export_analysis('complete_bct_analysis.json')
    # creator.print_summary()


if __name__ == "__main__":
    example_usage()
