"""
BCTCreator - Creates BCTs from History and FactorGraph

Takes the enhanced History object and FactorGraph, builds BCT trees,
and provides analysis and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from collections import defaultdict, deque


@dataclass
class BCTNode:
    """Node in a Backtrack Cost Tree"""
    name: str
    iteration: int
    cost: float
    node_type: str  # 'variable', 'factor', 'cost'
    children: List['BCTNode'] = None
    coefficient: float = 1.0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class BCTCreator:
    """Creates and analyzes BCTs from History and FactorGraph"""

    def __init__(self, factor_graph, history, damping_factor: float = 0.0):
        """
        Initialize BCTCreator

        Args:
            factor_graph: Your FactorGraph object
            history: Enhanced History object (with or without BCT data)
            damping_factor: Damping factor used in simulation
        """
        self.factor_graph = factor_graph
        self.history = history
        self.damping_factor = damping_factor
        self.bct_data = history.get_bct_data()
        self.bcts = {}  # Cache for built BCTs

        print(f"BCTCreator initialized:")
        print(f"  - BCT mode: {history.use_bct_history}")
        print(f"  - Variables tracked: {len(self.bct_data.get('beliefs', {}))}")
        print(f"  - Message flows: {len(self.bct_data.get('messages', {}))}")
        print(f"  - Total steps: {self.bct_data.get('metadata', {}).get('total_steps', 0)}")

    def create_bct(self, variable_name: str, final_iteration: int = -1) -> BCTNode:
        """
        Create BCT for a specific variable

        Args:
            variable_name: Variable to create BCT for
            final_iteration: Which iteration to analyze (-1 for last)

        Returns:
            BCTNode: Root of the BCT
        """
        if variable_name not in self.bct_data.get('beliefs', {}):
            raise ValueError(f"Variable {variable_name} not found in history data")

        beliefs = self.bct_data['beliefs'][variable_name]
        if not beliefs:
            raise ValueError(f"No belief data found for {variable_name}")

        # Handle final_iteration
        if final_iteration == -1 or final_iteration >= len(beliefs):
            final_iteration = len(beliefs) - 1

        final_belief = beliefs[final_iteration]

        # Create root node
        root = BCTNode(
            name=f"{variable_name}_belief",
            iteration=final_iteration,
            cost=final_belief,
            node_type='variable',
            coefficient=self._get_damping_coefficient(final_iteration)
        )

        # Build tree recursively
        self._build_bct_recursive(root, variable_name, final_iteration)

        # Cache the BCT
        cache_key = f"{variable_name}_{final_iteration}"
        self.bcts[cache_key] = root

        return root

    def _build_bct_recursive(self, node: BCTNode, variable_name: str, iteration: int):
        """Recursively build BCT tree"""
        if iteration <= 0:
            return

        # Find incoming messages to this variable at this iteration
        messages = self.bct_data.get('messages', {})

        for flow_key, msg_values in messages.items():
            if '->' in flow_key:
                sender, recipient = flow_key.split('->')

                # Check if this message is TO our variable
                if recipient == variable_name and iteration <= len(msg_values):

                    # Get message value from previous iteration
                    if iteration > 0 and (iteration - 1) < len(msg_values):
                        msg_cost = msg_values[iteration - 1]
                    else:
                        continue

                    # Determine node type (factor or variable based on sender name)
                    sender_type = 'factor' if 'f' in sender.lower() else 'variable'

                    # Create child node
                    child = BCTNode(
                        name=f"{sender}->msg",
                        iteration=iteration - 1,
                        cost=msg_cost,
                        node_type=sender_type,
                        coefficient=self._get_damping_coefficient(iteration - 1)
                    )

                    # Apply damping to cost
                    child.cost *= child.coefficient

                    node.children.append(child)

                    # Recurse for earlier iterations
                    self._build_bct_recursive(child, sender, iteration - 1)

    def _get_damping_coefficient(self, iteration: int) -> float:
        """Calculate damping coefficient for given iteration"""
        if self.damping_factor == 0.0:
            return 1.0
        return (1 - self.damping_factor) * (self.damping_factor ** max(0, iteration - 1))

    def analyze_convergence(self, variable_name: str) -> Dict:
        """
        Analyze convergence pattern for a variable

        Args:
            variable_name: Variable to analyze

        Returns:
            Convergence analysis dictionary
        """
        if variable_name not in self.bct_data.get('beliefs', {}):
            return {'error': f'Variable {variable_name} not found'}

        beliefs = self.bct_data['beliefs'][variable_name]
        assignments = self.bct_data.get('assignments', {}).get(variable_name, [])

        if not beliefs:
            return {'error': 'No belief data available'}

        # Calculate belief changes
        changes = []
        for i in range(1, len(beliefs)):
            change = abs(beliefs[i] - beliefs[i - 1])
            changes.append(change)

        # Detect convergence
        converged = False
        convergence_iter = -1

        if len(changes) >= 3:
            # Check if last 3 changes are small
            recent_changes = changes[-3:]
            if all(change < 0.001 for change in recent_changes):
                converged = True
                convergence_iter = len(beliefs) - 3

        # Assignment stability
        assignment_stable = False
        if assignments and len(assignments) >= 3:
            recent_assignments = assignments[-3:]
            assignment_stable = len(set(recent_assignments)) == 1

        return {
            'variable': variable_name,
            'total_iterations': len(beliefs),
            'initial_belief': beliefs[0],
            'final_belief': beliefs[-1],
            'total_change': abs(beliefs[-1] - beliefs[0]) if len(beliefs) >= 2 else 0.0,
            'max_change': max(changes) if changes else 0.0,
            'average_change': sum(changes) / len(changes) if changes else 0.0,
            'converged': converged,
            'convergence_iteration': convergence_iter,
            'assignment_stable': assignment_stable,
            'belief_evolution': beliefs,
            'assignment_evolution': assignments,
            'changes': changes
        }

    def visualize_bct(self, variable_name: str, iteration: int = -1, save_path: str = None) -> plt.Figure:
        """
        Visualize BCT for a variable

        Args:
            variable_name: Variable to visualize
            iteration: Which iteration to visualize (-1 for last)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        # Get or create BCT
        cache_key = f"{variable_name}_{iteration}"
        if cache_key in self.bcts:
            root = self.bcts[cache_key]
        else:
            root = self.create_bct(variable_name, iteration)

        # Build networkx graph
        G = nx.DiGraph()
        pos = {}
        labels = {}
        colors = []

        def add_nodes_recursive(node: BCTNode, x: int = 0, y: int = 0, level: int = 0):
            node_id = f"{node.name}_{node.iteration}_{id(node)}"
            G.add_node(node_id)

            # Position nodes
            pos[node_id] = (x, -level)

            # Create label
            label = f"{node.name}\niter:{node.iteration}\ncost:{node.cost:.3f}"
            if node.coefficient != 1.0:
                label += f"\ncoeff:{node.coefficient:.3f}"
            labels[node_id] = label

            # Color by type
            if node.node_type == 'variable':
                colors.append('lightblue')
            elif node.node_type == 'factor':
                colors.append('lightcoral')
            else:
                colors.append('lightgreen')

            # Add children
            for i, child in enumerate(node.children):
                child_id = f"{child.name}_{child.iteration}_{id(child)}"
                G.add_edge(node_id, child_id)
                add_nodes_recursive(child, x + i - len(node.children) / 2, y, level + 1)

        add_nodes_recursive(root)

        # Create plot
        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, with_labels=False, node_color=colors,
                node_size=2500, arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, pos, labels, font_size=7)

        plt.title(f"BCT for {variable_name} at iteration {iteration}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"BCT visualization saved to: {save_path}")

        return plt.gcf()

    def compare_variables(self, variable_names: List[str]) -> Dict:
        """
        Compare convergence across multiple variables

        Args:
            variable_names: List of variables to compare

        Returns:
            Comparison analysis
        """
        comparison = {
            'variables': variable_names,
            'analyses': {},
            'summary': {}
        }

        for var_name in variable_names:
            if var_name in self.bct_data.get('beliefs', {}):
                analysis = self.analyze_convergence(var_name)
                comparison['analyses'][var_name] = analysis

        # Create summary
        if comparison['analyses']:
            all_analyses = list(comparison['analyses'].values())

            comparison['summary'] = {
                'all_converged': all(a.get('converged', False) for a in all_analyses),
                'convergence_rates': [a.get('convergence_iteration', -1) for a in all_analyses],
                'final_beliefs': [a.get('final_belief', 0.0) for a in all_analyses],
                'total_changes': [a.get('total_change', 0.0) for a in all_analyses]
            }

        return comparison

    def export_analysis(self, output_file: str):
        """
        Export complete analysis to JSON

        Args:
            output_file: Path to save JSON file
        """
        analysis_data = {
            'metadata': {
                'damping_factor': self.damping_factor,
                'bct_mode': self.history.use_bct_history,
                'total_variables': len(self.bct_data.get('beliefs', {})),
                'total_steps': self.bct_data.get('metadata', {}).get('total_steps', 0)
            },
            'variable_analyses': {},
            'global_data': {
                'costs': self.bct_data.get('costs', []),
                'message_flows': list(self.bct_data.get('messages', {}).keys())
            }
        }

        # Analyze each variable
        for var_name in self.bct_data.get('beliefs', {}):
            analysis_data['variable_analyses'][var_name] = self.analyze_convergence(var_name)

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        print(f"Complete analysis exported to: {output_file}")

    def print_summary(self):
        """Print a summary of the BCT analysis"""
        print("\n=== BCT Analysis Summary ===")
        print(f"History mode: {'BCT (step-by-step)' if self.history.use_bct_history else 'Legacy (cycle-based)'}")
        print(f"Damping factor: {self.damping_factor}")

        beliefs = self.bct_data.get('beliefs', {})
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

        costs = self.bct_data.get('costs', [])
        if costs:
            print(f"\nGlobal costs:")
            print(f"  Initial: {costs[0]:.2f}")
            print(f"  Final: {costs[-1]:.2f}")
            print(f"  Improvement: {costs[0] - costs[-1]:.2f}")


# Quick usage functions
def quick_bct_analysis(factor_graph, history, variable_name: str, damping_factor: float = 0.0) -> Dict:
    """
    Quick BCT analysis - one function call

    Args:
        factor_graph: Your FactorGraph
        history: Enhanced History object
        variable_name: Variable to analyze
        damping_factor: Damping factor used

    Returns:
        Analysis dictionary
    """
    creator = BCTCreator(factor_graph, history, damping_factor)
    return creator.analyze_convergence(variable_name)


def quick_bct_visualization(factor_graph, history, variable_name: str,
                            save_path: str = None, damping_factor: float = 0.0) -> plt.Figure:
    """
    Quick BCT visualization - one function call

    Args:
        factor_graph: Your FactorGraph
        history: Enhanced History object
        variable_name: Variable to visualize
        save_path: Optional save path
        damping_factor: Damping factor used

    Returns:
        matplotlib Figure
    """
    creator = BCTCreator(factor_graph, history, damping_factor)
    return creator.visualize_bct(variable_name, save_path=save_path)


# Example usage
def example_usage():
    """Example of how to use BCTCreator"""

    # Your existing code:
    # engine = BPEngine(factor_graph=my_graph, use_bct_history=True)  # Enable BCT mode
    # engine.run(max_iter=20)

    # Create BCT analysis:
    # creator = BCTCreator(my_graph, engine.history, damping_factor=0.2)
    #
    # # Analyze convergence
    # analysis = creator.analyze_convergence('x1')
    # print(f"Variable x1 converged: {analysis['converged']}")
    #
    # # Visualize BCT
    # creator.visualize_bct('x1', save_path='x1_bct.png')
    #
    # # Compare variables
    # comparison = creator.compare_variables(['x1', 'x2', 'x3'])
    #
    # # Export complete analysis
    # creator.export_analysis('complete_bct_analysis.json')
    #
    # # Print summary
    # creator.print_summary()

    print("BCTCreator ready for use!")


if __name__ == "__main__":
    example_usage()