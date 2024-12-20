import itertools
import copy
import json
import networkx as nx
import matplotlib.pyplot as plt

class VariableNode:
    """
    Represents a variable node in the factor graph.
    A variable node represents a discrete random variable.
    """

    def __init__(self, name, domain):
        """
        Initialize a variable node.

        Parameters
        ----------
        name : str
            The unique identifier for the variable.
        domain : list
            The domain of possible values for this variable.
        """
        self.name = name
        self.domain = domain
        self.factors = []  # Will hold references to FactorNodes connected to this variable.

    def __repr__(self):
        return f"VariableNode(name={self.name}, domain={self.domain})"


class FactorNode:
    """
    Represents a factor node in the factor graph.
    A factor node defines a cost function over a subset of variables.
    """

    def __init__(self, name, variables, cost_function):
        """
        Initialize a factor node.

        Parameters
        ----------
        name : str
            Unique identifier for the factor.
        variables : list of VariableNode
            The variables that this factor node connects.
        cost_function : function or dict
            A mapping or function that returns cost given assignments to the variables.
            If using a dict-based cost function:
                Keys: tuples representing assignments (x_var1, x_var2, ...)
                Values: cost (float)
            Or a callable that given a dict {var_name: value} returns a cost.
        """
        self.name = name
        self.variables = variables
        self.cost_function = cost_function
        # Ensure all variable nodes know about this factor
        for v in variables:
            v.factors.append(self)

    def __repr__(self):
        var_names = [v.name for v in self.variables]
        return f"FactorNode(name={self.name}, variables={var_names})"


class FactorGraph:
    """
    Represents the factor graph and provides methods to run the Min-Sum algorithm.
    """

    def __init__(self, variables, factors, alpha=0.0001, max_iterations=100, convergence_threshold=1e-6):
        """
        Initialize the factor graph.

        Parameters
        ----------
        variables : list of VariableNode
            The variable nodes in the factor graph.
        factors : list of FactorNode
            The factor nodes in the factor graph.
        alpha : float
            A small constant to prevent cost escalation in variable-to-factor messages.
        max_iterations : int
            The maximum number of iterations to run if no convergence.
        convergence_threshold : float
            The threshold to determine convergence based on assignments.
        """
        self.variables = {v.name: v for v in variables}
        self.factors = {f.name: f for f in factors}
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Message data structure:
        # variable_to_factor_messages[(variable_name, factor_name)][value] = cost
        # factor_to_variable_messages[(factor_name, variable_name)][value] = cost
        self.variable_to_factor_messages = {}
        self.factor_to_variable_messages = {}

        # Initialize all messages to zero
        self._initialize_messages()

        # For logging iterations
        self.iteration_log = []

    def __repr__(self):
        return f"FactorGraph(variables={list(self.variables.keys())}, factors={list(self.factors.keys())})"
    def __str__(self):
        return f"FactorGraph(variables={list(self.variables.keys())}, factors={list(self.factors.keys())})"


    def _initialize_messages(self):
        """
        Initialize all messages to zero.
        """
        for f in self.factors.values():
            for v in f.variables:
                # Initialize factor->variable messages
                self.factor_to_variable_messages[(f.name, v.name)] = {val: 0.0 for val in v.domain}
                # Initialize variable->factor messages
                self.variable_to_factor_messages[(v.name, f.name)] = {val: 0.0 for val in v.domain}

    def _compute_factor_to_variable_message(self, factor, target_variable):
        """
        Compute the factor-to-variable message:
        R_{F->X}(x) = min_{PA\X} cost(<X,x>, PA\X) + sum(...) if needed.

        Since this is Min-Sum, we find the minimal cost assignment for each value x in target_variable's domain.
        """
        # Other variables in the factor except the target variable
        other_vars = [v for v in factor.variables if v.name != target_variable.name]

        # For each value of the target variable, we consider all assignments of other_vars
        # and find the minimal cost considering the incoming variable->factor messages.
        # Actually, in Min-Sum:
        # R_{F->X}(x) = min_{assignments_of_others} [ cost(X=x, others) + sum of Q_{other_variables->F}(other_value)]
        # The sum of Q_{other_variables->F}(other_value) is included in the factor's reasoning if needed.

        # Construct a list of domains for other variables
        domains = [ov.domain for ov in other_vars]

        new_message = {}
        for x_val in target_variable.domain:
            min_cost = float('inf')
            # Iterate over all combinations of other variables' values
            for combo in itertools.product(*domains):
                assignment = {target_variable.name: x_val}
                # Add other vars assignments
                for ov, val in zip(other_vars, combo):
                    assignment[ov.name] = val

                # Compute factor cost for this full assignment
                cost_value = self._factor_cost(factor, assignment)

                # Add the incoming messages from the other variables to factor node:
                # For each other variable ov:
                #   we add the Q_{ov->F}(val) = variable_to_factor_messages[(ov.name, factor.name)][val]
                # because in min-sum we consider the sum of incoming variable->factor messages as well.
                for ov, val in zip(other_vars, combo):
                    cost_value += self.variable_to_factor_messages[(ov.name, factor.name)][val]

                if cost_value < min_cost:
                    min_cost = cost_value

            new_message[x_val] = min_cost

        return new_message

    def _compute_variable_to_factor_message(self, variable, target_factor):
        """
        Compute the variable-to-factor message:
        Q_{X->F}(x) = sum_{F' in F_X, F' != F} R_{F'->X}(x) - alpha
        """
        connected_factors = variable.factors
        sum_messages = {x_val: 0.0 for x_val in variable.domain}

        for f_prime in connected_factors:
            if f_prime.name == target_factor.name:
                continue
            # Add the corresponding factor-to-variable message R_{F'->X}(x)
            for x_val in variable.domain:
                sum_messages[x_val] += self.factor_to_variable_messages[(f_prime.name, variable.name)][x_val]

        # Subtract alpha
        for x_val in variable.domain:
            sum_messages[x_val] -= self.alpha

        return sum_messages

    def _factor_cost(self, factor, assignment):
        """
        Given a factor and a dict of assignments {var_name: val}, return the cost.
        The factor's cost_function can be either a dict or a callable.
        """
        # Extract tuple of values in the order of factor.variables
        values_tuple = tuple(assignment[v.name] for v in factor.variables)

        if callable(factor.cost_function):
            return factor.cost_function(assignment)
        else:
            # Assuming dict-based cost function
            return factor.cost_function.get(values_tuple, float('inf'))

    def run_min_sum(self):
        """
        Run the Min-Sum algorithm until convergence or max_iterations is reached.
        """
        previous_assignments = self.current_best_assignments()

        for iteration in range(self.max_iterations):
            iteration_data = {
                'iteration': iteration,
                'variable_to_factor_messages': {},
                'factor_to_variable_messages': {},
                'assignments': {},
                'converged': False
            }

            # Update factor->variable messages
            new_factor_to_variable_messages = {}
            for (f_name, f_node) in self.factors.items():
                for v_node in f_node.variables:
                    new_factor_to_variable_messages[(f_name, v_node.name)] = self._compute_factor_to_variable_message(
                        f_node, v_node)

            # Update variable->factor messages
            new_variable_to_factor_messages = {}
            for (v_name, v_node) in self.variables.items():
                for f_node in v_node.factors:
                    new_variable_to_factor_messages[(v_name, f_node.name)] = self._compute_variable_to_factor_message(
                        v_node, f_node)

            # Commit updates
            self.factor_to_variable_messages = new_factor_to_variable_messages
            self.variable_to_factor_messages = new_variable_to_factor_messages

            # Compute current best assignments
            current_assigns = self.current_best_assignments()
            iteration_data['assignments'] = current_assigns

            # Check for convergence
            if self._check_convergence(previous_assignments, current_assigns):
                iteration_data['converged'] = True
                self.iteration_log.append(iteration_data)
                break

            # Prepare for next iteration
            previous_assignments = current_assigns

            # Log messages
            # Convert message dictionaries to serializable format
            iteration_data['variable_to_factor_messages'] = {
                str(k): v for k, v in self.variable_to_factor_messages.items()
            }
            iteration_data['factor_to_variable_messages'] = {
                str(k): v for k, v in self.factor_to_variable_messages.items()
            }

            self.iteration_log.append(iteration_data)

        else:
            # If we finish all iterations without convergence:
            # We might be dealing with a cycle or no convergence scenario.
            # Log that we didn't converge.
            self.iteration_log[-1]['converged'] = False

    def current_best_assignments(self):
        """
        Determine the best assignment for each variable based on the current factor-to-variable messages:
        x^ = argmin_x ∑_F R_{F->X}(x)
        """
        assignments = {}
        for v_name, v_node in self.variables.items():
            # Sum over factors connected to v_node
            best_val = None
            best_cost = float('inf')
            for x_val in v_node.domain:
                cost_val = 0.0
                for f_node in v_node.factors:
                    cost_val += self.factor_to_variable_messages[(f_node.name, v_name)][x_val]
                if cost_val < best_cost:
                    best_cost = cost_val
                    best_val = x_val
            assignments[v_name] = best_val
        return assignments

    def _check_convergence(self, old_assignments, new_assignments):
        """
        Check if assignments have converged.
        Convergence is declared if the assignments do not change or their difference is below a threshold.
        Here we assume discrete assignments, so we can check if assignments match exactly.
        """
        for v in old_assignments:
            if old_assignments[v] != new_assignments[v]:
                return False
        return True

    def save_log(self, filename='iteration_log.json'):
        """
        Save the iteration log as JSON.
        """
        with open(filename, 'w') as f:
            json.dump(self.iteration_log, f, indent=2)

    def visualize(self):
        """
        Visualize the factor graph using networkx and matplotlib.
        """
        G = nx.Graph()

        # Add variable nodes
        for var_name, var_node in self.variables.items():
            G.add_node(var_name, label=var_name, color='blue')

        # Add factor nodes and edges
        for factor_name, factor_node in self.factors.items():
            G.add_node(factor_name, label=factor_name, color='red')
            for var_node in factor_node.variables:
                G.add_edge(factor_name, var_node.name)

        # Draw the graph
        pos = nx.spring_layout(G)
        colors = [G.nodes[node]['color'] for node in G.nodes]
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, labels=labels, node_color=colors, with_labels=True, node_size=3000, font_size=10, font_color='white')
        plt.show()




# ------------------------
# Sample Usage / Test Code
# ------------------------
if __name__ == "__main__":
    # Example: A 3-variable cycle.
    # Variables: X1, X2, X3 each with domain {0,1,2}
    X1 = VariableNode('X1', [0, 1, 2])
    X2 = VariableNode('X2', [0, 1, 2])
    X3 = VariableNode('X3', [0, 1, 2])


    # Define factor cost functions
    # For simplicity, let's say each factor penalizes differing values between variables:
    # Cost(F12) = 0 if X1 == X2 else 1
    # Cost(F23) = 0 if X2 == X3 else 1
    # Cost(F31) = 0 if X3 == X1 else 1

    def cost_F12(assignment):
        return 0 if assignment['X1'] == assignment['X2'] else 1


    def cost_F23(assignment):
        return 0 if assignment['X2'] == assignment['X3'] else 1


    def cost_F31(assignment):
        return 0 if assignment['X3'] == assignment['X1'] else 1


    F12 = FactorNode('F12', [X1, X2], cost_F12)
    F23 = FactorNode('F23', [X2, X3], cost_F23)
    F31 = FactorNode('F31', [X3, X1], cost_F31)

    fg = FactorGraph([X1, X2, X3], [F12, F23, F31], alpha=0.0001, max_iterations=50)
    fg.run_min_sum()

    # Print results
    print("Final Assignments:", fg.current_best_assignments())
    print("Converged:", fg.iteration_log[-1]['converged'])
    # Save log if needed
    # fg.save_log('sample_run_log.json')
