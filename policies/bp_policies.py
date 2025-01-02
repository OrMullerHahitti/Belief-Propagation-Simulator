# min_sum_rule.py
import numpy as np
import math
from message_rules import MessageUpdateRule


class MinSumUpdateRule(MessageUpdateRule):
    """
    Standard min-sum (like max-product in negative log domain).
    """

    def compute_Q(self, vName, fName, factor_graph, Q, R, iteration):
        # Q_{v->f}(x_v) = sum_{f' in N(v)\f} R_{f'->v}(x_v)
        v_node = factor_graph.get_variable_node(vName)
        msg = np.zeros(len(v_node.domain))

        for neighbor in v_node.neighbors:
            if neighbor.name != fName and neighbor.is_factor():
                msg += R[(neighbor.name, vName)]
        return msg

    def compute_R(self, fName, vName, factor_graph, Q, R, iteration):
        # R_{f->v}(x_v) = min_{ x_{u in N(f)\v} } [ factor_cost(...) + sum_{u != v} Q_{u->f}(x_u) ]
        f_node = factor_graph.get_factor_node(fName)
        v_node = factor_graph.get_variable_node(vName)
        var_index = f_node.var_names.index(vName)
        domain_size_v = len(v_node.domain)
        cost_out = np.zeros(domain_size_v)

        var_nodes = [factor_graph.get_variable_node(vn) for vn in f_node.var_names]
        domain_sizes = [len(vn.domain) for vn in var_nodes]

        all_indices = [range(ds) for ds in domain_sizes]

        for x_v_idx in range(domain_size_v):
            min_cost = math.inf

            def recurse(dim=0, assignment=[]):
                nonlocal min_cost
                if dim == len(var_nodes):
                    factor_cost = f_node.potential_table[tuple(assignment)]
                    sum_q = 0.0
                    for i, var_n in enumerate(var_nodes):
                        if i != var_index:
                            sum_q += Q[(var_n.name, fName)][assignment[i]]
                    total_cost = factor_cost + sum_q
                    if total_cost < min_cost:
                        min_cost = total_cost
                else:
                    if dim == var_index:
                        recurse(dim + 1, assignment + [x_v_idx])
                    else:
                        for val_idx in all_indices[dim]:
                            recurse(dim + 1, assignment + [val_idx])

            recurse(0, [])
            cost_out[x_v_idx] = min_cost

        return cost_out
