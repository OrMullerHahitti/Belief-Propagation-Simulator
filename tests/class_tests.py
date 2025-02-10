import unittest
from abstract_base.interfaces import NeighborAddingPolicy
from policies.node_add_policy import BPNeighborAddingPolicy
from bp_variations.node import VariableNode, FactorNode

class TestNode(unittest.TestCase):

    def setUp(self):
        self.var_node = VariableNode(name="Variable1", domain=[0, 1])
        self.factor_node = FactorNode(name="Factor1", potential_table=[[0.1, 0.9], [0.8, 0.2]], var_names=["Variable1"])

    def test_add_neighbor_variable_node(self):
        self.var_node.add_neighbor(self.factor_node)
        self.assertIn(self.factor_node, self.var_node.neighbors)

    def test_add_neighbor_factor_node(self):
        self.factor_node.add_neighbor(self.var_node)
        self.assertIn(self.var_node, self.factor_node.neighbors)

if __name__ == '__main__':
    unittest.main()