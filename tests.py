import unittest
from models import VariableNode, FactorNode, FactorGraph

class TestFactorGraph(unittest.TestCase):

    def setUp(self):
        # Setup a simple factor graph for testing
        self.X1 = VariableNode('X1', [0, 1, 2])
        self.X2 = VariableNode('X2', [0, 1, 2])
        self.X3 = VariableNode('X3', [0, 1, 2])

        def cost_F12(assignment):
            return 0 if assignment['X1'] == assignment['X2'] else 1

        def cost_F23(assignment):
            return 0 if assignment['X2'] == assignment['X3'] else 1

        def cost_F31(assignment):
            return 0 if assignment['X3'] == assignment['X1'] else 1

        self.F12 = FactorNode('F12', [self.X1, self.X2], cost_F12)
        self.F23 = FactorNode('F23', [self.X2, self.X3], cost_F23)
        self.F31 = FactorNode('F31', [self.X3, self.X1], cost_F31)

        self.factor_graph = FactorGraph([self.X1, self.X2, self.X3], [self.F12, self.F23, self.F31], alpha=0.0001, max_iterations=50)

    def test_initialization(self):
        self.assertEqual(len(self.factor_graph.variables), 3)
        self.assertEqual(len(self.factor_graph.factors), 3)

    def test_run_min_sum(self):
        self.factor_graph.run_min_sum()
        assignments = self.factor_graph.current_best_assignments()
        self.assertIn(assignments['X1'], [0, 1, 2])
        self.assertIn(assignments['X2'], [0, 1, 2])
        self.assertIn(assignments['X3'], [0, 1, 2])
        self.assertTrue(self.factor_graph.iteration_log[-1]['converged'])

    def test_factor_cost(self):
        assignment = {'X1': 0, 'X2': 0}
        self.assertEqual(self.factor_graph._factor_cost(self.F12, assignment), 0)
        assignment = {'X1': 0, 'X2': 1}
        self.assertEqual(self.factor_graph._factor_cost(self.F12, assignment), 1)

    def test_save_log(self):
        self.factor_graph.run_min_sum()
        self.factor_graph.save_log('test_log.json')
        with open('test_log.json', 'r') as f:
            log = f.read()
        self.assertIn('iteration', log)

class TestFactorGraphVisualization(unittest.TestCase):
    def setUp(self):
        # Create a simple factor graph for testing
        X1 = VariableNode('X1', [0, 1])
        X2 = VariableNode('X2', [0, 1])
        X3 = VariableNode('X3', [0, 1])

        def cost_F12(assignment):
            return 0 if assignment['X1'] == assignment['X2'] else 1

        def cost_F23(assignment):
            return 0 if assignment['X2'] == assignment['X3'] else 1

        def cost_F31(assignment):
            return 0 if assignment['X3'] == assignment['X1'] else 1

        F12 = FactorNode('F12', [X1, X2], cost_F12)
        F23 = FactorNode('F23', [X2, X3], cost_F23)
        F31 = FactorNode('F31', [X3, X1], cost_F31)

        self.factor_graph = FactorGraph([X1, X2, X3], [F12, F23, F31])

    def test_visualize(self):
        # Test if the visualize method runs without errors
        try:
            self.factor_graph.visualize()
        except Exception as e:
            self.fail(f"visualize() raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()