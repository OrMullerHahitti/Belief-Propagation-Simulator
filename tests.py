import unittest
import pandas as pd
from services import CostTable

class TestCostTable(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame to use in tests
        self.df = pd.DataFrame({
            0: {0: 1, 1: 2, 2: 3},
            1: {0: 4, 1: 5, 2: 6}
        })
        self.cost_table = CostTable(self.df)

    def test_getitem_int(self):
        # Test integer indexing
        self.assertEqual(self.cost_table[1][2], 6)

    def test_getitem_str(self):
        # Test string indexing (assuming the DataFrame has string columns)
        df_str = pd.DataFrame({
            'a': {'b': 1, 'c': 2},
            'd': {'b': 3, 'c': 4}
        })
        cost_table_str = CostTable(df_str)
        self.assertEqual(cost_table_str['a']['b'], 1)

    def test_getitem_invalid(self):
        # Test invalid indexing
        with self.assertRaises(KeyError):
            _ = self.cost_table['invalid']

if __name__ == '__main__':
    unittest.main()
