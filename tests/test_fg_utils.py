import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fg_utils import create_ndarray

class TestFGUtils(unittest.TestCase):
    
    def test_create_ndarray_shape(self):
        """Test that create_ndarray returns an array with the correct shape."""
        # Test with domain=3, num_connected=2
        domain = 3
        num_connected = 2
        result = create_ndarray(domain, num_connected)
        
        # Check shape
        self.assertEqual(result.shape, (domain, domain))
        
        # Test with domain=5, num_connected=4
        domain = 5
        num_connected = 4
        result = create_ndarray(domain, num_connected)
        
        # Check shape
        self.assertEqual(result.shape, (domain, domain, domain, domain))
    
    def test_create_ndarray_values(self):
        """Test that the values in the array are integers between 0 and 9."""
        domain = 4
        num_connected = 3
        result = create_ndarray(domain, num_connected)
        
        # Check that all values are integers between 0 and 9
        self.assertTrue(np.issubdtype(result.dtype, np.integer))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 9))
    
    def test_create_ndarray_different_calls(self):
        """Test that subsequent calls produce different arrays (randomness check)."""
        domain = 3
        num_connected = 2
        result1 = create_ndarray(domain, num_connected)
        result2 = create_ndarray(domain, num_connected)
        
        # The probability of two random arrays being identical is extremely low
        # This test could theoretically fail, but it's very unlikely
        self.assertFalse(np.array_equal(result1, result2))

if __name__ == '__main__':
    unittest.main()
