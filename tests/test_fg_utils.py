import pytest
import numpy as np
import sys
import os
import logging

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fg_utils import create_ndarray

# Configure test logging
@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Add a StreamHandler to ensure logs are displayed to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    yield
    
    # Clean up handlers to prevent duplication
    root_logger.handlers.clear()

# Set up logger
logger = logging.getLogger(__name__)

def test_create_ndarray_shape():
    """Test that create_ndarray returns an array with the correct shape."""
    # Test with domain=3, num_connected=2
    domain = 3
    num_connected = 2
    result = create_ndarray(domain, num_connected)
    
    # Check shape
    expected_shape = (domain, domain)
    actual_shape = result.shape
    logger.info(f"Testing shape for domain={domain}, num_connected={num_connected}")
    logger.info(f"EXPECTED shape: {expected_shape}, ACTUAL shape: {actual_shape}")
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"
    
    # Test with domain=5, num_connected=4
    domain = 5
    num_connected = 4
    result = create_ndarray(domain, num_connected)
    
    # Check shape
    expected_shape = (domain, domain, domain, domain)
    actual_shape = result.shape
    logger.info(f"Testing shape for domain={domain}, num_connected={num_connected}")
    logger.info(f"EXPECTED shape: {expected_shape}, ACTUAL shape: {actual_shape}")
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"

def test_create_ndarray_values():
    """Test that the values in the array are integers between 0 and 9."""
    domain = 4
    num_connected = 3
    result = create_ndarray(domain, num_connected)
    
    # Check that all values are integers between 0 and 9
    logger.info(f"Testing values for domain={domain}, num_connected={num_connected}")
    
    # Check for integer dtype
    expected_is_integer = True
    actual_is_integer = np.issubdtype(result.dtype, np.integer)
    logger.info(f"EXPECTED result dtype is integer: {expected_is_integer}, ACTUAL: {actual_is_integer}")
    assert actual_is_integer, f"Expected integer dtype, got {result.dtype}"
    
    # Check for min value ≥ 0
    expected_min_range = 0
    actual_min = np.min(result)
    logger.info(f"EXPECTED min value >= {expected_min_range}, ACTUAL min value: {actual_min}")
    assert actual_min >= expected_min_range, f"Expected min value >= {expected_min_range}, got {actual_min}"
    
    # Check for max value ≤ 9
    expected_max_range = 9
    actual_max = np.max(result)
    logger.info(f"EXPECTED max value <= {expected_max_range}, ACTUAL max value: {actual_max}")
    assert actual_max <= expected_max_range, f"Expected max value <= {expected_max_range}, got {actual_max}"

def test_create_ndarray_different_calls():
    """Test that subsequent calls produce different arrays (randomness check)."""
    domain = 3
    num_connected = 2
    result1 = create_ndarray(domain, num_connected)
    result2 = create_ndarray(domain, num_connected)
    
    # The probability of two random arrays being identical is extremely low
    # This test could theoretically fail, but it's very unlikely
    logger.info(f"Testing randomness for domain={domain}, num_connected={num_connected}")
    logger.info(f"First array (sample): {result1.flatten()[:5]}")
    logger.info(f"Second array (sample): {result2.flatten()[:5]}")
    
    expected_outcome = "Arrays should be different"
    arrays_equal = np.array_equal(result1, result2)
    actual_outcome = "Arrays are same" if arrays_equal else "Arrays are different"
    logger.info(f"EXPECTED: {expected_outcome}, ACTUAL: {actual_outcome}")
    assert not arrays_equal, "Expected different arrays on subsequent calls, but got identical arrays"

if __name__ == '__main__':
    pytest.main(["-xvs", "--log-cli-level=INFO", __file__])
