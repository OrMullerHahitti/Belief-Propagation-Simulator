import logging
import os
import sys

import numpy as np
import pytest

# Configure logging once - remove the duplicate basicConfig call below
@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration
    )

logger = logging.getLogger(__name__)

# Add the parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ct_utils import _create_cost_table, normalize_cost_table, create_symmetric_cost_table


def test_cost_table_creation_with_randint():
    # Test with np.random.randint
    connections = 3
    domain = 4
    low = 0
    high = 10
    
    logger.info(f"Creating cost table with randint: connections={connections}, domain={domain}, low={low}, high={high}")
    
    cost_table = _create_cost_table(connections=connections, domain=domain, policy=np.random.randint, low=low,
                                    high=high)
    
    logger.info(f"Created cost table with shape {cost_table.shape}")
    logger.info(f"Cost table stats: min={cost_table.min()}, max={cost_table.max()}, mean={cost_table.mean()}")
    
    # Check shape
    assert cost_table.shape == (domain,) * connections
    
    # Check values are integers in the correct range
    assert np.all(cost_table >= low)
    assert np.all(cost_table < high)
    assert np.all(np.equal(np.mod(cost_table, 1), 0))  # Check all values are integers
    
    logger.info("All assertions passed for randint cost table")


def test_cost_table_creation_with_uniform():
    # Test with np.random.uniform
    connections = 2
    domain = 3
    low = 1.0
    high = 5.0
    
    logger.info(f"Creating cost table with uniform: connections={connections}, domain={domain}, low={low}, high={high}")
    
    cost_table = _create_cost_table(connections=connections, domain=domain, policy=np.random.uniform, low=low,
                                    high=high)
    
    logger.info(f"Created cost table with shape {cost_table.shape}")
    logger.info(f"Cost table stats: min={cost_table.min():.4f}, max={cost_table.max():.4f}, mean={cost_table.mean():.4f}")
    
    # Check shape
    assert cost_table.shape == (domain,) * connections
    
    # Check values are in the correct range
    assert np.all(cost_table >= low)
    assert np.all(cost_table <= high)
    
    logger.info("All assertions passed for uniform cost table")


def test_cost_table_creation_with_normal():
    # Test with np.random.normal
    connections = 2
    domain = 5
    loc = 0.0
    scale = 1.0
    
    logger.info(f"Creating cost table with normal distribution: connections={connections}, domain={domain}, loc={loc}, scale={scale}")
    
    cost_table = _create_cost_table(connections=connections, domain=domain, policy=np.random.normal, loc=loc,
                                    scale=scale)
    
    logger.info(f"Created cost table with shape {cost_table.shape}")
    logger.info(f"Cost table stats: min={cost_table.min():.4f}, max={cost_table.max():.4f}, mean={cost_table.mean():.4f}, std={cost_table.std():.4f}")
    
    # Check shape
    assert cost_table.shape == (domain,) * connections
    
    logger.info("All assertions passed for normal distribution cost table")


def test_symmetric_cost_table():
    n, m = 4, 4
    
    logger.info(f"Creating symmetric cost table with dimensions {n}x{m}")
    
    cost_table = create_symmetric_cost_table(n, m)
    
    logger.info(f"Created cost table with shape {cost_table.shape}")
    logger.info(f"Cost table stats: min={cost_table.min():.4f}, max={cost_table.max():.4f}, mean={cost_table.mean():.4f}")
    
    # Check shape
    assert cost_table.shape == (n, m)
    
    # Check symmetry
    is_symmetric = np.allclose(cost_table, cost_table.T)
    logger.info(f"Cost table symmetry check: {is_symmetric}")
    assert is_symmetric
    
    logger.info("All assertions passed for symmetric cost table")


def test_normalize_cost_table_2d():
    # Create a 2D cost table
    cost_table = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    
    logger.info(f"Normalizing 2D cost table with shape {cost_table.shape}")
    logger.info(f"Original cost table:\n{cost_table}")
    
    original_sum = np.sum(cost_table)
    logger.info(f"Original sum: {original_sum}")
    
    normalized = normalize_cost_table(cost_table.copy())
    logger.info(f"Normalized cost table:\n{normalized}")
    
    # Check total sum is preserved
    new_sum = np.sum(normalized)
    logger.info(f"New sum: {new_sum}")
    assert np.isclose(new_sum, original_sum)
    
    # Check sums along each dimension are equal
    for dim in range(len(normalized.shape)):
        sums = np.sum(normalized, axis=dim)
        logger.info(f"Sum along dimension {dim}: {sums}")
        assert np.allclose(sums, sums[0])  # All values in sums should be equal
    
    logger.info("All assertions passed for 2D normalization")


def test_normalize_cost_table_3d():
    # Create a 3D cost table
    cost_table = np.random.rand(3, 3, 3)
    
    logger.info(f"Normalizing 3D cost table with shape {cost_table.shape}")
    
    original_sum = np.sum(cost_table)
    logger.info(f"Original sum: {original_sum:.4f}")
    
    normalized = normalize_cost_table(cost_table.copy())
    
    # Check total sum is preserved
    new_sum = np.sum(normalized)
    logger.info(f"New sum: {new_sum:.4f}")
    assert np.isclose(new_sum, original_sum)
    
    # Check sums along each dimension are equal
    dim_sums = [np.sum(normalized, axis=i) for i in range(3)]
    for i, dim_sum in enumerate(dim_sums):
        flat_sum = dim_sum.flatten()
        logger.info(f"Dimension {i} sums: min={flat_sum.min():.4f}, max={flat_sum.max():.4f}")
        assert np.allclose(flat_sum, flat_sum[0])  # All values should be equal
    
    logger.info("All assertions passed for 3D normalization")


def test_invalid_inputs():
    # Test with non-int domain
    logger.info("Testing invalid domain type")
    with pytest.raises(ValueError) as excinfo:
        _create_cost_table(2, "invalid", np.random.randint, low=0, high=10)
    logger.info(f"Caught expected exception: {str(excinfo.value)}")
    
    # Test with non-callable policy
    logger.info("Testing non-callable policy")
    with pytest.raises(ValueError) as excinfo:
        _create_cost_table(2, 3, "not_callable")
    logger.info(f"Caught expected exception: {str(excinfo.value)}")
    
    logger.info("All assertions passed for invalid inputs")
