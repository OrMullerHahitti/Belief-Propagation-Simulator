import pytest
import numpy as np
import sys
import os
import logging

# Add the parent directory to the path to import policies
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policies.cost_reduction import (
    EveryTimeCostReduction,
    MinMaxEnvelopeCostReduction,
    EnvelopeBasedCostReduction,
)
from DCOP_base import Agent


# Configure test logging
@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Add a StreamHandler to ensure logs are displayed to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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


class TestCostReductionPolicy:
    """Test suite for cost reduction policies."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Create a mock Agent for testing
        self.agent = Agent(name="TestAgent", node_type="factor")

        # Sample cost table and message data for envelope calculations
        self.cost_table = np.array([[5.0, 2.0, 8.0], [1.0, 3.0, 4.0], [7.0, 0.0, 6.0]])

        self.message_data = np.array([0.0, 1.0, 2.0])

        logger.info("Test environment setup complete")

    def test_every_time_cost_reduction(self):
        """Test EveryTimeCostReduction policy."""
        logger.info("Testing EveryTimeCostReduction policy")

        # Create policy
        policy = EveryTimeCostReduction()

        # Test should_apply returns True for all iterations
        iterations_to_test = [0, 1, 2, 10, 100]
        for iteration in iterations_to_test:
            expected = True
            actual = policy.should_apply(iteration)
            logger.info(
                f"Iteration {iteration} - EXPECTED: {expected}, ACTUAL: {actual}"
            )
            assert actual == expected, f"should_apply failed for iteration {iteration}"

        # Test get_K returns the correct value
        expected_k = 0.5
        actual_k = policy.get_K()
        logger.info(f"get_K() - EXPECTED: {expected_k}, ACTUAL: {actual_k}")
        assert actual_k == expected_k, f"Expected K={expected_k}, got K={actual_k}"

    def test_min_max_envelope_cost_reduction(self):
        """Test MinMaxEnvelopeCostReduction policy."""
        logger.info("Testing MinMaxEnvelopeCostReduction policy")

        # Create policy
        policy = MinMaxEnvelopeCostReduction()

        # Test should_apply returns True only for even iterations
        even_iterations = [0, 2, 4, 10, 100]
        odd_iterations = [1, 3, 5, 11, 101]

        for iteration in even_iterations:
            expected = True
            actual = policy.should_apply(iteration)
            logger.info(
                f"Even iteration {iteration} - EXPECTED: {expected}, ACTUAL: {actual}"
            )
            assert (
                actual == expected
            ), f"should_apply failed for even iteration {iteration}"

        for iteration in odd_iterations:
            expected = False
            actual = policy.should_apply(iteration)
            logger.info(
                f"Odd iteration {iteration} - EXPECTED: {expected}, ACTUAL: {actual}"
            )
            assert (
                actual == expected
            ), f"should_apply failed for odd iteration {iteration}"

        # Test get_K returns the correct value
        expected_k = 0.5
        actual_k = policy.get_K()
        logger.info(f"get_K() - EXPECTED: {expected_k}, ACTUAL: {actual_k}")
        assert actual_k == expected_k, f"Expected K={expected_k}, got K={actual_k}"

    def test_envelope_based_cost_reduction(self):
        """Test EnvelopeBasedCostReduction policy."""
        logger.info("Testing EnvelopeBasedCostReduction policy")

        # Create policy with default parameters
        policy = EnvelopeBasedCostReduction(min_k=0.1, max_k=0.9, num_samples=20)

        # Test should_apply without criteria returns True for all iterations
        iterations_to_test = [0, 1, 2, 10, 100]
        for iteration in iterations_to_test:
            expected = True
            actual = policy.should_apply(iteration)
            logger.info(
                f"Iteration {iteration} - EXPECTED: {expected}, ACTUAL: {actual}"
            )
            assert actual == expected, f"should_apply failed for iteration {iteration}"

        # Create policy with custom criteria (apply only on odd iterations)
        custom_policy = EnvelopeBasedCostReduction(
            applying_critiria=lambda x: x % 2 == 1, min_k=0.1, max_k=0.9
        )

        # Test custom criteria
        even_iterations = [0, 2, 4, 10, 100]
        odd_iterations = [1, 3, 5, 11, 101]

        for iteration in even_iterations:
            expected = False
            actual = custom_policy.should_apply(iteration)
            logger.info(
                f"Custom policy, even iteration {iteration} - EXPECTED: {expected}, ACTUAL: {actual}"
            )
            assert (
                actual == expected
            ), f"Custom policy should_apply failed for even iteration {iteration}"

        for iteration in odd_iterations:
            expected = True
            actual = custom_policy.should_apply(iteration)
            logger.info(
                f"Custom policy, odd iteration {iteration} - EXPECTED: {expected}, ACTUAL: {actual}"
            )
            assert (
                actual == expected
            ), f"Custom policy should_apply failed for odd iteration {iteration}"

        # Test get_K with cost table and message data
        k_value = policy.get_K(self.cost_table, self.message_data)
        logger.info(f"get_K() returned: {k_value}")

        # K should be within the specified bounds
        assert (
            k_value >= policy.min_k and k_value <= policy.max_k
        ), f"K value {k_value} is out of bounds [{policy.min_k}, {policy.max_k}]"

        # Test that get_K returns the last_k value when no data is provided
        last_k = policy.last_k
        result_k = policy.get_K()
        logger.info(f"get_K() without data - EXPECTED: {last_k}, ACTUAL: {result_k}")
        assert result_k == last_k, f"Expected K={last_k}, got K={result_k}"

    def test_get_optimal_k_segments(self):
        """Test the get_optimal_k_segments method in EnvelopeBasedCostReduction."""
        logger.info("Testing get_optimal_k_segments method")

        policy = EnvelopeBasedCostReduction()

        # Get segments of the envelope
        segments = policy.get_optimal_k_segments(self.cost_table, self.message_data)

        logger.info(f"Found {len(segments)} optimal segments: {segments}")

        # Each segment should be a tuple (k_start, k_end, row_index, col_index)
        for i, segment in enumerate(segments):
            assert len(segment) == 4, f"Segment {i} has wrong format: {segment}"
            k_start, k_end, row, col = segment

            # k values should be between 0 and 1
            assert 0 <= k_start <= 1, f"k_start {k_start} out of range [0,1]"
            assert 0 <= k_end <= 1, f"k_end {k_end} out of range [0,1]"

            # Indices should be valid for the cost table
            assert 0 <= row < self.cost_table.shape[0], f"Row index {row} out of bounds"
            assert 0 <= col < self.cost_table.shape[1], f"Col index {col} out of bounds"

            logger.info(
                f"Segment {i}: k range [{k_start:.4f}, {k_end:.4f}], "
                f"optimal assignment: ({row}, {col})"
            )

        # Test that segments cover the entire [0,1] range
        if segments:
            min_k = min(segment[0] for segment in segments)
            max_k = max(segment[1] for segment in segments)
            logger.info(f"K value coverage: [{min_k:.4f}, {max_k:.4f}]")
            assert min_k <= 0.01, f"Segments don't start near 0, min_k={min_k}"
            assert max_k >= 0.99, f"Segments don't reach near 1, max_k={max_k}"


if __name__ == "__main__":
    pytest.main(["-xvs", "--log-cli-level=INFO", __file__])
