import pytest
import numpy as np
import os
import tempfile
import pickle
from src.propflow.utils import FGBuilder
from src.propflow.utils.fg_utils import (
    get_message_shape, 
    get_broadcast_shape, 
    generate_random_cost,
    get_bound,
    SafeUnpickler,
    load_pickle_safely,
    repair_factor_graph
)
from src.propflow.utils.general_utils import (
    get_project_root,
    create_logger,
    save_json,
    load_json
)
from src.propflow.utils.inbox_utils import (
    clear_mailboxes,
    get_mailbox_status,
    count_messages
)
from src.propflow.configs import create_random_int_table


class TestUtilsFunctions:
    """Test suite for utility functions."""

    @pytest.fixture
    def sample_factor_graph(self):
        """Create a sample factor graph for testing."""
        return FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10}
        )

    def test_get_message_shape(self):
        """Test message shape calculation."""
        # Test default binary connections
        shape = get_message_shape(domain_size=3)
        assert shape == (3, 3)
        
        # Test custom connections
        shape = get_message_shape(domain_size=2, connections=3)
        assert shape == (2, 2, 2)
        
        # Test single connection
        shape = get_message_shape(domain_size=4, connections=1)
        assert shape == (4,)

    def test_get_broadcast_shape(self):
        """Test broadcast shape calculation."""
        ct_dims = (3, 3)
        domain_size = 3
        
        # Test axis 0
        shape = get_broadcast_shape(ct_dims, domain_size, 0)
        assert shape == (3, 1)
        
        # Test axis 1
        shape = get_broadcast_shape(ct_dims, domain_size, 1)
        assert shape == (1, 3)

    def test_generate_random_cost(self, sample_factor_graph):
        """Test random cost generation."""
        cost = generate_random_cost(sample_factor_graph)
        
        assert isinstance(cost, (int, float))
        assert cost >= 0  # Cost should be non-negative
        
        # Test multiple generations give different results (probabilistic)
        costs = [generate_random_cost(sample_factor_graph) for _ in range(10)]
        assert len(set(costs)) > 1  # Should have some variation

    def test_get_bound(self, sample_factor_graph):
        """Test bound calculation."""
        # Test minimum bound
        min_bound = get_bound(sample_factor_graph, reduce_func=np.min)
        assert isinstance(min_bound, float)
        assert min_bound >= 0
        
        # Test maximum bound
        max_bound = get_bound(sample_factor_graph, reduce_func=np.max)
        assert isinstance(max_bound, float)
        assert max_bound >= min_bound
        
        # Test sum bound
        sum_bound = get_bound(sample_factor_graph, reduce_func=np.sum)
        assert isinstance(sum_bound, float)
        assert sum_bound >= max_bound

    def test_safe_unpickler(self, sample_factor_graph):
        """Test safe pickle loading."""
        # Create a temporary pickle file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            pickle.dump(sample_factor_graph, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path = tmp.name
        
        try:
            # Test safe loading
            loaded_fg = load_pickle_safely(tmp_path)
            assert loaded_fg is not None
            assert len(loaded_fg.variables) == len(sample_factor_graph.variables)
            assert len(loaded_fg.factors) == len(sample_factor_graph.factors)
        finally:
            os.unlink(tmp_path)

    def test_repair_factor_graph(self, sample_factor_graph):
        """Test factor graph repair functionality."""
        # Create a "broken" factor graph by removing NetworkX graph
        broken_fg = sample_factor_graph
        broken_fg.G = None
        
        # Repair it
        repaired_fg = repair_factor_graph(broken_fg)
        
        assert repaired_fg.G is not None
        assert len(repaired_fg.G.nodes()) > 0


class TestGeneralUtils:
    """Test suite for general utility functions."""

    def test_get_project_root(self):
        """Test project root detection."""
        from src.propflow.utils.general_utils import get_project_root
        
        root = get_project_root()
        assert root is not None
        assert os.path.exists(root)
        assert os.path.isdir(root)

    def test_create_logger(self):
        """Test logger creation."""
        from src.propflow.utils.general_utils import create_logger
        
        logger = create_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_save_and_load_json(self):
        """Test JSON save and load functionality."""
        from src.propflow.utils.general_utils import save_json, load_json
        
        test_data = {
            "test_key": "test_value",
            "numbers": [1, 2, 3],
            "nested": {"inner": "value"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test save
            save_json(test_data, tmp_path)
            assert os.path.exists(tmp_path)
            
            # Test load
            loaded_data = load_json(tmp_path)
            assert loaded_data == test_data
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestInboxUtils:
    """Test suite for inbox utility functions."""

    @pytest.fixture
    def sample_factor_graph_with_messages(self):
        """Create a factor graph and add some messages."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5}
        )
        
        # Add some dummy messages to mailboxes
        for agent in fg.variables + fg.factors:
            if hasattr(agent, 'mailbox'):
                agent.mailbox = [f"message_{i}" for i in range(np.random.randint(0, 5))]
        
        return fg

    def test_clear_mailboxes(self, sample_factor_graph_with_messages):
        """Test mailbox clearing functionality."""
        fg = sample_factor_graph_with_messages
        
        # Verify mailboxes have messages
        total_messages_before = sum(
            len(getattr(agent, 'mailbox', []))
            for agent in fg.variables + fg.factors
        )
        
        # Clear mailboxes
        clear_mailboxes(fg)
        
        # Verify mailboxes are empty
        total_messages_after = sum(
            len(getattr(agent, 'mailbox', []))
            for agent in fg.variables + fg.factors
        )
        
        assert total_messages_after == 0
        assert total_messages_before >= 0

    def test_get_mailbox_status(self, sample_factor_graph_with_messages):
        """Test mailbox status reporting."""
        fg = sample_factor_graph_with_messages
        
        status = get_mailbox_status(fg)
        
        assert isinstance(status, dict)
        assert 'total_messages' in status
        assert 'agents_with_messages' in status
        assert 'empty_mailboxes' in status
        
        assert status['total_messages'] >= 0
        assert status['agents_with_messages'] >= 0
        assert status['empty_mailboxes'] >= 0

    def test_count_messages(self, sample_factor_graph_with_messages):
        """Test message counting functionality."""
        fg = sample_factor_graph_with_messages
        
        # Count messages by type
        var_messages = count_messages(fg, agent_type='variable')
        factor_messages = count_messages(fg, agent_type='factor')
        total_messages = count_messages(fg)
        
        assert isinstance(var_messages, int)
        assert isinstance(factor_messages, int)
        assert isinstance(total_messages, int)
        
        assert var_messages >= 0
        assert factor_messages >= 0
        assert total_messages == var_messages + factor_messages


class TestPerformanceUtils:
    """Test suite for performance-related utilities."""

    def test_performance_monitoring(self):
        """Test performance monitoring tools."""
        from src.propflow.utils.tools.performance import (
            time_function,
            memory_usage,
            profile_function
        )
        
        # Test timing
        @time_function
        def dummy_function():
            return sum(range(1000))
        
        result = dummy_function()
        assert result == sum(range(1000))
        
        # Test memory usage
        usage = memory_usage()
        assert isinstance(usage, dict)
        assert 'rss' in usage
        assert 'vms' in usage

    def test_convex_hull_utils(self):
        """Test convex hull utility functions."""
        from src.propflow.utils.tools.convex_hull import (
            compute_convex_hull,
            point_in_hull
        )
        
        # Test with simple 2D points
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        hull = compute_convex_hull(points)
        
        assert hull is not None
        assert len(hull.vertices) <= len(points)
        
        # Test point in hull
        assert point_in_hull(np.array([0.5, 0.5]), hull)
        assert not point_in_hull(np.array([2, 2]), hull)


class TestCostTableCreation:
    """Test suite for cost table creation utilities."""

    def test_create_random_int_table(self):
        """Test random integer cost table creation."""
        from src.propflow.utils.create.create_cost_tables import create_random_int_table
        
        params = {"low": 1, "high": 10}
        table = create_random_int_table(num_vars=2, domain_size=3, **params)
        
        assert isinstance(table, np.ndarray)
        assert table.shape == (3, 3)
        assert table.dtype in [np.int32, np.int64]
        assert np.all(table >= 1)
        assert np.all(table <= 10)

    def test_create_attractive_table(self):
        """Test attractive cost table creation."""
        from src.propflow.utils.create.create_cost_tables import create_attractive_table
        
        params = {"strength": 2.0}
        table = create_attractive_table(num_vars=2, domain_size=2, **params)
        
        assert isinstance(table, np.ndarray)
        assert table.shape == (2, 2)
        # Diagonal should be lower cost
        assert table[0, 0] < table[0, 1]
        assert table[1, 1] < table[1, 0]

    def test_create_random_float_table(self):
        """Test random float cost table creation."""
        from src.propflow.utils.create.create_cost_tables import create_random_float_table
        
        params = {"low": 0.1, "high": 5.0}
        table = create_random_float_table(num_vars=2, domain_size=3, **params)
        
        assert isinstance(table, np.ndarray)
        assert table.shape == (3, 3)
        assert table.dtype in [np.float32, np.float64]
        assert np.all(table >= 0.1)
        assert np.all(table <= 5.0)