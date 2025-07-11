import pytest
import numpy as np
from src.propflow.bp_base.factor_graph import FactorGraph
from src.propflow.bp_base.engines_realizations import BPEngine
from src.propflow.utils import FGBuilder
from src.propflow.configs import create_random_int_table
from src.propflow.policies import ConvergenceConfig


# Flag to enable verbose output during tests
VERBOSE = False


def verbose_print(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)


class TestBPEngine:
    @pytest.fixture
    def convergence_config(self):
        """Create a standard convergence configuration for tests."""
        return ConvergenceConfig(
            max_iterations=50,
            convergence_threshold=1e-6,
            time_limit=10,
            check_interval=5,
        )

    @pytest.fixture(params=[2, 3, 4])
    def domain_size(self, request):
        """Parameterize tests with different domain sizes."""
        return request.param

    @pytest.fixture(
        params=[
            (FGBuilder.build_cycle_graph, {"num_vars": 5, "density": 1.0}),
            (
                FGBuilder.build_random_graph,
                {"num_vars": 4, "num_factors": 4, "density": 0.8},
            ),
        ]
    )
    def factor_graph(self, request, domain_size):
        """Create different types of factor graphs for testing."""
        builder_func, params = request.param

        # Add domain size to params
        params["domain_size"] = domain_size

        # Add cost table factory
        params["ct_factory"] = create_random_int_table
        params["ct_params"] = {"low": 1, "high": 10}

        # Build the graph
        variables, factors, edges = builder_func(**params)
        fg = FactorGraph(variables, factors, edges)

        verbose_print(
            f"Created {builder_func.__name__} with {len(variables)} variables and {len(factors)} factors"
        )
        return fg

    @pytest.fixture(
        params=[
            create_random_int_table,
        ]
    )
    def factor_graph_with_different_tables(self, request, domain_size):
        """Create factor graphs with different types of cost tables."""
        ct_factory = request.param
        ct_params = (
            {"low": 1, "high": 10} if ct_factory in [create_random_int_table] else {}
        )

        fg = FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=domain_size,
            ct_factory=ct_factory,
            ct_params=ct_params,
            density=1.0,
        )

        verbose_print(f"Created factor graph with {ct_factory.__name__} cost tables")
        return fg

    def convergence_is_valid(self, engine):
        """Check if engine convergence results are valid."""
        # Should have some iterations
        assert (
            engine.iteration_count > 0
        ), "Engine should perform at least one iteration"

        # Should have values for all variables
        for var in engine.factor_graph.variables:
            assert (
                var.name in engine.get_map_assignment()
            ), f"Missing MAP assignment for {var.name}"

        # Beliefs should sum to approximately 1 for each variable
        for var in engine.factor_graph.variables:
            belief = engine.get_belief(var.name)
            assert np.isclose(
                np.sum(belief), 1.0, atol=1e-5
            ), f"Beliefs for {var.name} don't sum to 1"

        return True

    def engine_runs_correctly(self, bp_engine):
        """Run engine and check basic functionality."""
        # Run the engine
        bp_engine.run()

        # Get MAP assignment
        map_assignment = bp_engine.get_map_assignment()

        # Check that we have assignments for all variables
        variable_names = [v.name for v in bp_engine.factor_graph.variables]
        for var_name in variable_names:
            assert (
                var_name in map_assignment
            ), f"Variable {var_name} not in MAP assignment"

        # Check that assignments are within domain
        for var_name, value in map_assignment.items():
            var = next(
                v for v in bp_engine.factor_graph.variables if v.name == var_name
            )
            assert (
                0 <= value < var.domain
            ), f"Assignment {value} for {var_name} outside domain {var.domain}"

        # Check that beliefs were computed
        for var_name in variable_names:
            belief = bp_engine.get_belief(var_name)
            assert belief is not None, f"No belief for {var_name}"
            assert len(belief) == next(
                v.domain for v in bp_engine.factor_graph.variables if v.name == var_name
            ), f"Belief length doesn't match domain size for {var_name}"

        return True

    def basic_engine_operations(self, engine):
        """Check standard BP engine operations."""
        # Check initial state
        assert engine.iteration_count == 0, "Initial iteration count should be 0"
        assert not engine.converged, "Engine should not be converged initially"

        # Run a single iteration
        engine.run_iteration()
        assert (
            engine.iteration_count == 1
        ), "Iteration count should be 1 after run_iteration"

        # Check for message passing
        for var in engine.factor_graph.variables:
            assert (
                len(var.mailer.history) > 0
            ), f"Variable {var.name} should have message history"

        for factor in engine.factor_graph.factors:
            assert (
                len(factor.mailer.history) > 0
            ), f"Factor {factor.name} should have message history"

        # Reset the engine
        engine.reset()
        assert engine.iteration_count == 0, "Iteration count should be 0 after reset"

        return True

    def bp_engine_produces_reasonable_results(self, engine):
        """Check that BP engine produces reasonable results."""
        # Run the engine
        engine.run()

        # Get energy before and after
        initial_energy = engine.get_initial_energy()
        final_energy = engine.get_energy()

        # Check that energy decreased or stayed the same
        assert (
            final_energy <= initial_energy
        ), f"Final energy {final_energy} should be <= initial energy {initial_energy}"

        return True

    def test_bp_engine_initialization(self, factor_graph, convergence_config):
        """BP engine initializes correctly with different factor graphs."""
        engine = BPEngine(factor_graph, convergence_config=convergence_config)

        # Check engine has correct factor graph
        assert (
            engine.factor_graph == factor_graph
        ), "Engine should have the provided factor graph"

        # Check convergence config was set
        assert (
            engine.convergence_config == convergence_config
        ), "Engine should have the provided convergence config"

        # Check initial state
        assert engine.iteration_count == 0, "Initial iteration count should be 0"
        assert not engine.converged, "Engine should not be converged initially"

    def test_bp_engine_basic_operations(self, factor_graph, convergence_config):
        """BP engine correctly performs basic operations."""
        engine = BPEngine(factor_graph, convergence_config=convergence_config)
        assert self.basic_engine_operations(engine), "Basic engine operations failed"

    def test_bp_engine_run_on_different_graphs(self, factor_graph, convergence_config):
        """BP engine runs correctly on different types of factor graphs."""
        engine = BPEngine(factor_graph, convergence_config=convergence_config)
        assert self.engine_runs_correctly(engine), "Engine run failed"
        assert self.convergence_is_valid(engine), "Convergence check failed"

    def test_bp_engine_with_different_cost_tables(
        self, factor_graph_with_different_tables, convergence_config
    ):
        """BP engine handles different types of cost tables correctly."""
        engine = BPEngine(
            factor_graph_with_different_tables, convergence_config=convergence_config
        )
        assert self.engine_runs_correctly(
            engine
        ), "Engine run failed with custom cost tables"

    def test_bp_engine_energy_minimization(self, factor_graph, convergence_config):
        """BP engine correctly minimizes energy."""
        engine = BPEngine(factor_graph, convergence_config=convergence_config)
        assert self.bp_engine_produces_reasonable_results(
            engine
        ), "Energy minimization failed"

    def test_bp_engine_early_convergence(self, factor_graph):
        """BP engine detects early convergence correctly."""
        # Create a configuration that should converge quickly
        quick_config = ConvergenceConfig(
            max_iterations=100,
            convergence_threshold=0.1,  # Very loose threshold
            time_limit=10,
            check_interval=1,
        )

        engine = BPEngine(factor_graph, convergence_config=quick_config)
        engine.run()

        # Should converge before max iterations
        assert engine.converged, "Engine should have converged"
        assert (
            engine.iteration_count < quick_config.max_iterations
        ), "Engine should converge early"

    def test_bp_engine_max_iterations(self, factor_graph):
        """BP engine respects max iterations limit."""
        # Create a configuration with very strict convergence threshold
        strict_config = ConvergenceConfig(
            max_iterations=10,
            convergence_threshold=1e-12,  # Very tight threshold
            time_limit=10,
            check_interval=1,
        )

        engine = BPEngine(factor_graph, convergence_config=strict_config)
        engine.run()

        # Should stop at max iterations
        assert (
            engine.iteration_count == strict_config.max_iterations
        ), "Engine should stop at max iterations"

    def test_bp_engine_beliefs_after_convergence(
        self, factor_graph, convergence_config
    ):
        """BP engine produces valid beliefs after convergence."""
        engine = BPEngine(factor_graph, convergence_config=convergence_config)
        engine.run()

        # Check beliefs for all variables
        for var in engine.factor_graph.variables:
            belief = engine.get_belief(var.name)

            # Belief should be a valid probability distribution
            assert np.all(belief >= 0), f"Negative values in belief for {var.name}"
            assert np.isclose(
                np.sum(belief), 1.0, atol=1e-5
            ), f"Belief for {var.name} doesn't sum to 1"

    def test_bp_engine_map_consistency(self, factor_graph, convergence_config):
        """BP engine MAP assignments are consistent with beliefs."""
        engine = BPEngine(factor_graph, convergence_config=convergence_config)
        engine.run()

        map_assignment = engine.get_map_assignment()

        # Check consistency with beliefs
        for var_name, value in map_assignment.items():
            belief = engine.get_belief(var_name)
            assert value == np.argmax(
                belief
            ), f"MAP assignment {value} inconsistent with belief argmax {np.argmax(belief)}"
