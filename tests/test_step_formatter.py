"""Tests for the StepByStepFormatter class and unary factor support."""

import numpy as np
import pytest

from propflow import FGBuilder, BPEngine, MinSumComputator
from propflow.configs import create_random_int_table
from propflow.snapshots import StepByStepFormatter
from propflow.bp.factor_graph import FactorGraph


class TestStepByStepFormatter:
    """Test cases for the StepByStepFormatter class."""

    @pytest.fixture
    def simple_engine(self):
        """Create a simple engine with a few iterations."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 0, "high": 10},
        )
        engine = BPEngine(fg, computator=MinSumComputator())
        engine.run(max_iter=3)
        return engine

    def test_formatter_initialization(self, simple_engine):
        """StepByStepFormatter initializes correctly with snapshots."""
        formatter = StepByStepFormatter(simple_engine.snapshots)

        assert len(formatter.steps) == 3
        assert len(formatter.variables) == 3
        assert len(formatter.factors) == 3
        assert formatter.domain_size == 2

    def test_formatter_rejects_empty_snapshots(self):
        """StepByStepFormatter raises ValueError for empty snapshots."""
        with pytest.raises(ValueError, match="empty"):
            StepByStepFormatter([])

    def test_format_cost_tables(self, simple_engine):
        """format_cost_tables returns properly formatted output."""
        formatter = StepByStepFormatter(simple_engine.snapshots)
        output = formatter.format_cost_tables()

        assert "COST TABLES" in output
        assert "F12" in output.upper() or "f12" in output
        # should contain factor names
        for factor in formatter.factors:
            assert factor.upper() in output.upper()

    def test_format_iteration(self, simple_engine):
        """format_iteration returns Q/R messages and assignments."""
        formatter = StepByStepFormatter(simple_engine.snapshots)
        output = formatter.format_iteration(0)

        assert "ITERATION 0" in output
        assert "Q Messages" in output
        assert "R Messages" in output
        assert "Assignments" in output
        assert "Solution Cost" in output

    def test_format_all_steps(self, simple_engine):
        """format_all_steps returns complete output."""
        formatter = StepByStepFormatter(simple_engine.snapshots)
        output = formatter.format_all_steps()

        assert "STEP-BY-STEP OUTPUT" in output
        assert "ITERATION 0" in output
        assert "ITERATION 1" in output
        assert "ITERATION 2" in output
        assert "COST TABLES" in output

    def test_format_summary(self, simple_engine):
        """format_summary returns summary with initial/final costs."""
        formatter = StepByStepFormatter(simple_engine.snapshots)
        output = formatter.format_summary()

        assert "SIMULATION SUMMARY" in output
        assert "Total iterations: 3" in output
        assert "Initial cost:" in output
        assert "Final cost:" in output
        assert "Final assignments:" in output


class TestUnaryFactorSupport:
    """Test cases for unary factor support in FGBuilder."""

    @pytest.fixture
    def base_graph(self):
        """Create a base factor graph."""
        return FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 0, "high": 10},
        )

    def test_build_with_unary_costs(self, base_graph):
        """build_with_unary_costs adds unary factors correctly."""
        unary_costs = {
            "x1": np.array([0, 5]),
            "x2": np.array([3, 0]),
        }

        fg = FGBuilder.build_with_unary_costs(base_graph, unary_costs)

        # should have 5 factors now (3 binary + 2 unary)
        assert len(fg.factors) == 5

        # check unary factor names
        factor_names = [f.name for f in fg.factors]
        assert "u1" in factor_names
        assert "u2" in factor_names

    def test_unary_factor_cost_table(self, base_graph):
        """Unary factors have correct 1D cost tables."""
        unary_costs = {"x1": np.array([0.0, 5.0])}

        fg = FGBuilder.build_with_unary_costs(base_graph, unary_costs)

        # find the unary factor
        u1 = next(f for f in fg.factors if f.name == "u1")
        np.testing.assert_array_equal(u1.cost_table, np.array([0.0, 5.0]))

    def test_unary_invalid_variable_raises(self, base_graph):
        """build_with_unary_costs raises for unknown variable."""
        with pytest.raises(ValueError, match="not found"):
            FGBuilder.build_with_unary_costs(
                base_graph, {"x99": np.array([0, 1])}
            )

    def test_unary_wrong_dimension_raises(self, base_graph):
        """build_with_unary_costs raises for 2D array."""
        with pytest.raises(ValueError, match="1D"):
            FGBuilder.build_with_unary_costs(
                base_graph, {"x1": np.array([[0, 1], [2, 3]])}
            )

    def test_unary_wrong_size_raises(self, base_graph):
        """build_with_unary_costs raises for wrong array size."""
        with pytest.raises(ValueError, match="length"):
            FGBuilder.build_with_unary_costs(
                base_graph, {"x1": np.array([0, 1, 2, 3])}  # domain is 2, not 4
            )

    def test_bp_runs_with_unary_factors(self, base_graph):
        """BP engine runs correctly with unary factors."""
        unary_costs = {
            "x1": np.array([0, 10]),  # strongly prefer a
            "x3": np.array([10, 0]),  # strongly prefer b
        }

        fg = FGBuilder.build_with_unary_costs(base_graph, unary_costs)
        engine = BPEngine(fg, computator=MinSumComputator())
        engine.run(max_iter=5)

        # should have assignments for all variables
        assert len(engine.assignments) == 3
        assert all(0 <= v < 2 for v in engine.assignments.values())

    def test_unary_formatter_shows_unary_factors(self, base_graph):
        """StepByStepFormatter displays unary factor cost tables."""
        unary_costs = {"x1": np.array([0, 5])}
        fg = FGBuilder.build_with_unary_costs(base_graph, unary_costs)

        engine = BPEngine(fg, computator=MinSumComputator())
        engine.run(max_iter=2)

        formatter = StepByStepFormatter(engine.snapshots)
        output = formatter.format_cost_tables()

        assert "U1" in output.upper()
        # unary factors should show 1D format
        assert "a:" in output.lower() or "0:" in output
