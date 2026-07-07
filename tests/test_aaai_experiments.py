from itertools import combinations

import networkx as nx
import numpy as np
import pytest

from propflow import FGBuilder
from propflow.bp.engine_base import BPEngine
from propflow.configs import create_random_int_table

from experiments.aaai.code import (
    plot_results,
    problems,
    problems_ternary,
    run_experiments,
)
from experiments.aaai.code.csv_backups import backup_existing_csvs
from experiments.aaai.code.merge import invert_binary_menu_assignment


def test_aaai_plain_min_sum_is_in_all_algorithm_sets():
    assert run_experiments.PLAIN_MS_LABEL == "MS"
    assert run_experiments.ENGINE_LABELS[0] == "MS"
    assert "MS" in run_experiments.ALL_LABELS
    assert "MS" in run_experiments.KNOWN_LABELS
    assert run_experiments.MGM_INVERTED_LABEL in run_experiments.ALL_LABELS
    assert run_experiments.MGM_INVERTED_LABEL in run_experiments.KNOWN_LABELS


def test_aaai_plain_min_sum_factory_builds_base_engine():
    graph = FGBuilder.build_cycle_graph(
        num_vars=3,
        domain_size=2,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 3},
    )

    engine = run_experiments.make_engine("MS", graph, seed=0)

    assert type(engine) is BPEngine
    costs = run_experiments.run_full_horizon(engine, max_iter=2)
    assert len(costs) == 2


def test_aaai_plain_min_sum_is_plotted():
    assert plot_results.LABELS["MS"] == "MS"
    assert plot_results.ORDER[0] == "MS"
    assert run_experiments.MGM_INVERTED_LABEL in plot_results.ORDER


def test_aaai_plotted_algorithms_have_fixed_colors():
    assert set(plot_results.ORDER) <= set(plot_results.COLORS)


def test_aaai_inverted_mgm_flips_disagreement_menu_values():
    branch1 = {"x1": 0, "x2": 3, "x3": 4}
    branch2 = {"x1": 1, "x2": 3, "x3": 9}
    assignment = {"x1": 1, "x2": 3, "x3": 4}

    inverted = invert_binary_menu_assignment(
        assignment, branch1, branch2, ["x1", "x2", "x3"]
    )

    assert inverted == {"x1": 0, "x2": 3, "x3": 9}


def test_aaai_inverted_mgm_schedules_split_merge_task():
    class Args:
        seed_start = 0
        n_problems = 1
        max_iter = 200
        merge_at = 100
        opt_time_limit = 1.0

    tasks = run_experiments.build_tasks(
        "graph_coloring", Args, {run_experiments.MGM_INVERTED_LABEL}
    )

    assert tasks == [
        (
            "split_ms",
            "graph_coloring",
            0,
            {
                "max_iter": 200,
                "merge_at": 100,
                "wanted": {run_experiments.MGM_INVERTED_LABEL},
            },
        )
    ]


def test_aaai_zoom_selection_keeps_crowded_lower_cluster():
    curves = [
        plot_results.CostCurve(
            algorithm=f"a{i}",
            xs=np.arange(100),
            ys=np.full(100, value, dtype=float),
            label=f"a{i}",
        )
        for i, value in enumerate([10, 11, 12, 13, 14, 60, 100])
    ]

    selected = plot_results.select_zoom_curves(curves, horizon=100)

    assert [curve.algorithm for curve in selected] == ["a0", "a1", "a2", "a3", "a4"]


def test_aaai_csv_backup_copies_current_csvs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "current.csv").write_text("a,b\n1,2\n")
    (data_dir / "notes.txt").write_text("not copied\n")

    backup_dir = backup_existing_csvs(
        data_dir,
        label="test_backup",
        backup_root=tmp_path / "backups",
    )

    assert backup_dir is not None
    assert (backup_dir / "current.csv").read_text() == "a,b\n1,2\n"
    assert not (backup_dir / "notes.txt").exists()


def test_aaai_random_ternary_builds_true_arity_three_graph():
    graph = problems.build_random_ternary(seed=0)

    assert len(graph.variables) == problems.NUM_AGENTS
    assert {v.domain for v in graph.variables} == {10}

    unary_factors = []
    ternary_factors = []
    primal = nx.Graph()
    primal.add_nodes_from(v.name for v in graph.variables)
    for factor, variables in graph.edges.items():
        arity = len(variables)
        if arity == 1:
            unary_factors.append(factor)
            assert factor.cost_table.shape == (10,)
        else:
            assert arity == 3
            ternary_factors.append(factor)
            assert factor.cost_table.shape == (10, 10, 10)
            primal.add_edges_from(
                combinations((variable.name for variable in variables), 2)
            )

    assert len(unary_factors) == problems.NUM_AGENTS
    assert ternary_factors
    assert nx.is_connected(primal)


def test_aaai_random_ternary_all_resolves_to_dms_split_only():
    labels, skipped = run_experiments.labels_for_benchmark(
        run_experiments.RANDOM_TERNARY_BENCHMARK,
        set(run_experiments.ALL_LABELS),
        all_requested=True,
    )

    assert labels == {"DMS_split_0.5"}
    assert skipped == set()


def test_aaai_random_ternary_skips_unsupported_explicit_labels():
    labels, skipped = run_experiments.labels_for_benchmark(
        run_experiments.RANDOM_TERNARY_BENCHMARK,
        {"MS", "DMS_split_0.5", "Attentive"},
        all_requested=False,
    )

    assert labels == {"DMS_split_0.5"}
    assert skipped == {"MS", "Attentive"}


def test_aaai_random_ternary_dms_split_smoke_run():
    graph = problems.build_random_ternary(seed=0)
    engine = run_experiments.make_engine("DMS_split_0.5", graph, seed=0)

    costs = run_experiments.run_full_horizon(engine, max_iter=2)

    assert len(costs) == 2
    assert np.isfinite(costs).all()


# --- ternary suite (problems_ternary) ---------------------------------------

TERNARY_SHAPES = {
    "random_sparse_ternary": (10, problems.NUM_AGENTS),
    "random_dense_ternary": (10, problems.NUM_AGENTS),
    "graph_coloring_ternary": (3, problems.NUM_AGENTS),
    "scale_free_ternary": (10, problems.NUM_AGENTS),
    "meeting_scheduling_ternary": (problems.MS_TIME_SLOTS, problems.MS_MEETINGS),
}


def _assert_true_ternary(graph, domain, num_vars):
    assert len(graph.variables) == num_vars
    assert {v.domain for v in graph.variables} == {domain}

    primal = nx.Graph()
    primal.add_nodes_from(v.name for v in graph.variables)
    ternary_factors = []
    for factor, variables in graph.edges.items():
        arity = len(variables)
        if arity == 1:  # tie-break unary preference
            assert factor.cost_table.shape == (domain,)
            continue
        assert arity == 3
        assert factor.cost_table.shape == (domain, domain, domain)
        ternary_factors.append(factor)
        primal.add_edges_from(combinations((v.name for v in variables), 2))

    assert ternary_factors
    assert nx.is_connected(primal)


@pytest.mark.parametrize("benchmark", sorted(TERNARY_SHAPES))
def test_aaai_ternary_builders_make_connected_arity_three_graphs(benchmark):
    domain, num_vars = TERNARY_SHAPES[benchmark]
    graph = problems_ternary.TERNARY_BENCHMARKS[benchmark](seed=0)
    _assert_true_ternary(graph, domain, num_vars)


def test_aaai_ternary_builders_are_deterministic_per_seed():
    a = problems_ternary.build_scale_free_ternary(seed=3)
    b = problems_ternary.build_scale_free_ternary(seed=3)
    assert sorted(f.name for f in a.factors) == sorted(f.name for f in b.factors)


def test_aaai_ternary_coloring_table_is_equal_pair_penalty():
    table = problems_ternary.create_ternary_coloring_table(domain=3, cost=10.0)
    assert table.shape == (3, 3, 3)
    assert table[0, 0, 0] == 30.0  # all three equal -> 3 equal pairs
    assert table[0, 0, 1] == 10.0  # one equal pair
    assert table[0, 1, 2] == 0.0  # all distinct


def test_aaai_ternary_suite_is_registered_but_excluded_from_binary_all():
    for name in TERNARY_SHAPES:
        assert name in run_experiments.BENCHMARK_BUILDERS
        assert name in run_experiments.TERNARY_SUITE
        # the binary "all" expansion must NOT pull in the ternary suite
        assert name not in problems.BENCHMARKS


def test_aaai_ternary_all_resolves_to_full_family_minus_dabp():
    labels, skipped = run_experiments.labels_for_benchmark(
        "random_dense_ternary",
        set(run_experiments.ALL_LABELS),
        all_requested=True,
    )

    assert labels == run_experiments.TERNARY_SUPPORTED_LABELS
    assert "MS" in labels and "DMS_split_0.5" in labels and "Optimal" in labels
    assert run_experiments.DABP_LABELS.isdisjoint(labels)
    assert skipped == set()


def test_aaai_ternary_skips_dabp_when_named_explicitly():
    labels, skipped = run_experiments.labels_for_benchmark(
        "scale_free_ternary",
        {"MS", "DMS", "Attentive", "Attentive_NoSplit"},
        all_requested=False,
    )

    assert labels == {"MS", "DMS"}
    assert skipped == {"Attentive", "Attentive_NoSplit"}


def test_aaai_ternary_allows_explicit_opt_in_extra_split_point():
    # DMS_split_at_1500 is an opt-in extra (not part of "all"); naming it
    # explicitly for a ternary benchmark must be honored, like the binary suite.
    labels, skipped = run_experiments.labels_for_benchmark(
        "random_dense_ternary",
        {"MS", "DMS_split_at_1500"},
        all_requested=False,
    )

    assert labels == {"MS", "DMS_split_at_1500"}
    assert skipped == set()


def test_aaai_ternary_all_excludes_opt_in_extras():
    labels, _ = run_experiments.labels_for_benchmark(
        "random_dense_ternary",
        set(run_experiments.ALL_LABELS),
        all_requested=True,
    )

    assert "DMS_split_at_1500" not in labels


@pytest.mark.parametrize("label", ["MS", "DMS", "DMS_split_0.5", "MS_split_0.5"])
def test_aaai_ternary_engines_smoke_run(label):
    graph = problems_ternary.build_graph_coloring_ternary(seed=0)
    engine = run_experiments.make_engine(label, graph, seed=0)

    costs = run_experiments.run_full_horizon(engine, max_iter=2)

    assert len(costs) == 2
    assert np.isfinite(costs).all()
