from itertools import combinations

import networkx as nx
import numpy as np

from propflow import FGBuilder
from propflow.bp.engine_base import BPEngine
from propflow.configs import create_random_int_table

from experiments.aaai.code import plot_results, problems, run_experiments
from experiments.aaai.code.csv_backups import backup_existing_csvs


def test_aaai_plain_min_sum_is_in_all_algorithm_sets():
    assert run_experiments.PLAIN_MS_LABEL == "MS"
    assert run_experiments.ENGINE_LABELS[0] == "MS"
    assert "MS" in run_experiments.ALL_LABELS
    assert "MS" in run_experiments.KNOWN_LABELS


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


def test_aaai_plotted_algorithms_have_fixed_colors():
    assert set(plot_results.ORDER) <= set(plot_results.COLORS)


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
