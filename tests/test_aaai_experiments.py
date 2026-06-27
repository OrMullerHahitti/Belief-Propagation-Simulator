from propflow import FGBuilder
from propflow.bp.engine_base import BPEngine
from propflow.configs import create_random_int_table

from experiments.aaai.code import plot_results, run_experiments


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
