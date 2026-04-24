from experiments.non_convergence_chain.config import load_config
from experiments.non_convergence_chain.run_non_convergence_study import _run_standard
from experiments.non_convergence_chain.config import build_chain_graph


def test_chain_config_builds_symmetric_parallel_factor_copies(tmp_path):
    config_path = tmp_path / "chain.yaml"
    config_path.write_text(
        """
graph_name: test_chain
run_chain: true
variable_order: ["X1", "X2", "X3"]
domain_values: [0, 1]
cost_tables:
  - name: F12
    variables: ["X1", "X2"]
    table: [[50, 60], [20, 300]]
  - name: F23
    variables: ["X2", "X3"]
    table: [[45, 100], [300, 6]]
max_iter: 8
tolerance: 1.0e-9
output_dir: unused
seed: 17
"""
    )
    cfg = load_config(config_path)

    graph = build_chain_graph(cfg)

    assert sorted(factor.name for factor in graph.factors) == [
        "F12_1",
        "F12_2",
        "F23_1",
        "F23_2",
    ]
    scopes = {
        factor.name: [variable.name for variable in graph.edges[factor]]
        for factor in graph.factors
    }
    assert scopes == {
        "F12_1": ["X1", "X2"],
        "F12_2": ["X1", "X2"],
        "F23_1": ["X2", "X3"],
        "F23_2": ["X2", "X3"],
    }
    tables = {factor.name: factor.cost_table.tolist() for factor in graph.factors}
    assert tables["F12_1"] == [[50.0, 60.0], [20.0, 300.0]]
    assert tables["F12_2"] == tables["F12_1"]
    assert tables["F23_1"] == [[45.0, 100.0], [300.0, 6.0]]
    assert tables["F23_2"] == tables["F23_1"]


def test_same_seed_and_config_give_identical_summary_metrics(tmp_path):
    config_path = tmp_path / "chain.yaml"
    config_path.write_text(
        """
graph_name: test_chain
variable_order: ["X1", "X2", "X3"]
domain_values: [0, 1]
cost_tables:
  - name: F12
    variables: ["X1", "X2"]
    table: [[0, 4], [5, 1]]
  - name: F23
    variables: ["X2", "X3"]
    table: [[0, 2], [3, 0]]
max_iter: 8
tolerance: 1.0e-9
split_at_iters: [2]
damping_factors: [0.5]
output_dir: unused
seed: 17
"""
    )
    cfg_a = load_config(config_path)
    cfg_b = load_config(config_path)

    run_a = _run_standard(
        lambda: build_chain_graph(cfg_a),
        max_iter=cfg_a.max_iter,
        trace_every=1,
        tolerance=cfg_a.tolerance,
    )
    run_b = _run_standard(
        lambda: build_chain_graph(cfg_b),
        max_iter=cfg_b.max_iter,
        trace_every=1,
        tolerance=cfg_b.tolerance,
    )

    assert run_a["summary"] == run_b["summary"]
    assert [row["assignments"] for row in run_a["trace"]] == [
        row["assignments"] for row in run_b["trace"]
    ]
