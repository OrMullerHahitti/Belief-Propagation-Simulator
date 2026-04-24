import json

from experiments.non_convergence_chain.run_non_convergence_study import main


def test_random_graph_mode_does_not_require_chain_tables(tmp_path):
    config_path = tmp_path / "random_only.yaml"
    out_dir = tmp_path / "out"
    config_path.write_text(
        """
graph_name: random_only
run_chain: false
variable_order: ["X1", "X2", "X3"]
domain_values: [0, 1]
cost_tables: []
max_iter: 3
tolerance: 1.0e-9
split_at_iters: [1]
split_ratio: 0.5
damping_factors: []
trace_every: 1
seed: 11
random_graph:
  enabled: true
  num_vars: 4
  domain_size: 2
  density: 1.0
  ct_factory: random_int
  ct_params:
    low: 0
    high: 5
  split_at_iters: [1, 2]
  split_percentages: [0.5]
  percentage_split_at_iter: 1
  run_split_at_sweep: true
  run_percentage_sweep: true
  run_combined_sweep: true
"""
    )

    exit_code = main(["--config", str(config_path), "--out", str(out_dir)])

    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    run_names = {row["run_name"] for row in summary["runs"]}
    prefix = "random_graph_vars_4_density_1"
    assert f"{prefix}_standard" in run_names
    assert f"{prefix}_split_at_1_all_transfer" in run_names
    assert f"{prefix}_split_at_2_all_transfer" in run_names
    assert f"{prefix}_split_pct_50_at_1_transfer" in run_names
    assert f"{prefix}_split_grid_at_1_pct_50_transfer" in run_names
    assert f"{prefix}_split_grid_at_2_pct_50_transfer" in run_names
    assert not any(name.endswith("_reset") for name in run_names)
    assert "trace_standard.jsonl" not in {path.name for path in out_dir.iterdir()}
    for run_name in run_names:
        assert (out_dir / run_name / "seed_11_trace.jsonl").exists()
        assert (out_dir / run_name / "seed_11_snapshots.jsonl").exists()
        assert (out_dir / run_name / "seed_11_summary.json").exists()
    fingerprints = {
        row["random_graph_fingerprint"]
        for row in summary["runs"]
        if row["run_name"].startswith(prefix)
    }
    assert len(fingerprints) == 1
    assert all(row.get("random_seed") == 11 for row in summary["runs"])
    assert all(row.get("random_num_variable_nodes") == 4 for row in summary["runs"])
    assert all(row.get("random_density") == 1.0 for row in summary["runs"])
    assert summary["route_analysis"]["status"] == "skipped"


def test_mode_random_graph_overrides_missing_chain_tables(tmp_path):
    config_path = tmp_path / "template_like.yaml"
    out_dir = tmp_path / "out"
    config_path.write_text(
        """
graph_name: template_like
run_chain: true
variable_order: ["X1", "X2", "X3"]
domain_values: [0, 1]
cost_tables:
  - name: F12
    variables: ["X1", "X2"]
    table: null
  - name: F23
    variables: ["X2", "X3"]
    table: null
max_iter: 2
split_at_iters: [1]
damping_factors: []
trace_every: 1
seed: 13
random_graph:
  enabled: false
  num_vars: 4
  domain_size: 2
  density: 1.0
  ct_factory: random_int
  ct_params:
    low: 0
    high: 5
  split_percentages: [0.5]
  percentage_split_at_iter: 1
  run_split_at_sweep: false
  run_percentage_sweep: false
  run_combined_sweep: true
"""
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--mode",
            "random-graph",
            "--max-iter",
            "2",
            "--split-at",
            "1",
        ]
    )

    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    run_names = {row["run_name"] for row in summary["runs"]}
    prefix = "random_graph_vars_4_density_1"
    assert run_names == {
        f"{prefix}_standard",
        f"{prefix}_split_grid_at_1_pct_50_transfer",
    }
    for run_name in run_names:
        assert (out_dir / run_name / "seed_13_trace.jsonl").exists()
        assert (out_dir / run_name / "seed_13_summary.json").exists()


def test_random_graph_reset_mode_runs_only_when_configured(tmp_path):
    config_path = tmp_path / "random_with_reset.yaml"
    out_dir = tmp_path / "out"
    config_path.write_text(
        """
graph_name: random_with_reset
run_chain: false
variable_order: ["X1", "X2", "X3"]
domain_values: [0, 1]
cost_tables: []
max_iter: 2
tolerance: 1.0e-9
split_at_iters: [1]
split_ratio: 0.5
damping_factors: []
trace_every: 1
seed: 19
random_graph:
  enabled: true
  num_vars: 4
  domain_size: 2
  density: 1.0
  ct_factory: random_int
  ct_params:
    low: 0
    high: 5
  split_at_iters: [1]
  split_percentages: [0.5]
  percentage_split_at_iter: 1
  run_split_at_sweep: false
  run_percentage_sweep: false
  run_combined_sweep: true
  split_transfer_modes: ["transfer", "reset"]
"""
    )

    exit_code = main(["--config", str(config_path), "--out", str(out_dir)])

    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    run_names = {row["run_name"] for row in summary["runs"]}
    prefix = "random_graph_vars_4_density_1"
    assert f"{prefix}_split_grid_at_1_pct_50_transfer" in run_names
    assert f"{prefix}_split_grid_at_1_pct_50_reset" in run_names
