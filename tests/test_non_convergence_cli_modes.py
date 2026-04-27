import json

from experiments.other.non_convergence_chain.code.config import build_chain_graph, load_config
from experiments.other.non_convergence_chain.code.run_non_convergence_study import main


def test_template_cost_tables_fail_clearly(tmp_path, capsys):
    config_path = tmp_path / "missing_tables.yaml"
    config_path.write_text(
        """
graph_name: missing_tables
run_chain: true
variable_order: ["X1", "X2", "X3"]
domain_values: [0, 1]
cost_tables:
  - name: F12
    variables: ["X1", "X2"]
    table: null
  - name: F23
    variables: ["X2", "X3"]
    table: [[0, 1], [1, 0]]
random_graph:
  enabled: false
"""
    )

    exit_code = main(["--config", str(config_path), "--out", str(tmp_path / "out")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Cost table for factor 'F12' is missing" in captured.err


def test_config_loads_and_builds_symmetric_chain(tmp_path):
    config_path = _write_meeting_like_config(tmp_path)

    cfg = load_config(config_path)
    graph = build_chain_graph(cfg)

    assert cfg.graph_name == "test_chain"
    assert cfg.domain_size == 2
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


def test_standard_cli_writes_deterministic_trace_and_classification(tmp_path):
    config_path = _write_meeting_like_config(tmp_path)
    out_dir = tmp_path / "out"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--max-iter",
            "40",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "condition_report.md").exists()
    assert (out_dir / "standard" / "trace.jsonl").exists()
    assert (out_dir / "standard" / "snapshots.jsonl").exists()
    assert (out_dir / "standard" / "summary.json").exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["route_analysis"]["status"] == "skipped"
    assert len(summary["runs"]) == 1
    run = summary["runs"][0]
    assert run["run_name"] == "standard"
    assert run["trace_path"] == "standard/trace.jsonl"
    assert run["snapshots_path"] == "standard/snapshots.jsonl"
    assert run["classification"] == "transient_then_oscillation"
    assert run["period"] == 2
    assert run["tail_start"] == 7
    assert run["normalize_messages"] is True
    assert run["stop_on_convergence"] is False

    trace_rows = (out_dir / "standard" / "trace.jsonl").read_text().splitlines()
    assert len(trace_rows) == 40
    first = json.loads(trace_rows[0])
    assert first["iteration"] == 0
    assert set(first["assignments"]) == {"X1", "X2", "X3"}


def test_config_can_disable_snapshot_output(tmp_path):
    config_path = _write_meeting_like_config(tmp_path, save_snapshots=False)
    out_dir = tmp_path / "out"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--max-iter",
            "20",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "standard" / "trace.jsonl").exists()
    assert not (out_dir / "standard" / "snapshots.jsonl").exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    run = summary["runs"][0]
    assert run["save_snapshots"] is False
    assert run["snapshots_path"] is None


def test_random_graph_config_is_rejected_in_minimal_slice(tmp_path, capsys):
    config_path = _write_meeting_like_config(tmp_path, random_enabled=True)

    exit_code = main(["--config", str(config_path), "--out", str(tmp_path / "out")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Random-graph studies are outside this minimal slice" in captured.err


def _write_meeting_like_config(
    tmp_path, *, random_enabled: bool = False, save_snapshots: bool = True
):
    config_path = tmp_path / "chain.yaml"
    config_path.write_text(
        f"""
graph_name: test_chain
run_chain: true
symmetric_chain_split: true
symmetric_chain_copies: 2
symmetric_chain_cost_scale: 1.0
variable_order: ["X1", "X2", "X3"]
domain_values: [0, 1]
cost_tables:
  - name: F12
    variables: ["X1", "X2"]
    table: [[50, 60], [20, 300]]
  - name: F23
    variables: ["X2", "X3"]
    table: [[45, 100], [300, 6]]
max_iter: 40
tolerance: 1.0e-9
trace_every: 1
save_snapshots: {str(save_snapshots).lower()}
output_dir: unused
seed: 17
random_graph:
  enabled: {str(random_enabled).lower()}
"""
    )
    return config_path
