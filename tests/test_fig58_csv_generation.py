import pandas as pd

from experiments.generate_fig58_csv import (
    build_examples_dataframe,
    build_traces_dataframe,
    generate_fig58_csv_datasets,
)
from experiments.utils.fig58_repro import (
    CASE_CONSISTENT_NO_TAIL,
    CASE_CONSISTENT_WITH_TAIL,
    CASE_INCONSISTENT_NO_TAIL,
    GEN_MOTIF_REPEAT,
    GEN_RANDOM_FULL,
    classify_case_cycle,
    find_case_cycle,
    find_cases_cycle,
    run_experiment_examples_cycle,
)


CT_5A = (
    [[7.0, 2.0], [5.0, 4.0]],
    [[1.0, 7.0], [5.0, 1.0]],
    [[4.0, 0.0], [9.0, 5.0]],
)

CT_5B = (
    [[6.0, 3.0], [7.0, 4.0]],
    [[6.0, 9.0], [2.0, 6.0]],
    [[7.0, 4.0], [3.0, 7.0]],
)

CT_8 = (
    [[9.0, 5.0], [2.0, 8.0]],
    [[4.0, 9.0], [8.0, 5.0]],
    [[9.0, 1.0], [5.0, 9.0]],
)


def test_classify_case_cycle_matches_known_three_node_examples():
    c5a = classify_case_cycle(CT_5A, max_iter=120)
    c5b = classify_case_cycle(CT_5B, max_iter=120)
    c8 = classify_case_cycle(CT_8, max_iter=120)

    assert c5a[CASE_CONSISTENT_NO_TAIL]
    assert not c5a[CASE_CONSISTENT_WITH_TAIL]
    assert not c5a[CASE_INCONSISTENT_NO_TAIL]

    assert c5b[CASE_CONSISTENT_WITH_TAIL]
    assert not c5b[CASE_CONSISTENT_NO_TAIL]
    assert not c5b[CASE_INCONSISTENT_NO_TAIL]

    assert c8[CASE_INCONSISTENT_NO_TAIL]
    assert not c8[CASE_CONSISTENT_NO_TAIL]
    assert not c8[CASE_CONSISTENT_WITH_TAIL]


def test_find_case_cycle_figure8_strict_large_cycle_with_motif_repeat():
    result = find_case_cycle(
        case_name=CASE_INCONSISTENT_NO_TAIL,
        cycle_size=12,
        seed=488,
        generation_strategy=GEN_MOTIF_REPEAT,
        max_attempts=1,
        classify_max_iter=140,
    )
    c = result["classification"]
    assert c[CASE_INCONSISTENT_NO_TAIL]
    assert c["no_tail"]
    assert c["inconsistent"]
    assert all(tuple(values) == (0, 1) for values in c["route_values_by_var"])


def test_run_experiment_examples_cycle_key_counts():
    ex_5a = find_cases_cycle(
        CASE_CONSISTENT_NO_TAIL,
        n_examples=2,
        cycle_size=6,
        seed_start=42,
        max_attempts_per_seed=2000,
        generation_strategy=GEN_RANDOM_FULL,
    )
    runs_5a = run_experiment_examples_cycle(
        ex_5a,
        case_name=CASE_CONSISTENT_NO_TAIL,
        max_iter=12,
        damping_factor=0.9,
        normalize_messages=False,
        subtract_initial=False,
    )
    assert len(runs_5a) == 2
    assert runs_5a[0]["key_order"] == [f"x{i+1}_route" for i in range(6)]
    assert len(runs_5a[0]["records"]) == 6

    ex_8 = find_cases_cycle(
        CASE_INCONSISTENT_NO_TAIL,
        n_examples=1,
        cycle_size=6,
        seed_start=488,
        max_attempts_per_seed=500,
        generation_strategy=GEN_MOTIF_REPEAT,
        classify_max_iter=140,
    )
    runs_8 = run_experiment_examples_cycle(
        ex_8,
        case_name=CASE_INCONSISTENT_NO_TAIL,
        max_iter=12,
        damping_factor=0.9,
        normalize_messages=False,
        subtract_initial=False,
    )
    assert len(runs_8) == 1
    assert runs_8[0]["key_order"] == [f"x{i+1}_v{v}" for v in range(2) for i in range(6)]
    assert len(runs_8[0]["records"]) == 12


def test_csv_builders_schema_and_counts():
    examples = find_cases_cycle(
        CASE_CONSISTENT_WITH_TAIL,
        n_examples=2,
        cycle_size=3,
        seed_start=42,
        max_attempts_per_seed=2000,
        generation_strategy=GEN_RANDOM_FULL,
    )
    runs = run_experiment_examples_cycle(
        examples=examples,
        case_name=CASE_CONSISTENT_WITH_TAIL,
        max_iter=15,
        damping_factor=0.9,
        normalize_messages=False,
        subtract_initial=False,
    )

    examples_df = build_examples_dataframe(
        figure_id="figure_5b",
        case_name=CASE_CONSISTENT_WITH_TAIL,
        cycle_size=3,
        examples=examples,
    )
    traces_df = build_traces_dataframe(
        figure_id="figure_5b",
        case_name=CASE_CONSISTENT_WITH_TAIL,
        cycle_size=3,
        runs=runs,
    )

    required_example_cols = {
        "figure_id",
        "case_name",
        "cycle_size",
        "example_index",
        "seed",
        "generation_strategy",
        "period",
        "periodic_start",
        "consistent",
        "inconsistent",
        "no_tail",
        "unclassified",
        "route_values_by_var_json",
        "periodic_route_json",
        "assignment_trace_json",
        "cost_tables_json",
    }
    required_trace_cols = {
        "figure_id",
        "case_name",
        "cycle_size",
        "example_index",
        "seed",
        "iteration",
        "series_key",
        "variable_name",
        "tracked_kind",
        "tracked_value",
        "belief",
    }

    assert required_example_cols.issubset(set(examples_df.columns))
    assert required_trace_cols.issubset(set(traces_df.columns))
    assert len(examples_df) == 2
    assert len(traces_df) == 2 * 3 * 15


def test_generate_fig58_csv_datasets_writes_files(tmp_path):
    out_root = tmp_path / "generated"
    generate_fig58_csv_datasets(output_root=out_root, n_examples=1, cycle_sizes=(3,))

    figure_specs = [
        ("figure_5a", 3, 50),
        ("figure_5b", 3, 50),
        ("figure_8", 6, 70),
    ]
    for figure_id, series_count, max_iter in figure_specs:
        combo_dir = out_root / figure_id / "cycle_3"
        examples_csv = combo_dir / "examples.csv"
        traces_csv = combo_dir / "traces.csv"
        assert examples_csv.exists()
        assert traces_csv.exists()

        examples_df = pd.read_csv(examples_csv)
        traces_df = pd.read_csv(traces_csv)

        assert len(examples_df) == 1
        assert len(traces_df) == 1 * series_count * max_iter
