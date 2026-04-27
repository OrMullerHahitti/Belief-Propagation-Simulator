import numpy as np

from experiments.aij.code.utils.fig58_repro import (
    aggregate_example_records,
    CASE_CONSISTENT_NO_TAIL,
    CASE_CONSISTENT_WITH_TAIL,
    CASE_INCONSISTENT_NO_TAIL,
    classify_case,
    compute_example_slope_summary,
    compute_slope_stats,
    find_case,
    find_cases,
    route_assignment_from_classification,
    run_experiment_examples,
    run_belief_trace,
)


# Deterministic, prevalidated cost tables used in the reproduction notebook.
CT_5A_F12 = np.array([[7.0, 2.0], [5.0, 4.0]])
CT_5A_F23 = np.array([[1.0, 7.0], [5.0, 1.0]])
CT_5A_F31 = np.array([[4.0, 0.0], [9.0, 5.0]])

CT_5B_F12 = np.array([[6.0, 3.0], [7.0, 4.0]])
CT_5B_F23 = np.array([[6.0, 9.0], [2.0, 6.0]])
CT_5B_F31 = np.array([[7.0, 4.0], [3.0, 7.0]])

CT_8_F12 = np.array([[9.0, 5.0], [2.0, 8.0]])
CT_8_F23 = np.array([[4.0, 9.0], [8.0, 5.0]])
CT_8_F31 = np.array([[9.0, 1.0], [5.0, 9.0]])


def test_classify_case_known_examples():
    c5a = classify_case(CT_5A_F12, CT_5A_F23, CT_5A_F31, max_iter=120)
    c5b = classify_case(CT_5B_F12, CT_5B_F23, CT_5B_F31, max_iter=120)
    c8 = classify_case(CT_8_F12, CT_8_F23, CT_8_F31, max_iter=120)

    assert c5a[CASE_CONSISTENT_NO_TAIL]
    assert not c5a[CASE_CONSISTENT_WITH_TAIL]
    assert not c5a[CASE_INCONSISTENT_NO_TAIL]

    assert c5b[CASE_CONSISTENT_WITH_TAIL]
    assert not c5b[CASE_CONSISTENT_NO_TAIL]
    assert not c5b[CASE_INCONSISTENT_NO_TAIL]

    assert c8[CASE_INCONSISTENT_NO_TAIL]
    assert not c8[CASE_CONSISTENT_NO_TAIL]
    assert not c8[CASE_CONSISTENT_WITH_TAIL]


def test_find_case_returns_requested_category():
    for case in [
        CASE_CONSISTENT_NO_TAIL,
        CASE_CONSISTENT_WITH_TAIL,
        CASE_INCONSISTENT_NO_TAIL,
    ]:
        result = find_case(
            case_name=case,
            seed=42,
            domain=2,
            low=0,
            high=10,
            max_attempts=5000,
            classify_max_iter=120,
        )
        assert result["classification"][case]


def test_run_belief_trace_and_slope_stats_for_consistent_case():
    classification = classify_case(CT_5A_F12, CT_5A_F23, CT_5A_F31, max_iter=120)
    route = route_assignment_from_classification(classification)

    records = run_belief_trace(
        CT_5A_F12,
        CT_5A_F23,
        CT_5A_F31,
        tracked_values={"x1": [route[0]], "x2": [route[1]], "x3": [route[2]]},
        max_iter=70,
        damping_factor=0.9,
        normalize_messages=False,
    )

    assert set(records.keys()) == {
        f"x1_v{route[0]}",
        f"x2_v{route[1]}",
        f"x3_v{route[2]}",
    }
    assert all(len(v) == 70 for v in records.values())

    stats = compute_slope_stats(records, 50, 70)
    assert set(stats.keys()) == {"slopes", "relative_spread_percent"}
    assert len(stats["slopes"]) == 3
    assert stats["relative_spread_percent"] >= 0.0


def test_figure8_late_regime_has_two_competing_values_per_variable():
    c8 = classify_case(CT_8_F12, CT_8_F23, CT_8_F31, max_iter=120)

    assert c8[CASE_INCONSISTENT_NO_TAIL]
    assert c8["no_tail"]
    assert c8["inconsistent"]
    assert c8["route_values_by_var"] == ((0, 1), (0, 1), (0, 1))


def test_find_cases_returns_n_deterministic_examples():
    first = find_cases(
        CASE_CONSISTENT_NO_TAIL,
        n_examples=3,
        seed_start=42,
        seed_step=1,
        max_attempts_per_seed=2000,
        classify_max_iter=120,
    )
    second = find_cases(
        CASE_CONSISTENT_NO_TAIL,
        n_examples=3,
        seed_start=42,
        seed_step=1,
        max_attempts_per_seed=2000,
        classify_max_iter=120,
    )

    assert len(first) == 3
    assert len(second) == 3

    sig_a = [
        (
            ex["seed"],
            ex["attempt"],
            tuple(ex["ct_f12"].reshape(-1).tolist()),
            tuple(ex["ct_f23"].reshape(-1).tolist()),
            tuple(ex["ct_f31"].reshape(-1).tolist()),
        )
        for ex in first
    ]
    sig_b = [
        (
            ex["seed"],
            ex["attempt"],
            tuple(ex["ct_f12"].reshape(-1).tolist()),
            tuple(ex["ct_f23"].reshape(-1).tolist()),
            tuple(ex["ct_f31"].reshape(-1).tolist()),
        )
        for ex in second
    ]
    assert sig_a == sig_b


def test_run_experiment_examples_key_alignment():
    ex_5a = find_cases(
        CASE_CONSISTENT_NO_TAIL,
        n_examples=2,
        seed_start=42,
        seed_step=1,
        max_attempts_per_seed=2000,
    )
    runs_5a = run_experiment_examples(
        ex_5a,
        case_name=CASE_CONSISTENT_NO_TAIL,
        max_iter=30,
        damping_factor=0.9,
        normalize_messages=False,
        subtract_initial=False,
    )
    assert len(runs_5a) == 2
    assert runs_5a[0]["key_order"] == ["x1_route", "x2_route", "x3_route"]
    assert set(runs_5a[0]["records"].keys()) == {"x1_route", "x2_route", "x3_route"}

    ex_8 = find_cases(
        CASE_INCONSISTENT_NO_TAIL,
        n_examples=2,
        seed_start=42,
        seed_step=1,
        max_attempts_per_seed=4000,
    )
    runs_8 = run_experiment_examples(
        ex_8,
        case_name=CASE_INCONSISTENT_NO_TAIL,
        max_iter=30,
        damping_factor=0.9,
        normalize_messages=False,
        subtract_initial=False,
    )
    assert len(runs_8) == 2
    assert runs_8[0]["key_order"] == [
        "x1_v0",
        "x2_v0",
        "x3_v0",
        "x1_v1",
        "x2_v1",
        "x3_v1",
    ]
    assert set(runs_8[0]["records"].keys()) == {
        "x1_v0",
        "x2_v0",
        "x3_v0",
        "x1_v1",
        "x2_v1",
        "x3_v1",
    }


def test_aggregate_example_records_shapes():
    ex_5b = find_cases(
        CASE_CONSISTENT_WITH_TAIL,
        n_examples=3,
        seed_start=42,
        seed_step=1,
        max_attempts_per_seed=3000,
    )
    runs = run_experiment_examples(
        ex_5b,
        case_name=CASE_CONSISTENT_WITH_TAIL,
        max_iter=40,
        damping_factor=0.9,
        normalize_messages=False,
        subtract_initial=False,
    )
    agg = aggregate_example_records(
        runs, key_order=["x1_route", "x2_route", "x3_route"]
    )
    assert agg["n_examples"] == 3
    assert list(agg["mean_records"].keys()) == ["x1_route", "x2_route", "x3_route"]
    assert list(agg["std_records"].keys()) == ["x1_route", "x2_route", "x3_route"]
    assert len(agg["mean_records"]["x1_route"]) == 40
    assert len(agg["std_records"]["x1_route"]) == 40


def test_compute_example_slope_summary_columns():
    ex_5a = find_cases(
        CASE_CONSISTENT_NO_TAIL,
        n_examples=2,
        seed_start=42,
        seed_step=1,
        max_attempts_per_seed=2000,
    )
    runs = run_experiment_examples(
        ex_5a,
        case_name=CASE_CONSISTENT_NO_TAIL,
        max_iter=40,
        damping_factor=0.9,
        normalize_messages=False,
        subtract_initial=False,
    )
    df = compute_example_slope_summary(runs, window_start=20, window_end=40)
    required_cols = {
        "experiment",
        "example_index",
        "seed",
        "attempt",
        "series_key",
        "slope",
        "example_relative_spread_percent",
    }
    assert not df.empty
    assert required_cols.issubset(set(df.columns))
