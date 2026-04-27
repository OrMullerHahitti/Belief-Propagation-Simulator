from experiments.other.non_convergence_chain.code.oscillation_detector import (
    classify_run,
    detect_assignment_convergence,
    detect_period,
)


def _trace(values):
    return [
        {"iteration": idx, "assignments": {"X1": value}}
        for idx, value in enumerate(values)
    ]


def test_synthetic_period_2_trace():
    trace = _trace([0, 1, 0, 1, 0, 1, 0, 1])

    period = detect_period(trace, max_period=4, min_repeats=3)

    assert period["period"] == 2
    assert period["start"] == 0
    assert classify_run(trace) == "period_2_oscillation"


def test_synthetic_convergent_trace():
    trace = _trace([1, 0, 0, 0, 0, 0, 0])

    assert detect_assignment_convergence(trace, window=5)
    assert classify_run(trace) == "converged"


def test_synthetic_transient_then_periodic_trace():
    trace = _trace([0, 1, 1, 0, 1, 0, 1, 0])

    period = detect_period(trace, max_period=4, min_repeats=3)

    assert period["period"] == 2
    assert period["start"] == 2
    assert classify_run(trace) == "transient_then_oscillation"
