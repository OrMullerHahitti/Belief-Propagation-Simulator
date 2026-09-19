"""Follow threshold entry, exit, and plateau changes through the saved two-cycle."""

from __future__ import annotations

import json

import numpy as np

from propflow.snapshots.commitment import evaluate_commitment, row_thresholds

from .run import OUTPUT, dump, load_problem, write_csv


def classify_window(passed: np.ndarray, plateau: np.ndarray) -> np.ndarray:
    """Classify each direction across at least three consecutive observations."""
    passed = np.asarray(passed, dtype=bool)
    plateau = np.asarray(plateau, dtype=float)
    if passed.ndim != 2 or passed.shape != plateau.shape[:2] or len(passed) < 3:
        raise ValueError("matching time-by-direction arrays with at least three steps")
    if not np.isfinite(plateau).all():
        raise ValueError("plateau differences must be finite")
    always, never = passed.all(0), ~passed.any(0)
    constant = (np.ptp(plateau, axis=0) <= 1e-9).reshape(passed.shape[1], -1).all(-1)
    period_two = (
        (np.abs(plateau[2:] - plateau[:-2]) <= 1e-9)
        .all(0)
        .reshape(passed.shape[1], -1)
        .all(-1)
    )
    crossings = np.count_nonzero(passed[1:] != passed[:-1], axis=0)
    result = np.full(passed.shape[1], "other", dtype=object)
    result[always & constant] = "above_constant_output"
    result[always & ~constant & period_two] = "above_alternating_output"
    result[crossings >= 2] = "crosses_and_returns"
    result[never] = "always_below"
    return result


def run(family: str) -> dict:
    """Analyze saved Q/R over updates 1995–2000 and verify the 300-update tail."""
    p = load_problem(family)
    with np.load(OUTPUT / f"{family}_scout.npz") as saved:
        assignments = saved["assignments"]
    with np.load(OUTPUT / f"{family}_messages.npz") as saved:
        all_q = saved["tail_q"][1:]
        q = all_q[:, ::2]
        # reverse the endpoint axis to align sender Q with receiver R
        outgoing = saved["tail_r"][1:, ::2, ::-1]
    np.testing.assert_array_equal(all_q[:, ::2], all_q[:, 1::2])
    np.testing.assert_array_equal(assignments[1702:2000], assignments[1700:1998])
    receivers = p.edges[:, ::-1]
    routes = np.stack([assignments[1998], assignments[1999]], axis=-1)[receivers]
    eligible = routes[..., 0] != routes[..., 1]
    tables = np.stack([p.costs, p.costs.transpose(0, 2, 1)], axis=1) * 0.5
    pair = np.take_along_axis(tables, routes[..., None, :], axis=-1)
    state = evaluate_commitment(q, row_thresholds(pair))
    with np.load(OUTPUT / f"{family}_thresholds.npz") as saved:
        np.testing.assert_array_equal(state.slack, saved["pair_slack"][1700:2000])
        np.testing.assert_array_equal(eligible, saved["pair_eligible"])
    pair_difference = pair[..., 1] - pair[..., 0]
    plateau = np.take_along_axis(
        np.broadcast_to(pair_difference, q.shape), state.row[..., None], axis=-1
    )[..., 0]
    r_pair = np.take_along_axis(outgoing, routes[None], axis=-1)
    actual_difference = r_pair[..., 1] - r_pair[..., 0]
    tail_categories = classify_window(state.strict[:, eligible], plateau[:, eligible])
    categories = classify_window(state.strict[-6:, eligible], plateau[-6:, eligible])
    np.testing.assert_array_equal(categories, tail_categories)
    labels, counts = np.unique(categories, return_counts=True)
    summary = dict(
        family=family,
        updates=list(range(1995, 2001)),
        comparison="outgoing difference between the receiver's two visited values",
        sender_domain_values=p.d,
        distinct_directions=int(eligible.sum()),
        oscillating_receivers=int((assignments[1998] != assignments[1999]).sum()),
        clone_copies_per_direction=2,
        counts=dict(zip(labels.tolist(), counts.tolist())),
        same_classification_throughout_updates_1701_to_2000=True,
        strict_tolerance=1e-6,
    )
    patterns, pattern_counts = np.unique(
        state.strict[-6:, eligible].T, axis=0, return_counts=True
    )
    summary["six_update_patterns"] = {
        " ".join("above" if b else "below" for b in pattern): int(count)
        for pattern, count in zip(patterns, pattern_counts)
    }
    directions, trace = [], []
    for j, (edge, axis) in enumerate(np.argwhere(eligible)):
        ends = p.edges[edge]
        identity = dict(
            factor=p.factor_names[edge],
            sender=p.variable_names[ends[axis]],
            receiver=p.variable_names[ends[1 - axis]],
            receiver_value_A=int(routes[edge, axis, 0]),
            receiver_value_B=int(routes[edge, axis, 1]),
        )
        directions.append(
            dict(
                **identity,
                category=categories[j],
                six_update_pattern=" ".join(
                    "above" if b else "below" for b in state.strict[-6:, edge, axis]
                ),
            )
        )
        for t in range(294, 300):
            trace.append(
                dict(
                    **identity,
                    update=1701 + t,
                    threshold_margin=float(state.slack[t, edge, axis]),
                    above_threshold=bool(state.strict[t, edge, axis]),
                    outgoing_difference_B_minus_A=float(
                        actual_difference[t, edge, axis]
                    ),
                )
            )
    write_csv(OUTPUT / f"{family}_threshold_transition_directions.csv", directions)
    write_csv(OUTPUT / f"{family}_threshold_transition_trace.csv", trace)

    # fixed-verdict receivers have no second visited value, so check all alternatives
    fixed = ~eligible
    full = evaluate_commitment(q, row_thresholds(tables))
    with np.load(OUTPUT / f"{family}_thresholds.npz") as saved:
        np.testing.assert_array_equal(full.slack, saved["slack"][1700:2000])
    forwarded = np.take_along_axis(
        np.broadcast_to(tables, q.shape + (p.d,)),
        full.row[..., None, None],
        axis=-2,
    )[..., 0, :]
    reference_values = routes[..., :1]
    fixed_plateau = forwarded - np.take_along_axis(
        forwarded, reference_values[None], axis=-1
    )
    fixed_actual = outgoing - np.take_along_axis(
        outgoing, reference_values[None], axis=-1
    )
    fixed_categories = classify_window(
        full.strict[-6:, fixed], fixed_plateau[-6:, fixed]
    )
    fixed_tail_categories = classify_window(
        full.strict[:, fixed], fixed_plateau[:, fixed]
    )
    fixed_labels, fixed_counts = np.unique(fixed_categories, return_counts=True)
    fixed_summary = dict(
        family=family,
        updates=list(range(1995, 2001)),
        comparison="held receiver value against all nine alternatives",
        fixed_receivers=int((assignments[1998] == assignments[1999]).sum()),
        distinct_directions=int(fixed.sum()),
        clone_copies_per_direction=2,
        counts=dict(zip(fixed_labels.tolist(), fixed_counts.tolist())),
        same_classification_throughout_updates_1701_to_2000=bool(
            np.array_equal(fixed_categories, fixed_tail_categories)
        ),
        strict_tolerance=1e-6,
    )
    patterns, pattern_counts = np.unique(
        full.strict[-6:, fixed].T, axis=0, return_counts=True
    )
    fixed_summary["six_update_patterns"] = {
        " ".join("above" if b else "below" for b in pattern): int(count)
        for pattern, count in zip(patterns, pattern_counts)
    }
    fixed_directions, fixed_trace = [], []
    for j, (edge, axis) in enumerate(np.argwhere(fixed)):
        ends = p.edges[edge]
        identity = dict(
            factor=p.factor_names[edge],
            sender=p.variable_names[ends[axis]],
            receiver=p.variable_names[ends[1 - axis]],
            held_receiver_value=int(reference_values[edge, axis, 0]),
        )
        fixed_directions.append(
            dict(
                **identity,
                category=fixed_categories[j],
                six_update_pattern=" ".join(
                    "above" if b else "below" for b in full.strict[-6:, edge, axis]
                ),
            )
        )
        for t in range(294, 300):
            fixed_trace.append(
                dict(
                    **identity,
                    update=1701 + t,
                    threshold_margin=float(full.slack[t, edge, axis]),
                    above_threshold=bool(full.strict[t, edge, axis]),
                    outgoing_differences_from_held_value=json.dumps(
                        fixed_actual[t, edge, axis].tolist()
                    ),
                )
            )
    write_csv(
        OUTPUT / f"{family}_fixed_threshold_transition_directions.csv", fixed_directions
    )
    write_csv(OUTPUT / f"{family}_fixed_threshold_transition_trace.csv", fixed_trace)
    variable_rows = []
    for i, name in enumerate(p.variable_names):
        oscillating = bool(assignments[1998, i] != assignments[1999, i])
        selected = directions if oscillating else fixed_directions
        incident = [row for row in selected if row["receiver"] == name]
        above = sum(row["category"].startswith("above_") for row in incident)
        round_trips = sum(row["category"] == "crosses_and_returns" for row in incident)
        below = sum(row["category"] == "always_below" for row in incident)
        other = sum(row["category"] == "other" for row in incident)
        assert above + round_trips + below + other == len(incident)
        variable_rows.append(
            dict(
                variable=name,
                verdict_behavior="alternating" if oscillating else "fixed",
                value_A=int(assignments[1998, i]),
                value_B=int(assignments[1999, i]),
                incoming_directions=len(incident),
                always_above=above,
                crosses_and_returns=round_trips,
                always_below=below,
                other=other,
            )
        )
    for behavior, target in [("alternating", summary), ("fixed", fixed_summary)]:
        selected = [row for row in variable_rows if row["verdict_behavior"] == behavior]
        target["receiver_counts"] = dict(
            at_least_one_direction_crosses_and_returns=sum(
                row["crosses_and_returns"] > 0 for row in selected
            ),
            all_directions_above_every_update=sum(
                row["always_above"] == row["incoming_directions"] for row in selected
            ),
        )
    write_csv(OUTPUT / f"{family}_threshold_transition_variables.csv", variable_rows)
    dump(OUTPUT / f"{family}_threshold_transition_summary.json", summary)
    dump(OUTPUT / f"{family}_fixed_threshold_transition_summary.json", fixed_summary)
    return dict(oscillating=summary, fixed=fixed_summary)


if __name__ == "__main__":
    for family in ["random_sparse", "random_dense"]:
        print(run(family), flush=True)
