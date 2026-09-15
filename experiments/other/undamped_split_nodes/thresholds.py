"""Measure when incoming Q stops affecting outgoing R differences."""

from __future__ import annotations

import json

import numpy as np

from experiments.other.aaai_derived_control.code.kernel import PairwiseKernel, gauge
from propflow.snapshots.commitment import evaluate_commitment, row_thresholds

from .run import OUTPUT, dump, load_problem, write_csv


def run(family: str, steps: int = 10000) -> dict:
    """Replay the frozen trajectory and test the full and two-route thresholds."""
    p = load_problem(family)
    with np.load(OUTPUT / f"{family}_scout.npz") as saved:
        reference_x, reference_b = saved["assignments"], saved["beliefs"]
    with np.load(OUTPUT / f"{family}_messages.npz") as saved:
        reference_q = saved["tail_q"]
    analysis = json.loads((OUTPUT / f"{family}_analysis.json").read_text())
    # axis 0 sends from edge.u to edge.v; axis 1 sends from edge.v to edge.u
    tables = np.stack([p.costs, p.costs.transpose(0, 2, 1)], axis=1) * 0.5
    omega = row_thresholds(tables)
    receivers = p.edges[:, ::-1]
    routes = np.stack([reference_x[1998], reference_x[1999]], axis=-1)[receivers]
    eligible = routes[..., 0] != routes[..., 1]
    route_tables = np.take_along_axis(tables, routes[..., None, :], axis=-1)
    route_omega = row_thresholds(route_tables)
    k = PairwiseKernel(p, weights=0.5, damping=0)
    slacks = np.empty((steps, len(p.edges), 2))
    rows = np.empty((steps, len(p.edges), 2), dtype=np.int8)
    route_slacks = np.empty_like(slacks)
    audits = dict(
        assignment_mismatches=0,
        max_reference_belief_error=0.0,
        max_saved_q_error=0.0,
        max_clone_q_error=0.0,
        max_algebraic_cancellation_error=0.0,
        max_local_perturbation_error=0.0,
        max_native_forwarded_row_error=0.0,
        tail_native_active_mask_disagreements=0,
    )
    perturbation_pattern = np.where(np.arange(p.d) % 2 == 0, 1.0, -1.0)
    for t in range(steps):
        raw_q = k.beliefs()[k.ends] - k.r if 1700 <= t < 2000 else None
        k.step()
        all_q = gauge(k.q).reshape(len(p.edges), 2, 2, p.d)
        q = all_q[:, 0]
        state = evaluate_commitment(q, omega)
        route_state = evaluate_commitment(q, route_omega)
        slacks[t] = state.slack
        rows[t] = state.row
        route_slacks[t] = route_state.slack
        audits["assignment_mismatches"] += int(
            not np.array_equal(k.assignment, reference_x[t])
        )
        audits["max_reference_belief_error"] = max(
            audits["max_reference_belief_error"],
            float(np.abs(gauge(k.beliefs()) - reference_b[t]).max()),
        )
        audits["max_clone_q_error"] = max(
            audits["max_clone_q_error"], float(np.abs(all_q[:, 0] - all_q[:, 1]).max())
        )
        if 1699 <= t < 2000:
            audits["max_saved_q_error"] = max(
                audits["max_saved_q_error"],
                float(np.abs(gauge(k.q) - reference_q[t - 1699]).max()),
            )
        if 1700 <= t < 2000:
            scores = tables + q[..., None]
            outgoing = scores.min(axis=-2)
            forwarded = np.take_along_axis(tables, state.row[..., None, None], axis=-2)[
                ..., 0, :
            ]
            strict = state.strict
            audits["max_algebraic_cancellation_error"] = max(
                audits["max_algebraic_cancellation_error"],
                float(
                    np.abs(gauge(outgoing)[strict] - gauge(forwarded)[strict]).max(
                        initial=0
                    )
                ),
            )
            radius = np.maximum(state.slack, 0) * 0.2
            for sign in [-1, 1]:
                perturbed_q = q + sign * radius[..., None] * perturbation_pattern
                perturbed = (tables + perturbed_q[..., None]).min(axis=-2)
                audits["max_local_perturbation_error"] = max(
                    audits["max_local_perturbation_error"],
                    float(
                        np.abs(gauge(perturbed)[strict] - gauge(outgoing)[strict]).max(
                            initial=0
                        )
                    ),
                )
            native_r = gauge(k.r[::2, ::-1])
            audits["max_native_forwarded_row_error"] = max(
                audits["max_native_forwarded_row_error"],
                float(
                    np.abs(native_r[strict] - gauge(forwarded)[strict]).max(initial=0)
                ),
            )
            aggregate = k.tables + raw_q[:, 0, :, None] + raw_q[:, 1, None, :]
            native_scores = np.stack(
                [
                    (aggregate - raw_q[:, 1, None, :])[::2],
                    (aggregate - raw_q[:, 0, :, None])[::2].transpose(0, 2, 1),
                ],
                axis=1,
            )
            winners = native_scores.argmin(axis=-2)
            ordered = np.sort(native_scores, axis=-2)
            native_strict = (winners == winners[..., :1]).all(-1) & (
                np.diff(ordered[..., :2, :], axis=-2).min(axis=(-2, -1)) > 1e-6
            )
            audits["tail_native_active_mask_disagreements"] += int(
                np.sum(native_strict != strict)
            )
        if (t + 1) % 2000 == 0:
            print(
                family,
                t + 1,
                "full-threshold directions",
                int(state.strict.sum()),
                "/",
                2 * len(p.edges),
                flush=True,
            )
    if any(
        audits[key]
        for key in [
            "assignment_mismatches",
            "max_reference_belief_error",
            "max_saved_q_error",
            "max_clone_q_error",
        ]
    ):
        raise RuntimeError(f"replay changed: {audits}")
    if (
        max(
            audits["max_algebraic_cancellation_error"],
            audits["max_local_perturbation_error"],
        )
        > 1e-8
    ):
        raise RuntimeError(f"cancellation or perturbation check failed: {audits}")
    masks = slacks > 1e-6
    weak = slacks >= -1e-6
    pair_masks = (route_slacks > 1e-6) & eligible
    tail = masks[1700:2000]
    tail_rows = rows[1700:2000]
    always = tail.all(0)
    row_fixed = np.ptp(tail_rows, axis=0) == 0
    incoming = np.zeros((steps, p.n), dtype=np.int16)
    for e, (u, v) in enumerate(p.edges):
        incoming[:, v] += masks[:, e, 0]
        incoming[:, u] += masks[:, e, 1]
    degrees = np.bincount(p.edges.ravel(), minlength=p.n)
    fully_received = incoming == degrees
    by_direction = []
    for e, ends in enumerate(p.edges):
        for axis in range(2):
            reached = np.flatnonzero(masks[:, e, axis])
            phase_fraction = [float(tail[phase::2, e, axis].mean()) for phase in [0, 1]]
            by_direction.append(
                dict(
                    factor=p.factor_names[e],
                    sender=p.variable_names[ends[axis]],
                    receiver=p.variable_names[ends[1 - axis]],
                    clone_copies=2,
                    first_crossing=int(reached[0] + 1) if len(reached) else None,
                    ever_first_2000=bool(masks[:2000, e, axis].any()),
                    tail_always=bool(always[e, axis]),
                    tail_ever=bool(tail[:, e, axis].any()),
                    phase_A_fraction=phase_fraction[0],
                    phase_B_fraction=phase_fraction[1],
                    phase_A_slack=float(slacks[1998, e, axis]),
                    phase_B_slack=float(slacks[1999, e, axis]),
                    tail_min_slack=float(slacks[1700:2000, e, axis].min()),
                    tail_max_slack=float(slacks[1700:2000, e, axis].max()),
                    same_row_throughout_tail=bool(
                        always[e, axis] and row_fixed[e, axis]
                    ),
                    phase_A_row=(
                        int(rows[1998, e, axis]) if masks[1998, e, axis] else None
                    ),
                    phase_B_row=(
                        int(rows[1999, e, axis]) if masks[1999, e, axis] else None
                    ),
                    route_pair_eligible=bool(eligible[e, axis]),
                    route_pair_tail_always=(
                        bool(pair_masks[1700:2000, e, axis].all())
                        if eligible[e, axis]
                        else None
                    ),
                )
            )
    by_variable = []
    for i, n in enumerate(analysis["nodes"]):
        by_variable.append(
            dict(
                variable=p.variable_names[i],
                unsettled=n["unsettled"],
                degree=int(degrees[i]),
                incoming_independent_phase_A=int(incoming[1998, i]),
                incoming_independent_phase_B=int(incoming[1999, i]),
                incoming_independent_tail_min=int(incoming[1700:2000, i].min()),
                incoming_independent_tail_max=int(incoming[1700:2000, i].max()),
                all_incoming_independent_entire_tail=bool(
                    fully_received[1700:2000, i].all()
                ),
                all_incoming_independent_some_tail_step=bool(
                    fully_received[1700:2000, i].any()
                ),
            )
        )

    def window_summary(start, end):
        w = masks[start:end]
        both = w.all(-1)
        pair = pair_masks[start:end]
        return dict(
            directed_count_min=int(w.sum((1, 2)).min()),
            directed_count_max=int(w.sum((1, 2)).max()),
            directed_always=int(w.all(0).sum()),
            directed_ever=int(w.any(0).sum()),
            original_factors_both_directions_always=int(both.all(0).sum()),
            original_factors_both_directions_min=int(both.sum(-1).min()),
            original_factors_both_directions_max=int(both.sum(-1).max()),
            variables_all_incoming_always=int(fully_received[start:end].all(0).sum()),
            variables_all_incoming_ever=int(fully_received[start:end].any(0).sum()),
            pair_only_count_min=int(pair.sum((1, 2)).min()),
            pair_only_count_max=int(pair.sum((1, 2)).max()),
            pair_only_always=int(pair.all(0).sum()),
        )

    summary = dict(
        family=family,
        split=0.5,
        damping=0,
        steps=steps,
        strict_tolerance=1e-6,
        original_pairwise_factors=len(p.edges),
        split_pairwise_factors=2 * len(p.edges),
        distinct_directions=2 * len(p.edges),
        actual_pairwise_incoming_Q_messages=4 * len(p.edges),
        omitted_split_unary_factors=2 * p.n,
        phase_A_update=1999,
        phase_B_update=2000,
        phase_A_directed=int(masks[1998].sum()),
        phase_B_directed=int(masks[1999].sum()),
        phase_A_factors_both=int(masks[1998].all(-1).sum()),
        phase_B_factors_both=int(masks[1999].all(-1).sum()),
        phase_A_variables_all_incoming=int(fully_received[1998].sum()),
        phase_B_variables_all_incoming=int(fully_received[1999].sum()),
        ever_directed_2000=int(masks[:2000].any(0).sum()),
        ever_directed_10000=int(masks.any(0).sum()),
        ever_fully_independent_graph=bool(masks.all((1, 2)).any()),
        always_committed_same_row=int((always & row_fixed).sum()),
        always_committed_changing_row=int((always & ~row_fixed).sum()),
        weak_not_strict_tail_observations=int((weak[1700:2000] & ~tail).sum()),
        phase_A_pair=int(pair_masks[1998].sum()),
        phase_B_pair=int(pair_masks[1999].sum()),
        pair_eligible_directions=int(eligible.sum()),
        tail=window_summary(1700, 2000),
        extended_tail=window_summary(9700, 10000),
        audits=audits,
    )
    np.savez_compressed(
        OUTPUT / f"{family}_thresholds.npz",
        slack=slacks,
        row=rows,
        pair_slack=route_slacks,
        pair_eligible=eligible,
        incoming_counts=incoming,
    )
    write_csv(OUTPUT / f"{family}_threshold_directions.csv", by_direction)
    write_csv(OUTPUT / f"{family}_threshold_variables.csv", by_variable)
    write_csv(
        OUTPUT / f"{family}_threshold_time.csv",
        [
            dict(
                update=t + 1,
                full_count=int(masks[t].sum()),
                full_factors=int(masks[t].all(-1).sum()),
                all_incoming_variables=int(fully_received[t].sum()),
                pair_count=int(pair_masks[t].sum()),
            )
            for t in range(steps)
        ],
    )
    dump(OUTPUT / f"{family}_threshold_summary.json", summary)
    return summary


if __name__ == "__main__":
    for name in ["random_sparse", "random_dense"]:
        run(name)
