"""Replay two saved AAAI graphs and inspect undamped equal-split dynamics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from experiments.other.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    PairwiseProblem,
    gauge,
)
from propflow import SplitEngine
from propflow.snapshots import EngineSnapshot, VariableDynamicsAnalyzer
from propflow.snapshots.variable_dynamics import component_profile, cost_table_profile


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/descriptive_splitting_damping"
INPUT = OUTPUT / "inputs"
SELECTED = {"random_sparse": [0, 27, 6], "random_dense": [40, 43, 47, 20]}


def load_problem(family: str) -> PairwiseProblem:
    """Load saved tables and their original ordered axes without regeneration."""
    with np.load(INPUT / f"{family}_5000.npz") as a:
        return PairwiseProblem(
            a["edges"],
            a["costs"],
            a["unary"],
            family,
            5000,
            family,
            *[tuple(a[k]) for k in ("variable_names", "factor_names", "unary_names")],
        )


def dump(path: Path, value: object) -> None:
    """Write strict JSON, converting NumPy scalar and array values."""

    def convert(x):
        return x.tolist() if isinstance(x, np.ndarray) else x.item()

    path.write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scout(family: str) -> None:
    """Record all verdicts and beliefs through 10,000 completed updates."""
    p = load_problem(family)
    k = PairwiseKernel(p, weights=0.5, damping=0)
    xs, bs, costs = [], [], []
    for _ in range(10000):
        k.step()
        xs.append(k.assignment.copy())
        bs.append(gauge(k.beliefs()))
        costs.append(k.cost)
    np.savez_compressed(
        OUTPUT / f"{family}_scout.npz", assignments=xs, beliefs=bs, costs=costs
    )


class VerdictSnapshotManager:
    """Capture native decoded assignments without retaining all engine state."""

    def capture_step(self, step_index, step, engine):
        return SimpleNamespace(
            step=step_index,
            global_cost=None,
            assignments={v.name: v.curr_assignment for v in engine.var_nodes},
        )


def native_trace(family: str) -> None:
    """Validate all native pairwise messages and retain selected full traces."""
    p = load_problem(family)
    k = PairwiseKernel(p, weights=0.5, damping=0)
    engine = SplitEngine(
        p.to_native(),
        split_factor=0.5,
        anytime=False,
        normalize_messages=True,
        snapshot_manager=VerdictSnapshotManager(),
    )
    variables = {v.name: v for v in engine.var_nodes}
    selected = SELECTED[family]
    incident = sorted(
        {e for e, ends in enumerate(p.edges) if set(ends) & set(selected)}
    )
    clone_indices = np.array([2 * e + c for e in incident for c in range(2)])
    q_saved, r_saved, uq_saved, ur_saved = [], [], [], []
    tail_q, tail_r, tail_uq, tail_ur = [], [], [], []
    report = dict(
        engine="SplitEngine",
        split=0.5,
        damping=0.0,
        steps=2000,
        max_raw_message_error=0.0,
        max_gauge_message_error=0.0,
        max_cost_error=0.0,
        assignment_mismatch_steps=[],
        input_sha256=hashlib.sha256(
            (INPUT / f"{family}_5000.npz").read_bytes()
        ).hexdigest(),
    )
    reference = np.load(OUTPUT / f"{family}_scout.npz")["assignments"]
    for t in range(2000):
        k.step()
        step = engine.step(t)
        snapshot = engine.latest_snapshot()
        x = np.array([snapshot.assignments[n] for n in p.variable_names])
        if not np.array_equal(x, reference[t]):
            report["assignment_mismatch_steps"].append(t + 1)
        report["max_cost_error"] = max(
            report["max_cost_error"], abs(snapshot.global_cost - k.cost)
        )
        try:
            engine._handle_cycle_events(t)
        except StopIteration:
            pass
        qmap = {
            (name, m.recipient.name): m.data
            for name, messages in step.q_messages.items()
            for m in messages
        }
        rmap = {
            (m.sender.name, v.name): m.data for v in variables.values() for m in v.inbox
        }
        native_q = np.array(
            [
                [qmap[(p.variable_names[v], name + suffix)] for v in ends]
                for name, ends in zip(p.factor_names, p.edges)
                for suffix in ("'", "''")
            ]
        )
        native_r = np.array(
            [
                [rmap[(name + suffix, p.variable_names[v])] for v in ends]
                for name, ends in zip(p.factor_names, p.edges)
                for suffix in ("'", "''")
            ]
        )
        native_ur = np.array(
            [
                [rmap[(name + suffix, p.variable_names[v])] for suffix in ("'", "''")]
                for v, name in enumerate(p.unary_names)
            ]
        )
        # plain SplitEngine's last Q may be unnormalized; compare its gauge too
        for native, fast in ((native_q, k.q), (native_r, k.r), (native_ur, k.unary_r)):
            report["max_raw_message_error"] = max(
                report["max_raw_message_error"], float(np.abs(native - fast).max())
            )
            report["max_gauge_message_error"] = max(
                report["max_gauge_message_error"],
                float(np.abs(gauge(native) - gauge(fast)).max()),
            )
        q_saved.append(gauge(k.q[clone_indices]))
        r_saved.append(gauge(k.r[clone_indices]))
        uq_saved.append(gauge(k.unary_q[selected]))
        ur_saved.append(gauge(k.unary_r[selected]))
        if t >= 1699:
            tail_q.append(gauge(k.q))
            tail_r.append(gauge(k.r))
            tail_uq.append(gauge(k.unary_q))
            tail_ur.append(gauge(k.unary_r))
        if (t + 1) % 250 == 0:
            print(
                family,
                t + 1,
                "native mismatches",
                len(report["assignment_mismatch_steps"]),
                "gauge error",
                report["max_gauge_message_error"],
                flush=True,
            )
    dump(OUTPUT / f"{family}_native_validation.json", report)
    if report["assignment_mismatch_steps"] or report["max_gauge_message_error"] > 1e-6:
        raise RuntimeError(f"native parity failed: {report}")
    np.savez_compressed(
        OUTPUT / f"{family}_messages.npz",
        incident=incident,
        selected=selected,
        q=q_saved,
        r=r_saved,
        unary_q=uq_saved,
        unary_r=ur_saved,
        tail_q=tail_q,
        tail_r=tail_r,
        tail_unary_q=tail_uq,
        tail_unary_r=tail_ur,
    )


def analyze(family: str) -> dict:
    """Join snapshot verdicts, component structure, and original objective costs."""
    p = load_problem(family)
    s = np.load(OUTPUT / f"{family}_scout.npz")
    with np.load(OUTPUT / f"{family}_messages.npz") as saved:
        trace = {key: saved[key] for key in saved.files}
    x, b = s["assignments"], s["beliefs"]
    snapshots = [
        EngineSnapshot(
            step=t + 1,
            lambda_=0.0,
            dom={},
            N_var={},
            N_fac={},
            Q={},
            R={},
            beliefs={name: b[t, i] for i, name in enumerate(p.variable_names)},
            assignments={name: int(x[t, i]) for i, name in enumerate(p.variable_names)},
        )
        for t in range(2000)
    ]
    analyzer = VariableDynamicsAnalyzer(
        snapshots, domain={n: p.d for n in p.variable_names}
    )
    rows = analyzer.verdicts()
    graph = nx.Graph()
    graph.add_nodes_from(p.variable_names)
    graph.add_edges_from((p.variable_names[u], p.variable_names[v]) for u, v in p.edges)
    unsettled = {r["variable"] for r in rows if r["unsettled"]}
    topology = component_profile(graph, unsettled)
    positions = nx.spring_layout(graph, seed=5000)
    for i, row in enumerate(rows):
        row.update(topology["nodes"][row["variable"]])
        row["split_factor_degree"] = 2 * (row["degree"] + 1)
        row["phase_A"] = int(x[1998, i])
        row["phase_B"] = int(x[1999, i])
        row["unsettled_10000"] = bool(np.any(np.diff(x[-300:, i])))
        row["belief_lag2_residual"] = float(
            np.abs(b[1702:2000, i] - b[1700:1998, i]).max()
        )
        row["position"] = positions[row["variable"]].tolist()
    factors = []
    local_cost = np.broadcast_to(p.unary, (300, p.n, p.d)).copy()
    phase_tables = []
    for e, ((u, v), name, table) in enumerate(zip(p.edges, p.factor_names, p.costs)):
        local_cost[:, u] += table[:, x[1699:1999, v]].T
        local_cost[:, v] += table[x[1699:1999, u], :]
        states = np.array([[x[1998, u], x[1998, v]], [x[1999, u], x[1999, v]]])
        categories = [p.variable_names[k] in unsettled for k in (u, v)]
        factor = dict(
            factor=name,
            u=p.variable_names[u],
            v=p.variable_names[v],
            edge_index=e,
            group="UU" if all(categories) else "US" if any(categories) else "SS",
            **cost_table_profile(table),
            phase_A_cost=float(table[tuple(states[0])]),
            phase_B_cost=float(table[tuple(states[1])]),
        )
        factors.append(factor)
        phase_tables.append(
            dict(
                factor=name,
                axes=[p.variable_names[u], p.variable_names[v]],
                table=table.tolist(),
                phase_states=states.tolist(),
            )
        )
    for i, row in enumerate(rows):
        phase_tables.append(
            dict(
                factor=p.unary_names[i],
                axes=[p.variable_names[i]],
                table=p.unary[i].tolist(),
            )
        )
        local = local_cost[:, i]
        chosen = local[np.arange(300), x[1700:2000, i]]
        regret = chosen - local.min(axis=1)
        row["next_verdict_best_response_fraction"] = float(np.mean(regret <= 1e-6))
        row["next_verdict_best_response_regret_max"] = float(regret.max())
        connected = [f for f in factors if row["variable"] in (f["u"], f["v"])]
        row["incident_interaction_rms"] = float(
            np.mean([f["interaction_rms"] for f in connected])
        )
        row["incident_table_std"] = float(np.mean([f["std"] for f in connected]))
        pairs = [e for e, ends in enumerate(p.edges) if i in ends]
        q = np.stack(
            [
                trace["tail_q"][:, 2 * e : 2 * e + 2, list(p.edges[e]).index(i)]
                for e in pairs
            ],
            axis=1,
        )
        r = np.stack(
            [
                trace["tail_r"][:, 2 * e : 2 * e + 2, list(p.edges[e]).index(i)]
                for e in pairs
            ],
            axis=1,
        )
        row["q_tail_span"] = float(np.ptp(q[1:], axis=0).max())
        row["r_tail_span"] = float(np.ptp(r[1:], axis=0).max())
    details = {}
    incidence = list(trace["incident"])
    for j, i in enumerate(SELECTED[family]):
        name = p.variable_names[i]
        rcols, qcols, labels, partners, edge_ids = [], [], [], [], []
        for e, ends in enumerate(p.edges):
            if i not in ends:
                continue
            axis = list(ends).index(i)
            pos = incidence.index(e)
            rcols.append(trace["r"][:, 2 * pos : 2 * pos + 2, axis])
            qcols.append(trace["q"][:, 2 * pos : 2 * pos + 2, axis])
            labels.append(p.factor_names[e])
            partners.append(p.variable_names[ends[1 - axis]])
            edge_ids.append(e)
        r = np.stack(rcols, axis=1)
        q = np.stack(qcols, axis=1)
        r = np.concatenate((r, trace["unary_r"][:, j : j + 1]), axis=1)
        q = np.concatenate((q, trace["unary_q"][:, j : j + 1]), axis=1)
        labels.append(p.unary_names[i])
        partners.append("unary")
        edge_ids.append(-1)
        reconstruction = float(np.abs(r.sum(axis=(1, 2)) - b[:2000, i]).max())
        cavity = b[:2000, i, None, None, :] - r
        external = b[:2000, i, None, :] - r.sum(axis=2)
        alignment = float(np.abs(q[1:] - cavity[:-1]).max())
        tail_margin = float(
            np.diff(np.sort(b[1700:2000, i], axis=1)[:, :2], axis=1).min()
        )
        if max(reconstruction, alignment) > tail_margin / 4:
            raise ValueError(
                f"roundoff too large relative to verdict margin for {name}: {reconstruction}, {alignment}"
            )
        values = [int(x[1998, i]), int(x[1999, i])]
        if values[0] == values[1]:
            competitors = b[1700:2000, i].mean(axis=0).argsort()
            values[1] = int(next(v for v in competitors if v != values[0]))
        a, z = values
        gaps = r[..., z] - r[..., a]
        contributions = gaps.sum(axis=2)
        factor_rows = []
        for f, (factor, partner, edge) in enumerate(zip(labels, partners, edge_ids)):
            swing = float(contributions[-1, f] - contributions[-2, f])
            factor_rows.append(
                dict(
                    variable=name,
                    factor=factor,
                    neighbor=partner,
                    edge_index=int(edge),
                    contribution_A=float(contributions[-2, f]),
                    contribution_B=float(contributions[-1, f]),
                    swing=swing,
                    clone_A_cavity_winner=int(cavity[-2, f, 0].argmin()),
                    clone_B_cavity_winner=int(cavity[-1, f, 0].argmin()),
                    external_A_winner=int(external[-2, f].argmin()),
                    external_B_winner=int(external[-1, f].argmin()),
                    sibling_changes_winner_fraction=float(
                        np.mean(
                            cavity[-300:, f, 0].argmin(1)
                            != external[-300:, f].argmin(1)
                        )
                    ),
                )
            )
        details[name] = dict(
            variable=name,
            labels=labels,
            partners=partners,
            edge_indices=edge_ids,
            values=values,
            reconstruction_error=reconstruction,
            sent_q_alignment_error=alignment,
            min_tail_margin=tail_margin,
            clone_difference=float(np.abs(r[:, :, 0] - r[:, :, 1]).max()),
            factors=factor_rows,
        )
        np.savez_compressed(
            OUTPUT / f"{family}_{name}.npz",
            belief=b[:2000, i],
            assignment=x[:2000, i],
            r=r,
            q=q,
            cavity=cavity,
            external_cavity=external,
            reconstruction_residual=b[:2000, i] - r.sum(axis=(1, 2)),
        )
        write_csv(OUTPUT / f"{family}_{name}_factor_contributions.csv", factor_rows)
    summary = dict(
        family=family,
        n=p.n,
        d=p.d,
        edges=len(p.edges),
        density=nx.density(graph),
        unsettled=sorted(unsettled, key=lambda n: int(n[1:])),
        settled=[r["variable"] for r in rows if not r["unsettled"]],
        components=topology["components"],
        nodes=rows,
        factors=factors,
        selected=details,
        phase_costs=s["costs"][1998:2000].tolist(),
        same_verdicts_2000_10000=bool(np.array_equal(x[1700:2000], x[-300:])),
        native_validation=json.loads(
            (OUTPUT / f"{family}_native_validation.json").read_text()
        ),
    )
    dump(OUTPUT / f"{family}_analysis.json", summary)
    dump(OUTPUT / f"{family}_cost_tables.json", phase_tables)
    write_csv(
        OUTPUT / f"{family}_variables.csv",
        [{k: v for k, v in row.items() if k != "position"} for row in rows],
    )
    write_csv(OUTPUT / f"{family}_factors.csv", factors)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["scout", "trace", "analyze", "all"])
    parser.add_argument("--family", choices=list(SELECTED))
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    for family in [args.family] if args.family else SELECTED:
        for name, function in (
            ("scout", scout),
            ("trace", native_trace),
            ("analyze", analyze),
        ):
            if args.stage in (name, "all"):
                function(family)


if __name__ == "__main__":
    main()
