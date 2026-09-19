"""Fresh-seed confirmation of an intervention-then-settle controller refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .lab import Action, PairwiseLab, TinyScorer, make_problem
from .run import FAMILIES, HORIZON, aggregate, run_policy, save_csv, select_key


def schedule(problem, switch, damping):
    lab = PairwiseLab(problem, 0.5, 0)
    lab.advance(switch)
    lab.act(Action(-1, 0.5, damping))
    lab.advance(HORIZON - switch)
    return lab


def load_model(path):
    model = TinyScorer()
    with np.load(path) as weights:
        for name in ("w1", "b1", "w2", "b2"):
            setattr(model, name, weights[name].copy())
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise SystemExit("choose a new output directory")
    model = load_model(args.pilot / "offline_model.npz")
    pilot = json.loads((args.pilot / "manifest.json").read_text())
    chosen_fixed = tuple(pilot["chosen_fixed"])
    validation = [
        make_problem("bowtie", f, s) for f in FAMILIES for s in range(100, 112)
    ]
    tuning = []
    for onset in (0, 8, 16, 32, 64):
        for damping in (0.3, 0.5, 0.7, 0.9):
            rows = [
                run_policy(
                    p,
                    "online",
                    model,
                    rng_seed=p.seed + 5000,
                    settle_after=onset,
                    settle_damping=damping,
                )[0].metrics()
                for p in validation
            ]
            tuning.append(
                {
                    "onset": onset,
                    "damping": damping,
                    "key": select_key(rows),
                    **aggregate(rows),
                }
            )
    selected = min(tuning, key=lambda r: r["key"])
    print(f"selected settling: {selected}", flush=True)
    schedule_tuning = []
    for switch in (0, 4, 8, 16, 32, 64):
        for damping in (0.3, 0.5, 0.7, 0.9):
            rows = [schedule(p, switch, damping).metrics() for p in validation]
            schedule_tuning.append(
                {
                    "switch": switch,
                    "damping": damping,
                    "key": select_key(rows),
                    **aggregate(rows),
                }
            )
    selected_schedule = min(schedule_tuning, key=lambda r: r["key"])
    print(f"selected schedule: {selected_schedule}", flush=True)
    (out / "validation.json").write_text(
        json.dumps({"settle": tuning, "schedule": schedule_tuning}, indent=2)
    )
    rows, traces, inputs = [], {}, {}
    methods = (
        "tuned_fixed",
        "tuned_schedule",
        "cycle_rule",
        "unrestricted_online",
        "settled_online",
        "settled_frozen_explore",
        "settled_random",
        "settled_rule",
    )
    for topology in ("bowtie", "k4"):
        for family in FAMILIES:
            for seed in range(2000, 2064):
                p = make_problem(topology, family, seed)
                key = f"{topology}_{family}_{seed}"
                inputs[key + "_edges"] = p.edges
                inputs[key + "_costs"] = p.costs
                for method in methods:
                    start = time.perf_counter()
                    if method == "tuned_fixed":
                        lab, trace, _, updates = run_policy(
                            p, "fixed", fixed_config=chosen_fixed
                        )
                    elif method == "tuned_schedule":
                        lab = schedule(
                            p, selected_schedule["switch"], selected_schedule["damping"]
                        )
                        trace, updates = [], 0
                    else:
                        policy = {
                            "cycle_rule": "rule",
                            "unrestricted_online": "online",
                            "settled_online": "online",
                            "settled_frozen_explore": "frozen_explore",
                            "settled_random": "random",
                            "settled_rule": "rule",
                        }[method]
                        onset = (
                            selected["onset"] if method.startswith("settled") else None
                        )
                        lab, trace, _, updates = run_policy(
                            p,
                            policy,
                            model,
                            rng_seed=seed + 5000,
                            settle_after=onset,
                            settle_damping=selected["damping"],
                        )
                    elapsed = time.perf_counter() - start
                    rows.append(
                        {
                            "topology": topology,
                            "family": family,
                            "seed": seed,
                            "method": method,
                            "seconds": elapsed,
                            "online_updates": updates,
                            **lab.metrics(),
                        }
                    )
                    traces[key + "_" + method] = {
                        "cost": lab.costs,
                        "residual": lab.residuals,
                        "qr_residual": lab.message_residuals,
                        "assignments": np.array(lab.assignments).tolist(),
                        "actions": trace,
                    }
            print(f"confirmed {topology}/{family}: 64 fresh graphs", flush=True)
    save_csv(out / "test_results.csv", rows)
    np.savez_compressed(out / "inputs.npz", **inputs)
    (out / "test_traces.json").write_text(json.dumps(traces))
    summary = {}
    comparisons = {}
    rng = np.random.default_rng(81)
    for topology in ("bowtie", "k4"):
        for method in methods:
            subset = [
                r for r in rows if r["topology"] == topology and r["method"] == method
            ]
            summary[f"{topology}/{method}"] = aggregate(subset)
        for target, control in (
            (m, "tuned_fixed") for m in methods if m != "tuned_fixed"
        ):
            comparisons[f"{topology}/{target}_vs_{control}"] = paired(
                rows, topology, target, control, rng
            )
        comparisons[f"{topology}/learning_effect"] = paired(
            rows, topology, "settled_online", "settled_frozen_explore", rng
        )
    manifest = {
        "test_seeds": [2000, 2063],
        "selected_settle": selected,
        "selected_schedule": selected_schedule,
        "chosen_fixed": chosen_fixed,
        "pilot_manifest_sha256": hashlib.sha256(
            (args.pilot / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "summary": summary,
        "comparisons": comparisons,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = [
        "# Fresh-seed confirmation",
        "",
        f"Settling starts at {selected['onset']}; damping={selected['damping']}.",
        f"Fixed schedule switches at {selected_schedule['switch']}; damping={selected_schedule['damping']}.",
        "Both choices used validation seeds 100–111 only. Confirmation seeds: 2000–2063 per family.",
        "",
        "| Graph | Method | N | Mean final gap ×100 | Optimal final | Stable assignments | Stable messages |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary.items():
        topology, method = key.split("/")
        lines.append(
            f"| {topology} | {method} | {row['n']} | {100 * row['mean_final_gap']:.3f} | "
            f"{row['optimal']} | {row['assignment_stable']} | {row['message_stable']} |"
        )
    lines.extend(
        [
            "",
            "Negative changes below favor the first method. Intervals are paired bootstrap 95% intervals, "
            "descriptive and uncorrected for multiple comparisons.",
            "",
        ]
    )
    for key, row in comparisons.items():
        lines.append(
            f"- {key}: {row['wins']}/{row['ties']}/{row['losses']} wins/ties/losses; "
            f"change ×100={100 * row['mean_delta']:.3f}; "
            f"interval [{100 * row['ci_low']:.3f}, {100 * row['ci_high']:.3f}]."
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"saved {out / 'RESULTS.md'}", flush=True)


def paired(rows, topology, target, control, rng):
    first = [r for r in rows if r["topology"] == topology and r["method"] == target]
    second = [r for r in rows if r["topology"] == topology and r["method"] == control]
    assert [(r["family"], r["seed"]) for r in first] == [
        (r["family"], r["seed"]) for r in second
    ]
    delta = np.array([a["final_gap"] - b["final_gap"] for a, b in zip(first, second)])
    boot = rng.choice(delta, (4000, len(delta)), replace=True).mean(axis=1)
    return {
        "wins": int(np.sum(delta < -1e-10)),
        "ties": int(np.sum(abs(delta) <= 1e-10)),
        "losses": int(np.sum(delta > 1e-10)),
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


if __name__ == "__main__":
    main()
