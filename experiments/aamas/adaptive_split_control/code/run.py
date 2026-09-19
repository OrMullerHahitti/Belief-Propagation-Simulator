"""Reproducible small-graph study with separate offline and online learning."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .lab import Action, PairwiseLab, TinyScorer, make_problem, native_parity


FAMILIES = ("random", "frustrated")
HORIZON = 128
BLOCK = 8
GRID = [(w, d) for w in (None, 0.5, 0.8, 0.95) for d in (0, 0.3, 0.5, 0.7, 0.9)]


def fixed(problem, weight, damping):
    lab = PairwiseLab(problem, weight, damping)
    lab.advance(HORIZON)
    return lab


def run_policy(
    problem,
    policy,
    model=None,
    fixed_config=(0.5, 0.9),
    rng_seed=0,
    settle_after=None,
    settle_damping=0.5,
):
    """Run one online trajectory; exact-solver evaluation happens only afterwards."""
    started = time.perf_counter()
    rng = np.random.default_rng(rng_seed)
    if policy == "fixed":
        lab = fixed(problem, *fixed_config)
        return lab, [], time.perf_counter() - started, 0
    lab = PairwiseLab(problem, 0.5, 0.9)
    scorer = copy.deepcopy(model) if model is not None else TinyScorer(rng_seed)
    initial_updates = scorer.updates
    replay_x, replay_y, trace = [], [], []
    for step in range(0, HORIZON, BLOCK):
        if settle_after is not None and step >= settle_after:
            lab.act(Action(-1, 0.5, settle_damping))
            lab.advance(BLOCK)
            trace.append(
                {
                    "step": step,
                    "phase": "settle",
                    "damping": settle_damping,
                    "cost": lab.cost,
                    "weights": lab.weights.tolist(),
                }
            )
            continue
        actions = lab.actions()
        features = lab.features(actions, HORIZON)
        scores = scorer.predict(features)
        if policy == "random":
            index = int(rng.integers(len(actions)))
        elif policy == "rule":
            cycle = all(
                [
                    lab.residual.max() > 1e-6 * problem.scale,
                    lab.residual2.max() < 0.2 * lab.residual.max(),
                ]
            )
            if cycle:
                edge = int(np.argmax(lab.residual))
                target = 0.95 if lab.weights[edge] < 0.9 else 0.5
                action = Action(edge, target, 0.9)
            else:
                action = Action(-1, 0.5, 0.9 if step >= 32 else 0.5)
            index = actions.index(action)
        elif policy == "schedule":
            index = actions.index(Action(-1, 0.5, 0 if step < 16 else 0.9))
        else:
            epsilon = 0.1 if policy in ("online", "scratch", "frozen_explore") else 0
            index = (
                int(rng.integers(len(actions)))
                if rng.random() < epsilon
                else int(np.argmax(scores))
            )
        action = actions[index]
        before = lab.cost
        lab.act(action)
        lab.advance(BLOCK)
        reward = lab.reward(before, BLOCK)
        if policy in ("online", "scratch"):
            replay_x.append(features[index])
            replay_y.append(reward)
            for _ in range(8):
                scorer.update(np.array(replay_x), np.array(replay_y), rate=0.05)
        trace.append(
            {
                "step": step,
                "edge": action.edge,
                "weight": action.weight,
                "damping": action.damping,
                "reward": reward,
                "predicted_reward": float(scores[index]),
                "cost": lab.cost,
                "weights": lab.weights.tolist(),
            }
        )
    return lab, trace, time.perf_counter() - started, scorer.updates - initial_updates


def collect_training(problems):
    rng = np.random.default_rng(20260914)
    xs, ys = [], []
    for i, problem in enumerate(problems):
        lab = PairwiseLab(
            problem, float(rng.choice([0.5, 0.95])), float(rng.choice([0, 0.5, 0.9]))
        )
        for checkpoint in (8, 16, 24, 40, 64, 88):
            lab.advance(checkpoint - lab.t)
            actions = lab.actions()
            features = lab.features(actions, HORIZON)
            for action, feature in zip(actions, features):
                trial = lab.clone()
                trial.act(action)
                trial.advance(BLOCK)
                xs.append(feature)
                ys.append(trial.reward(lab.cost, BLOCK))
            lab.act(actions[int(rng.integers(len(actions)))])
        if (i + 1) % 8 == 0:
            print(f"teacher states: {i + 1}/{len(problems)} graphs", flush=True)
    return np.array(xs), np.array(ys)


def train(x, y, seed):
    model = TinyScorer(seed)
    rng = np.random.default_rng(seed)
    for _ in range(120):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 256):
            end = start + 256
            batch = order[start:end]
            model.update(x[batch], y[batch], rate=0.05)
    return model


def aggregate(rows):
    return {
        "n": len(rows),
        "mean_final_gap": float(np.mean([r["final_gap"] for r in rows])),
        "mean_best_gap": float(np.mean([r["best_gap"] for r in rows])),
        "optimal": sum(r["optimal"] for r in rows),
        "assignment_stable": sum(r["assignment_stable"] for r in rows),
        "message_stable": sum(r["message_stable"] for r in rows),
        "median_seconds": float(np.median([r.get("seconds", 0) for r in rows])),
    }


def select_key(rows):
    summary = aggregate(rows)
    return (
        summary["mean_final_gap"],
        -summary["assignment_stable"],
        -summary["message_stable"],
    )


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mechanism_study(output):
    """Frozen-state pulse interventions, scored on the same original objective."""
    rows, traces = [], {}
    for topology in ("edge", "triangle", "bowtie", "k4"):
        for family in FAMILIES:
            for seed in range(8):
                problem = make_problem(topology, family, seed)
                for weight in (0.5, 0.95):
                    lab = PairwiseLab(problem, weight, 0)
                    lab.advance(24)
                    for treatment in ("sham", "damp", "split", "both"):
                        trial = lab.clone()
                        damp = 0.9 if treatment in ("damp", "both") else 0
                        edge = (
                            int(np.argmax(lab.residual))
                            if treatment in ("split", "both")
                            else -1
                        )
                        target = 0.95 if weight == 0.5 else 0.5
                        trial.act(Action(edge, target, damp))
                        trial.advance(BLOCK)
                        if edge >= 0:
                            trial.act(Action(edge, weight, 0))
                        else:
                            trial.act(Action(-1, 0.5, 0))
                        trial.advance(HORIZON - trial.t)
                        row = {
                            "topology": topology,
                            "family": family,
                            "seed": seed,
                            "initial_split": weight,
                            "treatment": treatment,
                            **trial.metrics(),
                        }
                        rows.append(row)
                        if topology == "bowtie" and seed == 0 and weight == 0.5:
                            traces[f"{family}_{treatment}"] = {
                                "cost": trial.costs,
                                "assignments": np.array(trial.assignments).tolist(),
                                "residual": trial.residuals,
                            }
    save_csv(output / "interventions.csv", rows)
    (output / "intervention_traces.json").write_text(json.dumps(traces, indent=2))
    return rows


def report(output, summary, chosen, model_seed, dataset_size, parity, pairs):
    lines = [
        "# Small adaptive splitting study: results",
        "",
        "Primary outcome: final original-objective cost; stability is reported separately.",
        "Gaps below are 100 × (cost − exact optimum) / sum of original table ranges.",
        "They are **not** percentage improvement relative to the optimum.",
        "",
        f"Validation-selected fixed setting: split={chosen[0]}, old-Q damping={chosen[1]}.",
        f"145-parameter offline scorer seed: {model_seed}; "
        f"teacher action outcomes: {dataset_size}.",
        f"Maximum native Q/R parity discrepancy: {parity:.3g}.",
        "",
        "| Topology | Method | N | Final gap | Best gap | Optimal final | "
        "Stable assignments | Stable messages | Median ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary.items():
        topology, method = key.split("/")
        lines.append(
            f"| {topology} | {method} | {row['n']} | {100 * row['mean_final_gap']:.3f} | "
            f"{100 * row['mean_best_gap']:.3f} | {row['optimal']} | "
            f"{row['assignment_stable']} | {row['message_stable']} | {1000 * row['median_seconds']:.2f} |"
        )
    lines.extend(["", "## Paired final-cost outcomes versus tuned fixed", ""])
    for key, row in pairs.items():
        lines.append(
            f"- {key}: {row['wins']} wins, {row['ties']} ties, {row['losses']} losses; "
            f"mean normalized change {100 * row['mean_delta']:.3f}, "
            f"paired 95% bootstrap interval [{100 * row['ci_low']:.3f}, {100 * row['ci_high']:.3f}]."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "These are small synthetic graphs and finite 128-step trajectories. "
            "A stable 16-step tail is not a convergence proof. "
            "The offline teacher used additional counterfactual computation; deployment timings exclude that training. "
            "Online policies used only chosen-action feedback and were reset independently for every test problem. "
            "All policies have the same BP sweep budget; "
            "split methods process twice as many factors as unsplit methods. "
            "A single global damping rate and one edge intervention per block limit the controller's expressiveness. "
            "No DABP rerun, large-instance improvement, or general convergence theorem is claimed.",
            "",
            "See README.md beside the source for equations, sources, seeds, and controls.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise SystemExit("choose a new output directory to preserve an existing run")
    started = time.perf_counter()
    train_problems = [make_problem("bowtie", f, s) for f in FAMILIES for s in range(24)]
    val_problems = [
        make_problem("bowtie", f, s) for f in FAMILIES for s in range(100, 112)
    ]
    test_problems = [
        make_problem(t, f, s)
        for t in ("bowtie", "k4")
        for f in FAMILIES
        for s in range(1000, 1032)
    ]
    inputs = {}
    for partition, problems in (
        ("train", train_problems),
        ("validation", val_problems),
        ("test", test_problems),
    ):
        for p in problems:
            key = f"{partition}_{p.topology}_{p.family}_{p.seed}"
            inputs[key + "_edges"] = p.edges
            inputs[key + "_costs"] = p.costs
    np.savez_compressed(out / "inputs.npz", **inputs)
    parity = max(
        native_parity(make_problem("bowtie", "random", 13), w, d)
        for w in (None, 0.5, 0.95)
        for d in (0, 0.9)
    )
    print(f"native runtime parity passed: {parity:.3g}", flush=True)
    validation = []
    for weight, damping in GRID:
        rows = [fixed(p, weight, damping).metrics() for p in val_problems]
        validation.append(
            {
                "weight": weight,
                "damping": damping,
                "key": select_key(rows),
                **aggregate(rows),
            }
        )
    winner = min(validation, key=lambda row: row["key"])
    chosen = (winner["weight"], winner["damping"])
    print(f"validation selected fixed configuration {chosen}", flush=True)
    (out / "fixed_validation.json").write_text(json.dumps(validation, indent=2))
    x, y = collect_training(train_problems)
    np.savez_compressed(out / "teacher_data.npz", x=x, y=y)
    candidates = []
    for seed in range(3):
        model = train(x, y, seed)
        rows = [run_policy(p, "frozen", model)[0].metrics() for p in val_problems]
        candidates.append((select_key(rows), seed, model))
        print(
            f"offline seed {seed}: validation final gap {select_key(rows)[0]:.5f}",
            flush=True,
        )
    _, model_seed, model = min(candidates, key=lambda row: row[0])
    model.save(out / "offline_model.npz")
    methods = {
        "unsplit_d09": ("fixed", (None, 0.9)),
        "split05_d09": ("fixed", (0.5, 0.9)),
        "split095_d09": ("fixed", (0.95, 0.9)),
        "tuned_fixed": ("fixed", chosen),
        "random": ("random", chosen),
        "cycle_rule": ("rule", chosen),
        "schedule": ("schedule", chosen),
        "offline_frozen": ("frozen", chosen),
        "offline_frozen_explore": ("frozen_explore", chosen),
        "offline_online": ("online", chosen),
        "online_scratch": ("scratch", chosen),
    }
    test_rows, traces = [], {}
    for i, p in enumerate(test_problems):
        for name, (policy, config) in methods.items():
            active_model = None if policy == "scratch" else model
            lab, trace, seconds, updates = run_policy(
                p, policy, active_model, config, p.seed + 5000
            )
            test_rows.append(
                {
                    "topology": p.topology,
                    "family": p.family,
                    "seed": p.seed,
                    "method": name,
                    "seconds": seconds,
                    "online_updates": updates,
                    **lab.metrics(),
                }
            )
            traces[f"{p.topology}_{p.family}_{p.seed}_{name}"] = {
                "cost": lab.costs,
                "residual": lab.residuals,
                "qr_residual": lab.message_residuals,
                "assignments": np.array(lab.assignments).tolist(),
                "actions": trace,
            }
        if (i + 1) % 16 == 0:
            print(f"test graphs: {i + 1}/{len(test_problems)}", flush=True)
    save_csv(out / "test_results.csv", test_rows)
    (out / "test_traces.json").write_text(json.dumps(traces))
    summary = {}
    pairs = {}
    bootstrap_rng = np.random.default_rng(77)
    for topology in ("bowtie", "k4"):
        base = [
            r
            for r in test_rows
            if r["topology"] == topology and r["method"] == "tuned_fixed"
        ]
        for method in methods:
            rows = [
                r
                for r in test_rows
                if r["topology"] == topology and r["method"] == method
            ]
            summary[f"{topology}/{method}"] = aggregate(rows)
            delta = np.array(
                [r["final_gap"] - b["final_gap"] for r, b in zip(rows, base)]
            )
            boot = bootstrap_rng.choice(delta, (4000, len(delta)), replace=True).mean(
                axis=1
            )
            pairs[f"{topology}/{method}"] = {
                "wins": int(np.sum(delta < -1e-10)),
                "ties": int(np.sum(abs(delta) <= 1e-10)),
                "losses": int(np.sum(delta > 1e-10)),
                "mean_delta": float(delta.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
            }
    mechanism_study(out)
    manifest = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "horizon": HORIZON,
        "block": BLOCK,
        "training_seeds": [0, 23],
        "validation_seeds": [100, 111],
        "test_seeds": [1000, 1031],
        "chosen_fixed": chosen,
        "model_seed": model_seed,
        "parameters": 145,
        "teacher_samples": len(y),
        "native_parity_max": parity,
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "inputs_sha256": hashlib.sha256((out / "inputs.npz").read_bytes()).hexdigest(),
        "summary": summary,
        "pairs": pairs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report(out, summary, chosen, model_seed, len(y), parity, pairs)
    print(
        f"completed in {manifest['wall_seconds']:.1f}s; {out / 'RESULTS.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
