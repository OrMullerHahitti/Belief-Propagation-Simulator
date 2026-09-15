"""Audit saved outcomes, learning decisions, tiny optima, and comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import native_replay, study  # noqa: E402


def audit_manifests(root: Path) -> dict:
    checked, mismatches = 0, []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest.get("artifact_sha256", {}).items():
            path = manifest_path.parent / name
            checked += 1
            if not path.is_file() or study.sha(path) != expected:
                mismatches.append(str(path))
    if mismatches:
        raise AssertionError(f"saved artifact hashes changed: {mismatches}")
    return {"checked_artifacts": checked, "mismatches": mismatches}


def learning_audit(stage: Path) -> list[dict]:
    result = []
    for family in ("k4_d10", "random_sparse"):
        for method, control in (
            ("online", "frozen_explore"),
            ("scratch_online", "scratch_frozen_explore"),
        ):
            arms_changed, decisions, targets, matched_draws = 0, 0, 0, True
            for seed in range(19000, 19016):
                prefix = f"{family}_{seed}_"
                path = stage / "events"
                a = json.loads((path / f"{prefix}{method}.json").read_text())
                b = json.loads((path / f"{prefix}{control}.json").read_text())
                for da, db in zip(a["decisions"], b["decisions"]):
                    decisions += 1
                    arms_changed += da["arm"] != db["arm"]
                    targets += da.get("online_update", False)
                    matched_draws &= da["exploration_draw"] == db["exploration_draw"]
            result.append(
                dict(
                    family=family,
                    method=method,
                    control=control,
                    decisions=decisions,
                    arm_differences=arms_changed,
                    recorded_online_updates=targets,
                    matched_exploration_draws=matched_draws,
                )
            )
    return result


def amplitude_audit(stage: Path) -> list[dict]:
    result = []
    for family in ("k4_d10", "random_sparse", "random_dense"):
        for method in ("weight_0.51", "weight_0.65", "weight_0.8", "pulse"):
            fractions = []
            for seed in range(18000, 18008):
                path = stage / "events" / f"{family}_{seed}_{method}.json"
                events = json.loads(path.read_text())["events"]
                fractions.append(
                    next(e["predicted_row_change"] for e in events if e["step"] == 64)
                )
            result.append(
                dict(
                    family=family,
                    method=method,
                    changed_row_inputs=sum(v > 0 for v in fractions),
                    maximum_changed_fraction=max(fractions),
                )
            )
    return result


def endpoint_audit(stage: Path) -> list[dict]:
    with (stage / "metrics.csv").open() as stream:
        records = list(csv.DictReader(stream))
    result = []
    for family in ("k4_d10", "random_sparse"):
        methods = sorted({r["method"] for r in records})
        for method in methods:
            by_seed = {}
            for r in records:
                if r["family"] == family and r["method"] == method:
                    by_seed.setdefault(int(r["seed"]), {})[int(r["horizon"])] = r
            changed, recovered, worsened = [], [], []
            for seed, pair in by_seed.items():
                early, late = pair[2000], pair[10000]
                difference = float(early["terminal_cost"]) - float(
                    late["terminal_cost"]
                )
                if abs(difference) > 1e-7:
                    changed.append(seed)
                if early["strict_stable"] != late["strict_stable"]:
                    (recovered if late["strict_stable"] == "True" else worsened).append(
                        seed
                    )
            result.append(
                dict(
                    family=family,
                    method=method,
                    changed_cost_seeds=changed,
                    recovered_stability_seeds=recovered,
                    lost_stability_seeds=worsened,
                )
            )
    return result


def tiny_optima(stage: Path) -> list[dict]:
    runtime = native_replay.load_runtime(stage)
    result = []
    for seed in range(19000, 19016):
        p = native_replay.load_problem(
            runtime, stage / "inputs" / f"k4_d10_{seed}.npz", "k4_d10", seed
        )
        optimum, _ = p.exact()
        for method in ("baseline", "pulse", "state_plateau", "scratch_online"):
            with np.load(
                stage / "trajectories" / f"k4_d10_{seed}_{method}.npz"
            ) as data:
                final = float(data["costs"][-1])
            result.append(
                dict(seed=seed, method=method, optimum=optimum, gap=final - optimum)
            )
    return result


def plot_confirmation(stage: Path, out: Path, horizon: int = 10000) -> None:
    paired = json.loads((stage / "paired.json").read_text())
    summary = json.loads((stage / "summary.json").read_text())
    names = {
        "pulse": "Fixed pulse",
        "state_plateau": "State plateau",
        "scratch_online": "Online from scratch",
        "scratch_frozen_explore": "Scratch frozen + exploration",
        "online": "Offline-trained online",
        "frozen": "Offline frozen",
        "pulse_no_damping_during": "Pulse, damping off during",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, family, title in zip(
        axes, ("k4_d10", "random_sparse"), ("K4, domain 10", "50-variable sparse")
    ):
        rows = {
            r["method"]: r
            for r in summary
            if r["family"] == family and r["horizon"] == horizon
        }
        scale = rows["baseline"]["mean_cost"] / 100
        comparison_key = (family, horizon, "baseline")
        estimates = {
            r["method"]: r
            for r in paired
            if (r["family"], r["horizon"], r["control"]) == comparison_key
        }
        for y, name in enumerate(names):
            r = estimates[name]
            mean, limits = r["mean_change"] / scale, np.array(r["ci95"]) / scale
            ax.plot(limits, [y, y], color="black", lw=1.4)
            ax.plot(mean, y, "o", color="#126a72", ms=5)
            ax.text(
                0.98,
                y,
                f"{rows[name]['strict_stable']}/16",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
            )
        ax.axvline(0, color="0.6", lw=1, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Final cost change vs .5 split / .9 damping (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
        low, high = ax.get_xlim()
        ax.set_xlim(low, high + (high - low) * 0.22)
    axes[0].set_yticks(range(len(names)), names.values())
    axes[0].invert_yaxis()
    fig.suptitle(
        f"Fresh inputs, {horizon:,} updates: cost and strict finite-tail stability"
    )
    fig.text(
        0.5,
        0.015,
        "Bars: marginal paired 95% bootstrap intervals. Right labels: stable inputs. Lower cost is better.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for extension in ("png", "pdf"):
        suffix = "" if horizon == 10000 else f"_{horizon}"
        fig.savefig(out / f"confirmation{suffix}.{extension}", dpi=180)
    plt.close(fig)


def plot_mechanism(root: Path, out: Path) -> None:
    stage = root / "development"
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for method, label, color, style in (
        ("baseline", "Equal split / damping .9", "0.35", "--"),
        ("pulse", "Fixed split pulse", "#126a72", "-"),
    ):
        prefix = f"random_sparse_18000_{method}"
        with np.load(stage / "trajectories" / f"{prefix}.npz") as data:
            axes[0].plot(
                np.arange(1, 513),
                data["costs"][:512],
                label=label,
                color=color,
                ls=style,
            )
        rows = json.loads((stage / "observations" / f"{prefix}.json").read_text())
        times = [r["step"] for r in rows]
        features = np.array([r["feature"] for r in rows])
        axes[1].plot(times, features[:, 1], color=color, ls=style)
        axes[2].plot(times, features[:, 2], color=color, ls=style)
    for ax in axes:
        ax.axvspan(64, 256, color="#126a72", alpha=0.09)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Original cost")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Common-row fraction")
    axes[2].set_ylabel("Minimizing-row change\nover preceding 8 updates")
    axes[2].set_xlabel("Completed updates")
    fig.suptitle(
        "Development sparse seed 18000: first input, selected without outcome ranking"
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"mechanism_trace.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    stage = args.root / "confirmation"
    study.write_json(args.out / "artifact_audit.json", audit_manifests(args.root))
    study.write_json(args.out / "learning_audit.json", learning_audit(stage))
    study.write_json(
        args.out / "amplitude_audit.json", amplitude_audit(args.root / "development")
    )
    study.write_json(args.out / "endpoint_audit.json", endpoint_audit(stage))
    study.write_json(args.out / "tiny_optima.json", tiny_optima(stage))
    plot_confirmation(stage, args.out)
    plot_confirmation(stage, args.out, 2000)
    plot_mechanism(args.root, args.out)


if __name__ == "__main__":
    main()
