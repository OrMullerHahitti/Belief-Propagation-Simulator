"""Append saved MGM trajectories to the original late-split cost plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.aaai.code.merge import score_assignment  # noqa: E402
from experiments.aaai.code.problems import capture_original  # noqa: E402
from experiments.aaai.code.utils.plot_helpers import remove_frame  # noqa: E402
from .core import load_input, verify_costs, write_json  # noqa: E402
from .run import sha  # noqa: E402


def plot_pipeline(
    run_dir: Path, mgm_dir: str = "mgm_followup", stem: str = "pipeline_mgm"
) -> None:
    """Save separate per-seed BP-to-MGM figures without changing prior plots."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    output = run_dir / "plots"
    output.mkdir(exist_ok=True)
    provenance = {"renderer_sha256": sha(Path(__file__)), "figures": []}
    for seed in manifest["seeds"]:
        case = run_dir / f"random_dense_{seed}"
        original = load_input(case / "input.npz")
        _, axes, tables = capture_original(original)
        with np.load(case / "prefix.npz", allow_pickle=False) as f:
            prefix = dict(f)
        with np.load(case / "references.npz", allow_pickle=False) as f:
            refs = dict(f)
        verify_costs(original, prefix)
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.plot(
            refs["DMS_split_0.5_iterations"] + 1,
            refs["DMS_split_0.5"],
            color="#484848",
            lw=1.3,
            label="Standard split + damping",
        )
        events = {0: "0"}
        endpoints = []
        records = []
        for mode, color, style, label in [
            ("fixed", "#2563A6", "-", "Split at 1,000 + MGM"),
            ("best", "#BE6B21", "--", "Split from best state + MGM"),
        ]:
            result = json.loads((case / f"{mode}_result.json").read_text())
            for name, checksum in result["files_sha256"].items():
                if sha(case / name) != checksum:
                    raise ValueError(f"evidence changed: {case / name}")
            merge_path = run_dir / mgm_dir / f"seed_{seed}_{mode}_mgm.json"
            merge = json.loads(merge_path.read_text())
            if merge["trace_sha256"] != sha(case / f"{mode}_trace.npz"):
                raise ValueError("MGM follows a different BP trajectory")
            if merge["input_sha256"] != result["input_sha256"]:
                raise ValueError("MGM original input differs")
            with np.load(case / f"{mode}_trace.npz", allow_pickle=False) as f:
                trace = dict(f)
            verify_costs(original, trace)
            names = list(trace["variable_names"])
            for index, start in enumerate(merge["starts"]):
                source_index = start.get("source_trace_index", -2 + index)
                endpoint = dict(
                    zip(names, map(int, trace["assignments"][source_index]))
                )
                initial = start.get("initial_assignment")
                if initial is None:
                    initial = merge["branches"][index]
                if endpoint != initial:
                    raise ValueError("MGM start differs from saved endpoint")
                if abs(start["costs"][0] - trace["costs"][source_index]) > 1e-8:
                    raise ValueError("MGM initial cost differs")
                if not np.all(np.diff(start["costs"]) < -1e-9):
                    raise ValueError("non-improving MGM round")
                score = score_assignment(start["assignment"], tables, axes)
                if abs(score - start["cost"]) > 1e-8:
                    raise ValueError("MGM final original-cost mismatch")
                if abs(start["cost"] - start["costs"][-1]) > 1e-8:
                    raise ValueError("MGM history endpoint mismatch")
            winner = min(merge["starts"], key=lambda start: start["cost"])
            if winner != merge["best"]:
                raise ValueError("MGM winner differs from saved result")
            split = result["split_before_iteration"]
            mask = prefix["iterations"] < split
            x = np.concatenate([prefix["iterations"][mask], trace["iterations"]]) + 1
            y = np.concatenate([prefix["costs"][mask], trace["costs"]])
            if not np.all(np.diff(x) == 1):
                raise ValueError("nonconsecutive BP iterations")
            ax.plot(x, y, color=color, ls=style, lw=1.25, label=label)
            boundary = int(x[-1])
            history = np.array(winner["costs"])
            if len(history) != winner["rounds"] + 1:
                raise ValueError("MGM round count differs from history")
            mgm_x = boundary + np.arange(len(history))
            # restarting from the selected saved state is not an MGM move
            if history[0] != y[-1]:
                ax.plot(
                    [boundary, boundary], [y[-1], history[0]], color=color, ls=":", lw=1
                )
            ax.plot(mgm_x, history, color=color, ls=style, lw=1.8, zorder=4)
            ax.scatter(
                mgm_x[-1],
                history[-1],
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=45,
                zorder=5,
            )
            endpoints.append(int(mgm_x[-1]))
            tag = "fixed" if mode == "fixed" else "best"
            for position, event in [
                (split, f"Split ({tag})"),
                (boundary, f"MGM ({tag})"),
            ]:
                if position in events:
                    events[position] += " / " + event
                else:
                    events[position] = f"{position:,}\n{event}"
            records.append(
                {
                    "mode": mode,
                    "split_after_update": split,
                    "mgm_start": boundary,
                    "mgm_end": int(mgm_x[-1]),
                    "mgm_rounds": winner["rounds"],
                    "mgm_cost": winner["cost"],
                    "start_index": merge["starts"].index(winner),
                    "mgm_source_sha256": sha(merge_path),
                }
            )
        maximum = max(endpoints)
        positions = sorted(events)
        ax.set_xticks(positions, [events[p] for p in positions])
        # neighboring event labels remain anchored to the actual event coordinates
        for left, right in zip(positions, positions[1:]):
            if right - left < maximum * 0.13:
                ax.get_xticklabels()[positions.index(left)].set_ha("right")
                ax.get_xticklabels()[positions.index(right)].set_ha("left")
        ax.set_xlim(0, maximum + 0.035 * maximum)
        ax.set_xlabel("Iterations (BP updates, then MGM rounds)", labelpad=10)
        ax.set_ylabel("Cost")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_title(
            f"Random dense — seed {seed} · 50 agents · domain {manifest['domain_size']}",
            loc="left",
            fontsize=14,
            pad=44,
        )
        ax.legend(
            frameon=False,
            fontsize=9.5,
            ncol=3,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            borderaxespad=0,
        )
        ax.grid(axis="y", color="#e7e7e7", lw=0.6)
        remove_frame(ax)
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{stem}_seed_{seed}.{extension}", dpi=180)
        plt.close(fig)
        provenance["figures"].append({"seed": seed, "curves": records})
    write_json(output / f"{stem}_provenance.json", provenance)
    shutil.copyfile(__file__, output / "plot_pipeline.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mgm-dir", default="mgm_followup")
    parser.add_argument("--stem", default="pipeline_mgm")
    args = parser.parse_args()
    plot_pipeline(args.run_dir, args.mgm_dir, args.stem)
