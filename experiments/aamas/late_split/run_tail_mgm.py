"""Merge every observed post-split tail value, starting from every tail state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.aaai.code.problems import capture_original
from .core import load_input, tail_kind, verify_costs, write_json
from .mgm import mgm_menu_search, observed_menus
from .run import sha


def run_tail_mgm(run_dir: Path) -> None:
    """Run MGM only on saved original inputs and the configured final tail window."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    window = manifest["config"]["tail_steps"]
    out = run_dir / "mgm_tail_values"
    out.mkdir(exist_ok=True)
    source = out / "source"
    source.mkdir(exist_ok=True)
    for name in ["mgm.py", "run_tail_mgm.py"]:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    lines = [
        "# MGM using all observed tail values",
        "",
        f"Menus include every value each variable uses in the final {window} post-split updates.",
        "Run MGM from every distinct complete assignment in that window; keep the best result.",
        "For nonperiodic tails these are observed-tail menus, not a claim of a complete cycle.",
        "Costs are evaluated on original tables. No additional BP or B&B was run.",
        "",
        "| Seed | Mode | Tail | Starts | Maximum menu size | MGM | Best BP seen | Standard damped split |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed in manifest["seeds"]:
        case = run_dir / f"random_dense_{seed}"
        original = load_input(case / "input.npz")
        names, axes, tables = capture_original(original)
        with np.load(case / "references.npz", allow_pickle=False) as f:
            baseline = float(f["DMS_split_0.5"][-1])
        for mode in ["fixed", "best"]:
            saved = json.loads((case / f"{mode}_result.json").read_text())
            for name, checksum in saved["files_sha256"].items():
                if sha(case / name) != checksum:
                    raise ValueError(f"evidence changed: {case / name}")
            with np.load(case / f"{mode}_trace.npz", allow_pickle=False) as f:
                trace = dict(f)
            verify_costs(original, trace)
            menus, indices = observed_menus(trace, window)
            starts = []
            order = list(trace["variable_names"])
            for index in indices:
                assignment = dict(zip(order, map(int, trace["assignments"][index])))
                result = mgm_menu_search(
                    assignment,
                    menus,
                    names,
                    axes,
                    tables,
                    max_rounds=manifest["config"]["mgm_rounds"],
                )
                if abs(result["costs"][0] - trace["costs"][index]) > 1e-8:
                    raise RuntimeError("MGM initial cost differs from its BP state")
                if any(result["assignment"][v] not in menus[v] for v in names):
                    raise RuntimeError("MGM assignment escaped the observed menus")
                result["source_trace_index"] = index
                result["source_iteration"] = int(trace["iterations"][index])
                starts.append(result)
            best = min(starts, key=lambda r: r["cost"])
            result = {
                "seed": seed,
                "mode": mode,
                "window": window,
                "menu": "all values observed in the final post-split tail window",
                "menus": menus,
                "tail_kind": tail_kind(trace["assignments"], window),
                "starts": starts,
                "best": best,
                "baseline": baseline,
                "prefix_best": saved["prefix_best_cost"],
                "best_bp_seen": saved["anytime_cost"],
                "input_sha256": saved["input_sha256"],
                "trace_sha256": sha(case / f"{mode}_trace.npz"),
                "mgm_source_sha256": sha(Path(__file__).with_name("mgm.py")),
                "bb_run": False,
            }
            write_json(out / f"seed_{seed}_{mode}_mgm.json", result)
            lines.append(
                f"| {seed} | {mode} | {result['tail_kind']} | {len(starts)} | "
                f"{max(map(len, menus.values()))} | {best['cost']:,.3f} | "
                f"{saved['anytime_cost']:,.3f} | {baseline:,.3f} |"
            )
            print(
                f"MGM seed={seed} {mode}: {len(starts)} starts, cost={best['cost']:.6f}",
                flush=True,
            )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run_tail_mgm(parser.parse_args().run_dir)
