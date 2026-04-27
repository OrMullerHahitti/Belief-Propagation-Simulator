"""Generate CSV datasets for Figures 5a, 5b, and 8 across multiple cycle sizes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.aij.code.utils.fig58_repro import (
    CASE_CONSISTENT_NO_TAIL,
    CASE_CONSISTENT_WITH_TAIL,
    CASE_INCONSISTENT_NO_TAIL,
    construct_consistent_no_tail_examples,
    construct_inconsistent_no_tail_examples,
    derive_tail_examples,
    run_experiment_examples_cycle,
)


N_EXAMPLES = 50
CYCLE_SIZES = (3, 6, 9, 12)
DOMAIN = 2
LOW = 0
HIGH = 10

MAX_ITER_BY_FIGURE = {
    "figure_5a": 50,
    "figure_5b": 50,
    "figure_8": 70,
}

SEED_START_BY_FIGURE = {
    "figure_5a": 42,
    "figure_5b": 4_042,
    "figure_8": 8_042,
}

SEED_STEP = 1
MAX_ATTEMPTS_PER_SEED = 5_000
CLASSIFY_MAX_ITER = 140
DAMPING_FACTOR = 0.9
NORMALIZE_MESSAGES = False
SUBTRACT_INITIAL = False

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data"  # experiments/aij/data/


FIGURE_CONFIGS = (
    {
        "figure_id": "figure_5a",
        "case_name": CASE_CONSISTENT_NO_TAIL,
    },
    {
        "figure_id": "figure_5b",
        "case_name": CASE_CONSISTENT_WITH_TAIL,
    },
    {
        "figure_id": "figure_8",
        "case_name": CASE_INCONSISTENT_NO_TAIL,
    },
)


_SERIES_RE = re.compile(r"^(x\d+)_v(\d+)$")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _json_dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), separators=(",", ":"))


def _series_meta(series_key: str, classification: Mapping[str, Any]) -> tuple[str, str, int]:
    if series_key.endswith("_route"):
        variable_name = series_key[:-6]
        var_idx = int(variable_name[1:]) - 1
        route_values = classification["route_values_by_var"]
        tracked_value = int(route_values[var_idx][0])
        return variable_name, "route", tracked_value

    match = _SERIES_RE.match(series_key)
    if not match:
        raise ValueError(f"Unsupported series key format: {series_key}")
    variable_name = match.group(1)
    tracked_value = int(match.group(2))
    return variable_name, "value", tracked_value


def build_examples_dataframe(
    *,
    figure_id: str,
    case_name: str,
    cycle_size: int,
    examples: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        classification = example["classification"]
        rows.append(
            {
                "figure_id": figure_id,
                "case_name": case_name,
                "cycle_size": int(cycle_size),
                "example_index": int(example_index),
                "seed": int(example["seed"]),
                "generation_strategy": str(example["generation_strategy"]),
                "period": classification["period"],
                "periodic_start": classification["periodic_start"],
                "consistent": bool(classification["consistent"]),
                "inconsistent": bool(classification["inconsistent"]),
                "no_tail": bool(classification["no_tail"]),
                "unclassified": bool(classification["unclassified"]),
                "route_values_by_var_json": _json_dumps(classification["route_values_by_var"]),
                "periodic_route_json": _json_dumps(classification["periodic_route"]),
                "assignment_trace_json": _json_dumps(classification["assignment_trace"]),
                "cost_tables_json": _json_dumps(example["cost_tables"]),
            }
        )

    return pd.DataFrame(rows)


def build_traces_dataframe(
    *,
    figure_id: str,
    case_name: str,
    cycle_size: int,
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        example_index = int(run["example_index"])
        seed = int(run["seed"])
        classification = run["classification"]
        for series_key in run["key_order"]:
            variable_name, tracked_kind, tracked_value = _series_meta(series_key, classification)
            values = run["records"][series_key]
            for iteration, belief in enumerate(values):
                rows.append(
                    {
                        "figure_id": figure_id,
                        "case_name": case_name,
                        "cycle_size": int(cycle_size),
                        "example_index": example_index,
                        "seed": seed,
                        "iteration": int(iteration),
                        "series_key": series_key,
                        "variable_name": variable_name,
                        "tracked_kind": tracked_kind,
                        "tracked_value": int(tracked_value),
                        "belief": float(belief),
                    }
                )
    return pd.DataFrame(rows)


def _iter_figure_configs() -> Iterable[Mapping[str, str]]:
    for cfg in FIGURE_CONFIGS:
        yield cfg


def _collect_examples(
    *,
    case_name: str,
    cycle_size: int,
    n_examples: int,
    seed_start: int,
) -> List[Dict[str, Any]]:
    """Dispatch to the appropriate example-generation strategy."""
    if case_name == CASE_CONSISTENT_NO_TAIL:
        # 5a: construct directly — F[a,a]=0, rest large
        return construct_consistent_no_tail_examples(
            n_examples,
            cycle_size=cycle_size,
            seed_start=seed_start,
            domain=DOMAIN,
            classify_max_iter=CLASSIFY_MAX_ITER,
        )

    if case_name == CASE_CONSISTENT_WITH_TAIL:
        # 5b: derive from 5a base — generate extra since not all convert to tail
        base_examples = construct_consistent_no_tail_examples(
            n_examples * 4,
            cycle_size=cycle_size,
            seed_start=seed_start,
            domain=DOMAIN,
            classify_max_iter=CLASSIFY_MAX_ITER,
        )
        return derive_tail_examples(base_examples, tail_length=2, classify_max_iter=CLASSIFY_MAX_ITER)[:n_examples]

    if case_name == CASE_INCONSISTENT_NO_TAIL:
        # fig 8: one off-diagonal factor + rest diagonal
        return construct_inconsistent_no_tail_examples(
            n_examples,
            cycle_size=cycle_size,
            domain=DOMAIN,
            seed_start=seed_start,
            classify_max_iter=CLASSIFY_MAX_ITER,
        )

    raise ValueError(f"Unknown case: {case_name}")


def generate_fig58_csv_datasets(
    *,
    output_root: Path = OUTPUT_ROOT,
    n_examples: int = N_EXAMPLES,
    cycle_sizes: Sequence[int] = CYCLE_SIZES,
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for figure_cfg in _iter_figure_configs():
        figure_id = figure_cfg["figure_id"]
        case_name = figure_cfg["case_name"]
        max_iter = MAX_ITER_BY_FIGURE[figure_id]
        seed_base = SEED_START_BY_FIGURE[figure_id]

        print(f"[{figure_id}] case={case_name}")

        for cycle_size in cycle_sizes:
            seed_start = seed_base + cycle_size * 100_000
            print(f"  - cycle={cycle_size}: collecting {n_examples} examples (seed_start={seed_start})")

            examples = _collect_examples(
                case_name=case_name,
                cycle_size=cycle_size,
                n_examples=n_examples,
                seed_start=seed_start,
            )

            runs = run_experiment_examples_cycle(
                examples=examples,
                case_name=case_name,
                max_iter=max_iter,
                damping_factor=DAMPING_FACTOR,
                normalize_messages=NORMALIZE_MESSAGES,
                subtract_initial=SUBTRACT_INITIAL,
            )

            output_root.mkdir(parents=True, exist_ok=True)

            examples_df = build_examples_dataframe(
                figure_id=figure_id,
                case_name=case_name,
                cycle_size=cycle_size,
                examples=examples,
            )
            traces_df = build_traces_dataframe(
                figure_id=figure_id,
                case_name=case_name,
                cycle_size=cycle_size,
                runs=runs,
            )

            # flat layout: aij/data/{figure_id}_cycle_{N}_{kind}.csv
            examples_csv = output_root / f"{figure_id}_cycle_{cycle_size}_examples.csv"
            traces_csv = output_root / f"{figure_id}_cycle_{cycle_size}_traces.csv"
            examples_df.to_csv(examples_csv, index=False)
            traces_df.to_csv(traces_csv, index=False)

            print(
                "    wrote "
                f"examples={len(examples_df)} rows, traces={len(traces_df)} rows -> "
                f"{examples_csv.name}, {traces_csv.name}"
            )


def main() -> None:
    generate_fig58_csv_datasets()


if __name__ == "__main__":
    main()
