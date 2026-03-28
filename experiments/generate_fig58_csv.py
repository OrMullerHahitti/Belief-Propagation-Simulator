"""Generate CSV datasets for Figures 5a, 5b, and 8 across multiple cycle sizes."""

from __future__ import annotations

import json
import re
import sys
from itertools import count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from expiriments.utils.fig58_repro import (
    CASE_CONSISTENT_NO_TAIL,
    CASE_CONSISTENT_WITH_TAIL,
    CASE_INCONSISTENT_NO_TAIL,
    classify_case_cycle,
    find_case_cycle,
    GEN_MOTIF_REPEAT,
    GEN_RANDOM_FULL,
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

OUTPUT_ROOT = Path("expiriments/generated_csv")


FIGURE_CONFIGS = (
    {
        "figure_id": "figure_5a",
        "case_name": CASE_CONSISTENT_NO_TAIL,
        "generation_strategy": GEN_RANDOM_FULL,
    },
    {
        "figure_id": "figure_5b",
        "case_name": CASE_CONSISTENT_WITH_TAIL,
        "generation_strategy": GEN_RANDOM_FULL,
    },
    {
        "figure_id": "figure_8",
        "case_name": CASE_INCONSISTENT_NO_TAIL,
        "generation_strategy": GEN_MOTIF_REPEAT,
    },
)

# Curated motif seeds that satisfy strict Figure-8 classification with max_attempts=1.
CURATED_FIG8_BASE_SEEDS = {
    3: [488, 619, 781, 1079, 1100, 1244, 1472, 1876],
    6: [488],
    9: [488],
    12: [488],
}

FIG8_FALLBACK_MAX_ATTEMPTS_PER_SEED = 1_000


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


def _cost_tables_signature(cost_tables: Sequence[np.ndarray]) -> tuple:
    return tuple(tuple(np.asarray(ct, dtype=float).reshape(-1).tolist()) for ct in cost_tables)


def _swap_binary_tables(cost_tables: Sequence[np.ndarray], swap_mask: Sequence[int]) -> tuple[np.ndarray, ...]:
    cycle_size = len(cost_tables)
    swapped: List[np.ndarray] = []
    for idx, table in enumerate(cost_tables):
        left_var = idx
        right_var = (idx + 1) % cycle_size
        ct = np.array(table, dtype=float)
        if swap_mask[left_var]:
            ct = ct[[1, 0], :]
        if swap_mask[right_var]:
            ct = ct[:, [1, 0]]
        swapped.append(ct)
    return tuple(swapped)


def _format_example(
    *,
    case_name: str,
    seed: int,
    attempt: int,
    cycle_size: int,
    generation_strategy: str,
    cost_tables: Sequence[np.ndarray],
    classification: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "case_name": case_name,
        "seed": int(seed),
        "attempt": int(attempt),
        "cycle_size": int(cycle_size),
        "generation_strategy": generation_strategy,
        "cost_tables": tuple(np.asarray(ct, dtype=float) for ct in cost_tables),
        "classification": dict(classification),
    }


def _collect_examples_figure8(
    *,
    case_name: str,
    cycle_size: int,
    n_examples: int,
    seed_start: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    seen_signatures: set[tuple] = set()
    used_base_seeds: set[int] = set()

    curated = list(CURATED_FIG8_BASE_SEEDS.get(cycle_size, []))
    fallback_seed_iter = count(seed_start, SEED_STEP)
    curated_idx = 0

    while len(examples) < n_examples:
        if curated_idx < len(curated):
            seed = curated[curated_idx]
            max_attempts = 1
            curated_idx += 1
        else:
            seed = next(fallback_seed_iter)
            max_attempts = FIG8_FALLBACK_MAX_ATTEMPTS_PER_SEED

        if seed in used_base_seeds:
            continue
        used_base_seeds.add(seed)

        try:
            base = find_case_cycle(
                case_name=case_name,
                cycle_size=cycle_size,
                seed=seed,
                domain=DOMAIN,
                low=LOW,
                high=HIGH,
                max_attempts=max_attempts,
                classify_max_iter=CLASSIFY_MAX_ITER,
                generation_strategy=GEN_MOTIF_REPEAT,
            )
        except RuntimeError:
            continue

        base_tables = base["cost_tables"]
        # Binary value-label symmetries preserve route structure; still re-validate each variant.
        for mask_int in range(1 << cycle_size):
            mask = tuple((mask_int >> bit) & 1 for bit in range(cycle_size))
            candidate_tables = _swap_binary_tables(base_tables, mask)
            signature = _cost_tables_signature(candidate_tables)
            if signature in seen_signatures:
                continue

            classification = classify_case_cycle(candidate_tables, max_iter=CLASSIFY_MAX_ITER)
            if not classification.get(case_name, False):
                continue

            seen_signatures.add(signature)
            examples.append(
                _format_example(
                    case_name=case_name,
                    seed=int(seed) * 1_000_000 + mask_int,
                    attempt=int(base["attempt"]),
                    cycle_size=cycle_size,
                    generation_strategy=f"{GEN_MOTIF_REPEAT}_swapmask",
                    cost_tables=candidate_tables,
                    classification=classification,
                )
            )
            if len(examples) >= n_examples:
                break

    return examples


def _collect_examples_non_figure8(
    *,
    case_name: str,
    cycle_size: int,
    n_examples: int,
    seed_start: int,
    generation_strategy: str,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    seen_signatures: set[tuple] = set()
    seed = seed_start

    while len(examples) < n_examples:
        result = find_case_cycle(
            case_name=case_name,
            cycle_size=cycle_size,
            seed=seed,
            domain=DOMAIN,
            low=LOW,
            high=HIGH,
            max_attempts=MAX_ATTEMPTS_PER_SEED,
            classify_max_iter=CLASSIFY_MAX_ITER,
            generation_strategy=generation_strategy,
        )
        signature = _cost_tables_signature(result["cost_tables"])
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            examples.append(result)
        seed += SEED_STEP

    return examples


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
        generation_strategy = figure_cfg["generation_strategy"]
        max_iter = MAX_ITER_BY_FIGURE[figure_id]
        seed_base = SEED_START_BY_FIGURE[figure_id]

        print(f"[{figure_id}] case={case_name} strategy={generation_strategy}")

        for cycle_size in cycle_sizes:
            seed_start = seed_base + cycle_size * 100_000
            print(f"  - cycle={cycle_size}: collecting {n_examples} examples (seed_start={seed_start})")
            if figure_id == "figure_8":
                examples = _collect_examples_figure8(
                    case_name=case_name,
                    cycle_size=cycle_size,
                    n_examples=n_examples,
                    seed_start=seed_start,
                )
            else:
                examples = _collect_examples_non_figure8(
                    case_name=case_name,
                    cycle_size=cycle_size,
                    n_examples=n_examples,
                    seed_start=seed_start,
                    generation_strategy=generation_strategy,
                )

            runs = run_experiment_examples_cycle(
                examples=examples,
                case_name=case_name,
                max_iter=max_iter,
                damping_factor=DAMPING_FACTOR,
                normalize_messages=NORMALIZE_MESSAGES,
                subtract_initial=SUBTRACT_INITIAL,
            )

            out_dir = output_root / figure_id / f"cycle_{cycle_size}"
            out_dir.mkdir(parents=True, exist_ok=True)

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

            examples_csv = out_dir / "examples.csv"
            traces_csv = out_dir / "traces.csv"
            examples_df.to_csv(examples_csv, index=False)
            traces_df.to_csv(traces_csv, index=False)

            print(
                "    wrote "
                f"examples={len(examples_df)} rows, traces={len(traces_df)} rows -> {out_dir}"
            )


def main() -> None:
    generate_fig58_csv_datasets()


if __name__ == "__main__":
    main()
