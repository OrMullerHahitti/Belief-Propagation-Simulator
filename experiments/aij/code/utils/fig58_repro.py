"""Utilities for reproducing paper Figures 5/8 on cycle graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from propflow import BPEngine, DampingEngine, FactorAgent, FactorGraph, FGBuilder, VariableAgent


Assignment = Tuple[int, ...]
CostTables = Tuple[np.ndarray, np.ndarray, np.ndarray]
CycleCostTables = Tuple[np.ndarray, ...]

CASE_CONSISTENT_NO_TAIL = "consistent_no_tail"
CASE_CONSISTENT_WITH_TAIL = "consistent_with_tail"
CASE_INCONSISTENT_NO_TAIL = "inconsistent_no_tail"

GEN_RANDOM_FULL = "random_full"
GEN_MOTIF_REPEAT = "motif_repeat"


@dataclass(frozen=True)
class RouteInfo:
    """Information about a detected periodic route in assignment traces."""

    period: int
    start_index: int
    route: Tuple[Assignment, ...]


def _require_valid_case_name(case_name: str) -> None:
    valid = {
        CASE_CONSISTENT_NO_TAIL,
        CASE_CONSISTENT_WITH_TAIL,
        CASE_INCONSISTENT_NO_TAIL,
    }
    if case_name not in valid:
        raise ValueError(f"Unknown case_name: {case_name}")


def _as_cycle_cost_tables(cost_tables: Sequence[np.ndarray]) -> CycleCostTables:
    if not cost_tables:
        raise ValueError("At least one cost table is required.")

    normalized = tuple(np.asarray(ct, dtype=float) for ct in cost_tables)
    domain = int(normalized[0].shape[0])
    for idx, ct in enumerate(normalized):
        if ct.ndim != 2:
            raise ValueError(f"Cost table at index {idx} must be 2D, got ndim={ct.ndim}.")
        if ct.shape[0] != ct.shape[1]:
            raise ValueError(f"Cost table at index {idx} must be square, got shape={ct.shape}.")
        if ct.shape[0] != domain:
            raise ValueError(
                "All cost tables must share the same domain size. "
                f"Expected {domain}, got {ct.shape[0]} at index {idx}."
            )
    return normalized


def _build_cycle_graph_from_cost_tables(cost_tables: Sequence[np.ndarray]) -> FactorGraph:
    """Build an N-variable cycle graph from explicit 2D cost tables."""
    tables = _as_cycle_cost_tables(cost_tables)
    cycle_size = len(tables)
    domain = int(tables[0].shape[0])

    variables = [VariableAgent(f"x{i+1}", domain=domain) for i in range(cycle_size)]
    factors: List[FactorAgent] = []
    edges: Dict[FactorAgent, List[VariableAgent]] = {}

    for idx, ct in enumerate(tables):
        left_idx = idx + 1
        right_idx = (idx + 1) % cycle_size + 1
        factor = FactorAgent.create_from_cost_table(f"f{left_idx}{right_idx}", np.array(ct, dtype=float))
        factors.append(factor)
        edges[factor] = [variables[idx], variables[(idx + 1) % cycle_size]]

    return FGBuilder.build_from_edges(variables=variables, factors=factors, edges=edges)


def build_cycle_graph_from_tables(*cost_tables: np.ndarray | Sequence[np.ndarray]) -> FactorGraph:
    """Build a cycle graph from explicit cost tables.

    Backward compatible forms:
    - ``build_cycle_graph_from_tables(ct_f12, ct_f23, ct_f31)``
    - ``build_cycle_graph_from_tables([ct_f12, ct_f23, ct_f31, ...])``
    """
    if len(cost_tables) == 1 and isinstance(cost_tables[0], Sequence):
        return _build_cycle_graph_from_cost_tables(cost_tables[0])  # type: ignore[arg-type]
    return _build_cycle_graph_from_cost_tables(cost_tables)  # type: ignore[arg-type]


def _is_periodic_from(assignments: Sequence[Assignment], start: int, pattern: Sequence[Assignment]) -> bool:
    """Return True if assignments[start:] repeat the given pattern exactly."""
    if not pattern:
        return False
    p = len(pattern)
    base = len(assignments) - p
    for idx in range(start, len(assignments)):
        expected = pattern[(idx - base) % p]
        if assignments[idx] != expected:
            return False
    return True


def _find_route_info(
    assignments: Sequence[Assignment],
    min_period: int = 1,
    max_period: int = 12,
    min_repeats: int = 4,
) -> RouteInfo | None:
    """Detect the stable periodic suffix route and its earliest stable start."""
    n = len(assignments)
    upper = min(max_period, max(1, n // max(1, min_repeats)))

    for period in range(min_period, upper + 1):
        needed = period * min_repeats
        if n < needed:
            continue

        pattern = tuple(assignments[n - period : n])

        stable_suffix = True
        for repeat in range(2, min_repeats + 1):
            left = n - repeat * period
            right = n - (repeat - 1) * period
            if tuple(assignments[left:right]) != pattern:
                stable_suffix = False
                break
        if not stable_suffix:
            continue

        for start in range(0, n - needed + 1):
            if _is_periodic_from(assignments, start, pattern):
                route = tuple(assignments[start : start + period])
                return RouteInfo(period=period, start_index=start, route=route)

    return None


def _unclassified_result(assignments: Sequence[Assignment]) -> Dict[str, Any]:
    return {
        "unclassified": True,
        "assignment_trace": tuple(assignments),
        "period": None,
        "periodic_start": None,
        "periodic_route": tuple(),
        "route_values_by_var": tuple(),
        "no_tail": False,
        "consistent": False,
        "inconsistent": False,
        CASE_CONSISTENT_NO_TAIL: False,
        CASE_CONSISTENT_WITH_TAIL: False,
        CASE_INCONSISTENT_NO_TAIL: False,
    }


def classify_case_cycle(
    cost_tables: Sequence[np.ndarray],
    *,
    max_iter: int = 120,
    min_period: int = 1,
    max_period: int | None = None,
    min_repeats: int = 4,
) -> Dict[str, Any]:
    """Classify cycle-graph behavior from undamped Min-Sum assignment traces."""
    tables = _as_cycle_cost_tables(cost_tables)
    cycle_size = len(tables)
    domain = int(tables[0].shape[0])
    if max_period is None:
        max_period = max(12, cycle_size * domain)

    fg = _build_cycle_graph_from_cost_tables(tables)
    engine = BPEngine(factor_graph=fg, normalize_messages=False)

    assignments: List[Assignment] = []
    for i in range(max_iter):
        engine.step(i)
        assignments.append(tuple(int(engine.assignments[f"x{j+1}"]) for j in range(cycle_size)))

    info = _find_route_info(
        assignments,
        min_period=min_period,
        max_period=max_period,
        min_repeats=min_repeats,
    )
    if info is None:
        return _unclassified_result(assignments)

    route_values_by_var = tuple(
        tuple(sorted({assignment[var_idx] for assignment in info.route}))
        for var_idx in range(cycle_size)
    )
    consistent = all(len(values) == 1 for values in route_values_by_var)
    inconsistent = not consistent
    no_tail = info.start_index == 0

    all_domain_values = tuple(range(domain))
    inconsistent_all_domain_values = all(
        tuple(values) == all_domain_values for values in route_values_by_var
    )

    return {
        "unclassified": False,
        "assignment_trace": tuple(assignments),
        "period": info.period,
        "periodic_start": info.start_index,
        "periodic_route": info.route,
        "route_values_by_var": route_values_by_var,
        "no_tail": no_tail,
        "consistent": consistent,
        "inconsistent": inconsistent,
        CASE_CONSISTENT_NO_TAIL: bool(consistent and no_tail),
        CASE_CONSISTENT_WITH_TAIL: bool(consistent and not no_tail),
        CASE_INCONSISTENT_NO_TAIL: bool(inconsistent and no_tail and inconsistent_all_domain_values),
    }


def classify_case(
    ct_f12: np.ndarray,
    ct_f23: np.ndarray,
    ct_f31: np.ndarray,
    *,
    max_iter: int = 120,
    min_period: int = 1,
    max_period: int = 12,
    min_repeats: int = 4,
) -> Dict[str, Any]:
    """Backward-compatible classifier for 3-variable cycles."""
    return classify_case_cycle(
        [ct_f12, ct_f23, ct_f31],
        max_iter=max_iter,
        min_period=min_period,
        max_period=max_period,
        min_repeats=min_repeats,
    )


def route_assignment_from_classification(classification: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return the single assignment tuple from a consistent route classification."""
    values = classification["route_values_by_var"]
    if not classification["consistent"]:
        raise ValueError("Classification is not consistent.")
    return tuple(int(v[0]) for v in values)


def _generate_candidate_cost_tables(
    rng: np.random.RandomState,
    *,
    cycle_size: int,
    domain: int,
    low: int,
    high: int,
    generation_strategy: str,
) -> CycleCostTables:
    if generation_strategy == GEN_RANDOM_FULL:
        return tuple(
            rng.randint(low, high, size=(domain, domain)).astype(float) for _ in range(cycle_size)
        )

    if generation_strategy == GEN_MOTIF_REPEAT:
        motif = [rng.randint(low, high, size=(domain, domain)).astype(float) for _ in range(3)]
        return tuple(np.array(motif[idx % 3], dtype=float) for idx in range(cycle_size))

    raise ValueError(f"Unknown generation strategy: {generation_strategy}")


def _cost_tables_signature(cost_tables: Sequence[np.ndarray]) -> tuple:
    return tuple(tuple(np.asarray(ct, dtype=float).reshape(-1).tolist()) for ct in cost_tables)


def _format_example_result(
    *,
    case_name: str,
    seed: int,
    attempt: int,
    cycle_size: int,
    generation_strategy: str,
    cost_tables: Sequence[np.ndarray],
    classification: Mapping[str, Any],
) -> Dict[str, Any]:
    tables = tuple(np.asarray(ct, dtype=float) for ct in cost_tables)
    result: Dict[str, Any] = {
        "case_name": case_name,
        "seed": int(seed),
        "attempt": int(attempt),
        "cycle_size": int(cycle_size),
        "generation_strategy": generation_strategy,
        "cost_tables": tables,
        "classification": dict(classification),
    }
    if len(tables) == 3:
        result["ct_f12"] = tables[0]
        result["ct_f23"] = tables[1]
        result["ct_f31"] = tables[2]
    return result


def find_case_cycle(
    case_name: str,
    *,
    cycle_size: int,
    seed: int = 42,
    domain: int = 2,
    low: int = 0,
    high: int = 10,
    max_attempts: int = 50_000,
    classify_max_iter: int = 120,
    generation_strategy: str = GEN_RANDOM_FULL,
) -> Dict[str, Any]:
    """Find deterministic random cost tables for an N-variable cycle case."""
    _require_valid_case_name(case_name)

    if cycle_size < 3:
        raise ValueError(f"cycle_size must be >= 3, got {cycle_size}.")
    if domain < 2:
        raise ValueError(f"domain must be >= 2, got {domain}.")

    rng = np.random.RandomState(seed)
    for attempt in range(1, max_attempts + 1):
        tables = _generate_candidate_cost_tables(
            rng,
            cycle_size=cycle_size,
            domain=domain,
            low=low,
            high=high,
            generation_strategy=generation_strategy,
        )
        classification = classify_case_cycle(tables, max_iter=classify_max_iter)
        if classification.get(case_name, False):
            return _format_example_result(
                case_name=case_name,
                seed=seed,
                attempt=attempt,
                cycle_size=cycle_size,
                generation_strategy=generation_strategy,
                cost_tables=tables,
                classification=classification,
            )

    raise RuntimeError(
        f"No valid graph found for case='{case_name}' after {max_attempts} attempts "
        f"(seed={seed}, cycle_size={cycle_size}, domain={domain}, range=[{low},{high}), "
        f"strategy='{generation_strategy}')."
    )


def find_case(
    case_name: str,
    *,
    seed: int = 42,
    domain: int = 2,
    low: int = 0,
    high: int = 10,
    max_attempts: int = 50_000,
    classify_max_iter: int = 120,
) -> Dict[str, Any]:
    """Backward-compatible single-cycle finder for 3-variable cycles."""
    return find_case_cycle(
        case_name=case_name,
        cycle_size=3,
        seed=seed,
        domain=domain,
        low=low,
        high=high,
        max_attempts=max_attempts,
        classify_max_iter=classify_max_iter,
        generation_strategy=GEN_RANDOM_FULL,
    )


def find_cases_cycle(
    case_name: str,
    n_examples: int,
    *,
    cycle_size: int,
    seed_start: int = 42,
    seed_step: int = 1,
    max_attempts_per_seed: int = 5000,
    domain: int = 2,
    low: int = 0,
    high: int = 10,
    classify_max_iter: int = 120,
    generation_strategy: str = GEN_RANDOM_FULL,
) -> List[Dict[str, Any]]:
    """Deterministically collect the first N unique valid examples for a cycle case."""
    _require_valid_case_name(case_name)
    if n_examples <= 0:
        return []

    examples: List[Dict[str, Any]] = []
    seen_signatures: set[tuple] = set()
    seed = seed_start

    while len(examples) < n_examples:
        result = find_case_cycle(
            case_name=case_name,
            cycle_size=cycle_size,
            seed=seed,
            domain=domain,
            low=low,
            high=high,
            max_attempts=max_attempts_per_seed,
            classify_max_iter=classify_max_iter,
            generation_strategy=generation_strategy,
        )

        signature = _cost_tables_signature(result["cost_tables"])
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            examples.append(result)

        seed += seed_step

    return examples


def find_cases(
    case_name: str,
    n_examples: int,
    *,
    seed_start: int = 42,
    seed_step: int = 1,
    max_attempts_per_seed: int = 5000,
    domain: int = 2,
    low: int = 0,
    high: int = 10,
    classify_max_iter: int = 120,
) -> List[Dict[str, Any]]:
    """Backward-compatible batch finder for 3-variable cycles."""
    return find_cases_cycle(
        case_name=case_name,
        n_examples=n_examples,
        cycle_size=3,
        seed_start=seed_start,
        seed_step=seed_step,
        max_attempts_per_seed=max_attempts_per_seed,
        domain=domain,
        low=low,
        high=high,
        classify_max_iter=classify_max_iter,
        generation_strategy=GEN_RANDOM_FULL,
    )


def derive_tail_examples(
    base_examples: List[Dict[str, Any]],
    *,
    tail_length: int = 2,
    classify_max_iter: int = 120,
) -> List[Dict[str, Any]]:
    """Derive consistent-with-tail examples from consistent-no-tail bases.

    Modifies cost table entries in a chain pattern to create a detour
    of the specified length before the route takes over.

    chain[0]: F_{0,1}[route_0, alt] = 0  (route-left, alt-right is cheap)
    chain[k]: F_{k,k+1}[alt, alt] = 0    (alt-left, alt-right is cheap)
    """
    derived: List[Dict[str, Any]] = []
    for base in base_examples:
        tables_orig = _extract_cost_tables(base)
        cycle_size = len(tables_orig)
        domain = int(tables_orig[0].shape[0])
        classification = base["classification"]
        route = route_assignment_from_classification(classification)

        success = False
        for alt_offset in range(1, domain):
            alt_val = (route[1] + alt_offset) % domain
            modified = [np.array(ct, dtype=float) for ct in tables_orig]

            # chain[0]: make (x0=route, x1=alt) cheap
            modified[0][route[0], alt_val] = 0.0
            # chain[k]: make (xk=alt, x{k+1}=alt) cheap
            for k in range(1, min(tail_length, cycle_size)):
                modified[k][alt_val, alt_val] = 0.0

            new_cls = classify_case_cycle(modified, max_iter=classify_max_iter)
            if new_cls.get(CASE_CONSISTENT_WITH_TAIL, False):
                example = _format_example_result(
                    case_name=CASE_CONSISTENT_WITH_TAIL,
                    seed=base["seed"],
                    attempt=base["attempt"],
                    cycle_size=cycle_size,
                    generation_strategy="derived_tail",
                    cost_tables=modified,
                    classification=new_cls,
                )
                example["base_seed"] = int(base["seed"])
                derived.append(example)
                success = True
                break

        if not success:
            print(f"  [derive_tail] skipping seed={base['seed']}: "
                  f"no alt_val produced a valid tail")

    return derived


def construct_consistent_no_tail_examples(
    n_examples: int,
    *,
    cycle_size: int,
    domain: int = 2,
    low: int = 0,
    high: int = 10,
    seed_start: int = 42,
    classify_max_iter: int = 60,
    route_value: int = 0,
) -> List[Dict[str, Any]]:
    """Construct consistent-no-tail examples with natural random values.

    Each factor gets random entries in [low, high), then F[a,a] is forced to
    be the strict minimum. This keeps cost gaps small (important for tail
    derivation) while guaranteeing route = (a, a, ..., a) from iteration 0.
    """
    examples: List[Dict[str, Any]] = []
    seed = seed_start

    while len(examples) < n_examples:
        rng = np.random.RandomState(seed)
        tables: List[np.ndarray] = []

        for _ in range(cycle_size):
            ct = rng.randint(low, high, size=(domain, domain)).astype(float)
            # force [route_value, route_value] to be strictly the minimum
            current_min = ct.min()
            ct[route_value, route_value] = max(0.0, current_min - 1.0)
            tables.append(ct)

        cls = classify_case_cycle(tables, max_iter=classify_max_iter)
        if cls.get(CASE_CONSISTENT_NO_TAIL, False):
            examples.append(_format_example_result(
                case_name=CASE_CONSISTENT_NO_TAIL,
                seed=seed,
                attempt=1,
                cycle_size=cycle_size,
                generation_strategy="constructed_diagonal",
                cost_tables=tables,
                classification=cls,
            ))
        seed += 1

    return examples


def construct_inconsistent_no_tail_examples(
    n_examples: int,
    *,
    cycle_size: int,
    domain: int = 2,
    low: int = 1,
    high: int = 10,
    seed_start: int = 42,
    classify_max_iter: int = 120,
) -> List[Dict[str, Any]]:
    """Construct Figure 8 examples: one off-diagonal factor + rest diagonal.

    All but the last factor have diagonal minimums (agreement).
    The last factor has off-diagonal minimums (disagreement).
    This structural frustration forces immediate oscillation (no tail).
    """
    examples: List[Dict[str, Any]] = []
    seen: set[tuple] = set()
    seed = seed_start

    while len(examples) < n_examples:
        rng = np.random.RandomState(seed)
        tables: List[np.ndarray] = []

        for idx in range(cycle_size):
            ct = rng.randint(low, high, size=(domain, domain)).astype(float)
            if idx < cycle_size - 1:
                # diagonal: same-value preference
                for v in range(domain):
                    ct[v, v] = 0.0
            else:
                # off-diagonal: different-value preference
                for v in range(domain):
                    ct[v, (v + 1) % domain] = 0.0
            tables.append(ct)

        sig = _cost_tables_signature(tables)
        if sig not in seen:
            cls = classify_case_cycle(tables, max_iter=classify_max_iter)
            if cls.get(CASE_INCONSISTENT_NO_TAIL, False):
                seen.add(sig)
                examples.append(_format_example_result(
                    case_name=CASE_INCONSISTENT_NO_TAIL,
                    seed=seed,
                    attempt=1,
                    cycle_size=cycle_size,
                    generation_strategy="constructed_off_diagonal",
                    cost_tables=tables,
                    classification=cls,
                ))

        seed += 1

    return examples


def run_belief_trace_cycle(
    cost_tables: Sequence[np.ndarray],
    *,
    tracked_values: Mapping[str, Sequence[int]],
    max_iter: int,
    damping_factor: float = 0.9,
    normalize_messages: bool = False,
    subtract_initial: bool = False,
) -> Dict[str, List[float]]:
    """Run Q-damped Min-Sum and record belief traces on an N-variable cycle."""
    fg = _build_cycle_graph_from_cost_tables(cost_tables)
    engine = DampingEngine(
        factor_graph=fg,
        damping_factor=damping_factor,
        normalize_messages=normalize_messages,
    )

    records: Dict[str, List[float]] = {}
    first: Dict[str, float] = {}

    for i in range(max_iter):
        engine.step(i)
        beliefs = engine.get_beliefs()
        for var_name, values in tracked_values.items():
            for value in values:
                key = f"{var_name}_v{int(value)}"
                current = float(beliefs[var_name][value])
                if key not in records:
                    records[key] = []
                    first[key] = current
                records[key].append(current - first[key] if subtract_initial else current)

    return records


def run_belief_trace(
    ct_f12: np.ndarray,
    ct_f23: np.ndarray,
    ct_f31: np.ndarray,
    *,
    tracked_values: Mapping[str, Sequence[int]],
    max_iter: int,
    damping_factor: float = 0.9,
    normalize_messages: bool = False,
    subtract_initial: bool = False,
) -> Dict[str, List[float]]:
    """Backward-compatible trace runner for 3-variable cycles."""
    return run_belief_trace_cycle(
        [ct_f12, ct_f23, ct_f31],
        tracked_values=tracked_values,
        max_iter=max_iter,
        damping_factor=damping_factor,
        normalize_messages=normalize_messages,
        subtract_initial=subtract_initial,
    )


def _canonical_key_order_for_case(case_name: str, *, cycle_size: int = 3, domain: int = 2) -> List[str]:
    _require_valid_case_name(case_name)
    if case_name in {CASE_CONSISTENT_NO_TAIL, CASE_CONSISTENT_WITH_TAIL}:
        return [f"x{i+1}_route" for i in range(cycle_size)]
    return [f"x{i+1}_v{value}" for value in range(domain) for i in range(cycle_size)]


def _extract_cost_tables(example: Mapping[str, Any]) -> CycleCostTables:
    if "cost_tables" in example:
        return _as_cycle_cost_tables(example["cost_tables"])
    return _as_cycle_cost_tables([example["ct_f12"], example["ct_f23"], example["ct_f31"]])


def run_experiment_examples_cycle(
    examples: List[Dict[str, Any]],
    case_name: str,
    max_iter: int,
    damping_factor: float = 0.9,
    normalize_messages: bool = False,
    subtract_initial: bool = False,
) -> List[Dict[str, Any]]:
    """Run traces for a batch of N-variable cycle examples."""
    _require_valid_case_name(case_name)
    runs: List[Dict[str, Any]] = []
    if not examples:
        return runs

    for idx, example in enumerate(examples):
        tables = _extract_cost_tables(example)
        cycle_size = len(tables)
        domain = int(tables[0].shape[0])
        classification = example["classification"]
        key_order = _canonical_key_order_for_case(case_name, cycle_size=cycle_size, domain=domain)

        if case_name in {CASE_CONSISTENT_NO_TAIL, CASE_CONSISTENT_WITH_TAIL}:
            route = route_assignment_from_classification(classification)
            tracked_values = {f"x{i+1}": [int(route[i])] for i in range(cycle_size)}
            raw_records = run_belief_trace_cycle(
                tables,
                tracked_values=tracked_values,
                max_iter=max_iter,
                damping_factor=damping_factor,
                normalize_messages=normalize_messages,
                subtract_initial=subtract_initial,
            )
            canonical_records = {
                f"x{i+1}_route": raw_records[f"x{i+1}_v{int(route[i])}"] for i in range(cycle_size)
            }
        else:
            tracked_values = {f"x{i+1}": list(range(domain)) for i in range(cycle_size)}
            canonical_records = run_belief_trace_cycle(
                tables,
                tracked_values=tracked_values,
                max_iter=max_iter,
                damping_factor=damping_factor,
                normalize_messages=normalize_messages,
                subtract_initial=subtract_initial,
            )

        runs.append(
            {
                "experiment": case_name,
                "example_index": idx,
                "seed": int(example["seed"]),
                "attempt": int(example["attempt"]),
                "cycle_size": cycle_size,
                "domain": domain,
                "classification": classification,
                "records": canonical_records,
                "key_order": key_order,
            }
        )

    return runs


def run_experiment_examples(
    examples: List[Dict[str, Any]],
    case_name: str,
    max_iter: int,
    damping_factor: float = 0.9,
    normalize_messages: bool = False,
    subtract_initial: bool = False,
) -> List[Dict[str, Any]]:
    """Backward-compatible batch trace runner for 3-variable cycles."""
    return run_experiment_examples_cycle(
        examples=examples,
        case_name=case_name,
        max_iter=max_iter,
        damping_factor=damping_factor,
        normalize_messages=normalize_messages,
        subtract_initial=subtract_initial,
    )


def aggregate_example_records(
    example_runs: List[Dict[str, Any]],
    key_order: List[str],
) -> Dict[str, Any]:
    """Aggregate multiple example traces into pointwise mean/std records."""
    if not example_runs:
        return {
            "mean_records": {},
            "std_records": {},
            "n_examples": 0,
            "key_order": key_order,
        }

    mean_records: Dict[str, np.ndarray] = {}
    std_records: Dict[str, np.ndarray] = {}
    for key in key_order:
        arr = np.array([run["records"][key] for run in example_runs], dtype=float)
        mean_records[key] = arr.mean(axis=0)
        std_records[key] = arr.std(axis=0)

    return {
        "mean_records": mean_records,
        "std_records": std_records,
        "n_examples": len(example_runs),
        "key_order": key_order,
    }


def compute_example_slope_summary(
    example_runs: List[Dict[str, Any]],
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    """Compute per-example/per-series slope metrics and spread summary."""
    rows: List[Dict[str, Any]] = []

    for run in example_runs:
        records: Mapping[str, Sequence[float]] = run["records"]
        stats = compute_slope_stats(records, window_start, window_end)
        spread = float(stats["relative_spread_percent"])
        slopes = stats["slopes"]

        for series_key, slope in slopes.items():
            rows.append(
                {
                    "experiment": run["experiment"],
                    "example_index": int(run["example_index"]),
                    "seed": int(run["seed"]),
                    "attempt": int(run["attempt"]),
                    "series_key": series_key,
                    "slope": float(slope),
                    "example_relative_spread_percent": spread,
                }
            )

    return pd.DataFrame(rows)


def compute_slope_stats(
    records: Mapping[str, Sequence[float]],
    window_start: int,
    window_end: int,
) -> Dict[str, Any]:
    """Compute per-series linear slopes and relative spread in the given window."""
    slopes: Dict[str, float] = {}
    for key, values in records.items():
        y = np.array(values[window_start:window_end], dtype=float)
        x = np.arange(len(y), dtype=float)
        slopes[key] = float(np.polyfit(x, y, 1)[0])

    if not slopes:
        spread = 0.0
    else:
        vals = np.array(list(slopes.values()), dtype=float)
        mean_val = float(np.mean(vals))
        spread = 0.0 if abs(mean_val) < 1e-12 else float((np.max(vals) - np.min(vals)) / abs(mean_val) * 100.0)

    return {
        "slopes": slopes,
        "relative_spread_percent": spread,
    }
