"""Assignment scoring, MGM-1 binary-menu merge, and branch and bound.

Adapted from notebooks/06_split_oscillation_mgm_merge.ipynb. All costs are
evaluated on the original (pre-split) cost tables captured by
problems.capture_original().
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _var_key(name: str) -> Tuple[int, str]:
    return (int(name[1:]) if name[1:].isdigit() else 0, name)


def score_assignment(
    assignment: Dict[str, int],
    tables: Dict[str, np.ndarray],
    factor_vars: Dict[str, List[str]],
) -> float:
    """total cost of an assignment on the original cost tables."""
    total = 0.0
    for fname, table in tables.items():
        idx = tuple(int(assignment[v]) for v in factor_vars[fname])
        total += float(table[idx])
    return total


def _factors_touching(
    factor_vars: Dict[str, List[str]], var_names: List[str]
) -> Dict[str, List[str]]:
    touching: Dict[str, List[str]] = {v: [] for v in var_names}
    for fname, vs in factor_vars.items():
        for v in vs:
            touching[v].append(fname)
    return touching


def _neighbors(
    factor_vars: Dict[str, List[str]], var_names: List[str]
) -> Dict[str, set]:
    nb: Dict[str, set] = {v: set() for v in var_names}
    for vs in factor_vars.values():
        for a in vs:
            for b in vs:
                if a != b:
                    nb[a].add(b)
    return nb


def _local_cost(
    v: str,
    assignment: Dict[str, int],
    tables: Dict[str, np.ndarray],
    factor_vars: Dict[str, List[str]],
    factors_touching: Dict[str, List[str]],
) -> float:
    total = 0.0
    for fname in factors_touching[v]:
        idx = tuple(assignment[u] for u in factor_vars[fname])
        total += float(tables[fname][idx])
    return total


def mgm1_binary_merge(
    branch1: Dict[str, int],
    branch2: Dict[str, int],
    start_branch: str,
    var_names: List[str],
    factor_vars: Dict[str, List[str]],
    tables: Dict[str, np.ndarray],
    max_rounds: int = 10_000,
    eps: float = 1e-9,
) -> Tuple[Dict[str, int], List[float], List[int]]:
    """MGM-1 (Maximum Gain Message) coordinated 1-opt local search restricted to
    the {branch1[v], branch2[v]} menu at every variable. agreement variables are
    frozen, so the result is a genuine merge of the two branches.

    returns (assignment, cost_history, moves_per_round). seeded from
    ``start_branch`` the result never exceeds that branch's cost.
    """
    menu: Dict[str, tuple] = {}
    for v in var_names:
        a, b = int(branch1[v]), int(branch2[v])
        menu[v] = (a,) if a == b else (a, b)

    factors_touching = _factors_touching(factor_vars, var_names)
    neighbors = _neighbors(factor_vars, var_names)
    priority = {v: i for i, v in enumerate(sorted(var_names, key=_var_key))}

    start = {"branch1": branch1, "branch2": branch2}[start_branch]
    assignment = {v: int(start[v]) for v in var_names}

    history = [score_assignment(assignment, tables, factor_vars)]
    moves_per_round: List[int] = []

    for _ in range(max_rounds):
        gain: Dict[str, float] = {}
        best_val: Dict[str, int] = {}
        for v in var_names:
            cur = assignment[v]
            cur_lc = _local_cost(v, assignment, tables, factor_vars, factors_touching)
            bv, blc = cur, cur_lc
            for cand in menu[v]:
                if cand == cur:
                    continue
                assignment[v] = cand
                lc = _local_cost(v, assignment, tables, factor_vars, factors_touching)
                if lc < blc - eps:
                    blc, bv = lc, cand
                assignment[v] = cur
            gain[v] = cur_lc - blc
            best_val[v] = bv

        def beats(a: str, b: str) -> bool:
            if abs(gain[a] - gain[b]) > eps:
                return gain[a] > gain[b]
            return priority[a] > priority[b]

        movers = [
            v
            for v in var_names
            if gain[v] > eps and all(beats(v, u) for u in neighbors[v])
        ]
        if not movers:
            break
        for v in movers:
            assignment[v] = best_val[v]
        history.append(score_assignment(assignment, tables, factor_vars))
        moves_per_round.append(len(movers))

    return assignment, history, moves_per_round


def branch_and_bound(
    var_names: List[str],
    factor_vars: Dict[str, List[str]],
    tables: Dict[str, np.ndarray],
    domains: Dict[str, Sequence[int]],
    initial_upper_bound: float = float("inf"),
    initial_assignment: Dict[str, int] | None = None,
    time_limit_s: float | None = 60.0,
) -> Tuple[float, Dict[str, int], Dict[str, object]]:
    """depth-first branch and bound over per-variable candidate values.

    `domains` maps each variable to its candidate values — the full domain for
    an exact solve, or the two-branch menu for the optimal merge. the lower
    bound is the running sum over factors of the table minimum consistent with
    the current partial assignment.

    returns (best_cost, best_assignment, stats); stats["complete"] is False
    when the time limit was hit, in which case best_cost is only an upper bound.
    """
    deg = {v: 0 for v in var_names}
    for vs in factor_vars.values():
        for u in vs:
            for w in vs:
                if u != w:
                    deg[u] += 1
    var_order = sorted(var_names, key=lambda v: (-deg[v], _var_key(v)))

    factors_touching = _factors_touching(factor_vars, var_names)
    factor_contrib = {f: float(tables[f].min()) for f in factor_vars}
    running_bound = sum(factor_contrib.values())

    best_cost = float(initial_upper_bound)
    best_assignment = dict(initial_assignment) if initial_assignment else {}
    assignment: Dict[str, int] = {}

    start = time.time()
    stats: Dict[str, object] = {"nodes": 0, "prunes": 0, "complete": False, "elapsed_s": 0.0}

    def factor_min_given_partial(fname: str) -> float:
        vs = factor_vars[fname]
        tbl = tables[fname]
        if all(v in assignment for v in vs):
            return float(tbl[tuple(assignment[v] for v in vs)])
        slicer = tuple(
            assignment[v] if v in assignment else slice(None) for v in vs
        )
        return float(tbl[slicer].min())

    def search(depth: int, bound_in: float) -> None:
        nonlocal best_cost, best_assignment
        stats["nodes"] += 1
        if time_limit_s is not None and (time.time() - start) > time_limit_s:
            raise TimeoutError()
        if depth == len(var_order):
            if bound_in < best_cost:
                best_cost = bound_in
                best_assignment = dict(assignment)
            return
        v = var_order[depth]
        touched = factors_touching[v]
        saved = {f: factor_contrib[f] for f in touched}
        for val in domains[v]:
            assignment[v] = int(val)
            delta = 0.0
            for f in touched:
                new = factor_min_given_partial(f)
                delta += new - factor_contrib[f]
                factor_contrib[f] = new
            new_bound = bound_in + delta
            if new_bound < best_cost:
                search(depth + 1, new_bound)
            else:
                stats["prunes"] += 1
            for f in touched:
                factor_contrib[f] = saved[f]
        del assignment[v]

    try:
        search(0, running_bound)
        stats["complete"] = True
    except TimeoutError:
        stats["complete"] = False
    stats["elapsed_s"] = time.time() - start
    return best_cost, best_assignment, stats
