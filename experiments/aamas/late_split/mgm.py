"""MGM on every per-variable value observed in the post-split assignment tail."""

from __future__ import annotations

import numpy as np

from experiments.aaai.code.merge import (
    _factors_touching,
    _neighbors,
    _var_key,
    score_assignment,
)


def observed_menus(trace: dict, window: int) -> tuple[dict[str, list[int]], list[int]]:
    """Return all tail values and the first index of each distinct tail assignment."""
    assignments = trace["assignments"]
    if not 1 <= window <= len(assignments):
        raise ValueError("tail window must fit the saved trace")
    names = list(trace["variable_names"])
    tail = assignments[-window:]
    menus = {
        name: sorted(map(int, np.unique(tail[:, i]))) for i, name in enumerate(names)
    }
    _, first = np.unique(tail, axis=0, return_index=True)
    indices = (np.sort(first) + len(assignments) - window).tolist()
    return menus, indices


def mgm_menu_search(
    initial: dict[str, int],
    menus: dict[str, list[int]],
    names: list[str],
    factor_vars: dict[str, list[str]],
    tables: dict[str, np.ndarray],
    max_rounds: int = 10000,
    eps: float = 1e-9,
) -> dict:
    """Run deterministic synchronous MGM-1 over arbitrary finite value menus."""
    if set(initial) != set(names) or set(menus) != set(names) or max_rounds < 1:
        raise ValueError("MGM requires complete assignments/menus and positive rounds")
    candidates = {v: np.array(sorted(set(menus[v])), dtype=int) for v in names}
    if any(len(candidates[v]) == 0 or initial[v] not in candidates[v] for v in names):
        raise ValueError("initial assignment must belong to nonempty menus")
    touching = _factors_touching(factor_vars, names)
    neighbors = _neighbors(factor_vars, names)
    priority = {v: i for i, v in enumerate(sorted(names, key=_var_key))}
    for factor, axes in factor_vars.items():
        for axis, v in enumerate(axes):
            if np.any(candidates[v] < 0) or np.any(
                candidates[v] >= tables[factor].shape[axis]
            ):
                raise ValueError("menu value outside the original domain")
    assignment = dict(initial)
    costs = [score_assignment(assignment, tables, factor_vars)]
    moves = []
    hit_cap = True
    for _ in range(max_rounds):
        gains, best_values = {}, {}
        for v in names:
            values = candidates[v]
            local = np.zeros(len(values), dtype=float)
            for factor in touching[v]:
                indices = tuple(
                    values if u == v else assignment[u] for u in factor_vars[factor]
                )
                local += tables[factor][indices]
            current = int(np.flatnonzero(values == assignment[v])[0])
            best = int(np.argmin(local))
            gain = float(local[current] - local[best])
            gains[v] = gain
            best_values[v] = int(values[best]) if gain > eps else assignment[v]

        def beats(a: str, b: str) -> bool:
            if abs(gains[a] - gains[b]) > eps:
                return gains[a] > gains[b]
            return priority[a] > priority[b]

        movers = [
            v
            for v in names
            if gains[v] > eps and all(beats(v, u) for u in neighbors[v])
        ]
        if not movers:
            hit_cap = False
            break
        for v in movers:
            assignment[v] = best_values[v]
        score = score_assignment(assignment, tables, factor_vars)
        if score >= costs[-1] - eps:
            raise RuntimeError("MGM move did not improve the original objective")
        costs.append(score)
        moves.append(len(movers))
    return {
        "initial_assignment": initial,
        "assignment": assignment,
        "cost": costs[-1],
        "costs": costs,
        "moves_per_round": moves,
        "rounds": len(moves),
        "hit_round_cap": hit_cap,
    }
