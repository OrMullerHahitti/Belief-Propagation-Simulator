"""Exact bistability check for the butterfly (SCFG) factor graph.

predicts convergence/non-convergence of min-sum BP from cost tables alone,
without running the algorithm. based on the parity map framework:

the center variable x2 receives messages from left (x1 side) and right (x3 side).
due to the butterfly symmetry (f1=f5, f4=f6), the message dynamics reduce to a
two-dimensional piecewise-linear map T on the center message differences (p, q):

    T(p, q) = (G_L(p) + 2·G_R(q),  2·G_L(p) + G_R(q))

where G_L, G_R are round-trip maps through each wing of the butterfly.
non-convergence occurs when odd and even iteration subsequences converge to
different fixed points of T (bistability).
"""

from __future__ import annotations

import numpy as np


def make_round_trip_map(ct: np.ndarray):
    """builds the round-trip map G for one wing of the butterfly.

    for the left wing (ct = L):
        phi(p) = min(L[0,0]+p, L[0,1]) - min(L[1,0]+p, L[1,1])
        psi(r) = min(L[0,0]+r, L[1,0]) - min(L[0,1]+r, L[1,1])
        G(p) = psi(phi(p))

    for the right wing (ct = R), the roles of rows/cols swap:
        phi(q) = min(R[0,0]+q, R[1,0]) - min(R[0,1]+q, R[1,1])
        psi(s) = min(R[0,0]+s, R[0,1]) - min(R[1,0]+s, R[1,1])
        G(q) = psi(phi(q))
    """
    a, b, c, d = ct[0, 0], ct[0, 1], ct[1, 0], ct[1, 1]

    def phi(p):
        return min(a + p, b) - min(c + p, d)

    def psi(r):
        return min(a + r, c) - min(b + r, d)

    def G(p):
        return psi(phi(p))

    return G


def make_right_round_trip_map(ct: np.ndarray):
    """builds the round-trip map G_R for the right wing.

    the right wing connects x2-x3, so the message flow has transposed roles
    compared to the left wing (x1-x2).
    """
    a, b, c, d = ct[0, 0], ct[0, 1], ct[1, 0], ct[1, 1]

    def phi(q):
        return min(a + q, c) - min(b + q, d)

    def psi(s):
        return min(a + s, b) - min(c + s, d)

    def G(q):
        return psi(phi(q))

    return G


def T_map(p: float, q: float, GL, GR) -> tuple[float, float]:
    """two-step center map: T(p,q) = (G_L(p) + 2·G_R(q), 2·G_L(p) + G_R(q))."""
    gl = GL(p)
    gr = GR(q)
    return gl + 2 * gr, 2 * gl + gr


def check_bistability(
    ct_left: np.ndarray,
    ct_right: np.ndarray,
    max_steps: int = 50,
    atol: float = 1e-10,
) -> dict:
    """checks if a butterfly instance is bistable (non-convergent).

    iterates the parity map T from two initial conditions representing
    odd and even iteration subsequences. if they converge to different
    fixed points, the instance is bistable.

    args:
        ct_left: 2x2 cost table for the left wing (f1=f5).
        ct_right: 2x2 cost table for the right wing (f4=f6).
        max_steps: number of T iterations for each subsequence.
        atol: tolerance for comparing fixed points.

    returns:
        dict with keys:
            bistable: True if odd/even fixed points differ
            odd_fp: (p, q) fixed point of the odd subsequence
            even_fp: (p, q) fixed point of the even subsequence
    """
    GL = make_round_trip_map(ct_left)
    GR = make_right_round_trip_map(ct_right)

    # odd subsequence initial condition: first iteration messages
    L = ct_left
    R = ct_right
    u0 = min(L[0, 0], L[1, 0]) - min(L[0, 1], L[1, 1])
    v0 = min(R[0, 0], R[0, 1]) - min(R[1, 0], R[1, 1])
    p_odd = u0 + 2 * v0
    q_odd = 2 * u0 + v0

    # even subsequence initial condition: all-zero messages
    p_even, q_even = T_map(0, 0, GL, GR)

    # iterate to fixed points
    for _ in range(max_steps):
        p_odd, q_odd = T_map(p_odd, q_odd, GL, GR)
        p_even, q_even = T_map(p_even, q_even, GL, GR)

    same = np.isclose(p_odd, p_even, atol=atol) and np.isclose(q_odd, q_even, atol=atol)

    return {
        "bistable": not same,
        "odd_fp": (p_odd, q_odd),
        "even_fp": (p_even, q_even),
    }


def scan_seeds(
    n_samples: int = 10000,
    seed_start: int = 0,
    domain: int = 2,
) -> list[dict]:
    """scans random butterfly instances and returns all bistable ones.

    uses the same RNG seeding as butterfly_convergence.py for reproducibility.
    """
    bistable_instances = []

    for i in range(n_samples):
        seed = seed_start + i
        rng = np.random.RandomState(seed)
        ct_left = rng.uniform(0, 1, (domain, domain))
        ct_right = rng.uniform(0, 1, (domain, domain))

        result = check_bistability(ct_left, ct_right)
        if result["bistable"]:
            bistable_instances.append({
                "seed": seed,
                "ct_left": ct_left.tolist(),
                "ct_right": ct_right.tolist(),
                "odd_fp": result["odd_fp"],
                "even_fp": result["even_fp"],
            })

    return bistable_instances


if __name__ == "__main__":
    import json
    from pathlib import Path

    print("scanning 10,000 butterfly instances for bistability...")
    results = scan_seeds(n_samples=10000)
    print(f"found {len(results)} bistable instances")

    # print seeds
    seeds = [r["seed"] for r in results]
    print(f"seeds: {seeds}")

    # save
    out_path = Path(__file__).resolve().parent / "results" / "bistable_instances.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved: {out_path}")
