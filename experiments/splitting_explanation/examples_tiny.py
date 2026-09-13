"""two tiny worked examples used in the explanation documents.

A. one edge, two values: plain min-sum is exact on a tree and stops after one step; the split adds
   the echo (a two-step loop on the edge) and the doubling, and the messages keep moving until the
   edge locks on one row of the table.
B. a triangle of "be different" constraints (an odd cycle): the split's decision dynamics is
   synchronous best response; without damping it alternates between 000 and 111 (each the worst
   possible assignment) while the alternation itself has cost 0 on the bipartite double cover;
   with damping it settles on a proper 1-opt assignment.
output: results/examples_tiny.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lab import FastEngine, Inst, run_record  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


# -- A: one edge -------------------------------------------------------------------------------
say("== A. one edge, values {0, 1}")
C = np.array([[0.0, 3.0], [4.0, 1.0]])  # C[u, v]: X1 = u, X2 = v
theta = np.array([[0.0, 0.6], [0.5, 0.0]])
inst = Inst(2, 2, [(0, 1)], [C], theta)
say(
    f"table C[X1, X2] = {C.tolist()},  unary X1 = {theta[0].tolist()},  unary X2 = {theta[1].tolist()}"
)
for x in ((0, 0), (0, 1), (1, 0), (1, 1)):
    say(f"  cost{x} = {inst.cost(x):.1f}")
say("optimum: (0, 0) with cost 0.5")
for label, kw in (
    ("plain min-sum", dict()),
    ("split (Q = cavity + belief)", dict(rule="cav+belief")),
    ("split, damping 0.5", dict(rule="cav+belief", lam=0.5)),
):
    e = FastEngine(inst, **kw)
    say(f"-- {label}")
    say(
        "   t   Q(X1->X2)          Q(X2->X1)          belief X1          belief X2      decoded  committed arcs"
    )
    for t in range(8):
        e.step()
        q = e.Q - e.Q.min(axis=1, keepdims=True)
        b = e.beliefs()
        b = b - b.min(axis=1, keepdims=True)
        say(
            f"   {t}   {q[0]}   {q[1]}   {b[0]}   {b[1]}   {tuple(int(v) for v in e.assignment())}   {e.commit_mask().tolist()}"
        )

# -- B: triangle -------------------------------------------------------------------------------
say()
say("== B. triangle X1 - X2 - X3 - X1 of 'be different' constraints, values {0, 1}")
D = np.array([[10.0, 0.0], [0.0, 10.0]])
rng = np.random.default_rng(3)
theta = rng.uniform(0, 1e-2, size=(3, 2))
tri = Inst(3, 2, [(0, 1), (0, 2), (1, 2)], [D, D, D], theta)
say(
    f"every edge: C = {D.tolist()} (10 if equal, 0 if different); tiny random unaries for tie breaking"
)
say(
    "any assignment violates at least one edge (odd cycle): best cost 10, e.g. (0, 0, 1); worst cost 30: (0, 0, 0) or (1, 1, 1)"
)
for label, kw in (
    ("plain min-sum", dict()),
    ("split, no damping", dict(split=0.5)),
    ("split, damping 0.9", dict(split=0.5, lam=0.9)),
):
    r = run_record(FastEngine(tri, **kw), 40)
    seq = ["".join(map(str, a)) for a in r["assigns"]]
    say(f"-- {label}: decoded assignment per iteration")
    say("   " + " ".join(seq[:20]))
    say("   " + " ".join(seq[20:]))
    say(
        f"   costs of the last two: {r['costs'][-2]:.1f}, {r['costs'][-1]:.1f};  committed arcs at the end: {r['sats'][-1]:.2f}"
    )
x, y = np.array([0, 0, 0]), np.array([1, 1, 1])
say(
    f"cost_2(000, 111) = {tri.pair_cost(x, y):.1f}: the alternation between the two layers satisfies every edge of the double cover,"
)
say(
    "while each layer on its own costs 30. no re-phasing of the layers is a single assignment (the triangle is not bipartite)."
)

(HERE / "results" / "examples_tiny.txt").write_text("\n".join(lines) + "\n")
