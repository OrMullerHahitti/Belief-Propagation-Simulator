# Result: a temporary split improves the sparse paper benchmark

The requested improvement over ordinary .5/.5 splitting and .9 old-Q damping
is confirmed on 32 untouched sparse AAAI inputs: 50 variables, domain 10,
integer pairwise costs and the paper's unary tie-breaking preferences.

| Method | Mean final cost | Strictly stable /32 |
| --- | ---: | ---: |
| Ordinary .5 split, .9 damping | 14263.065 | 31 |
| Fixed .8 split, .9 damping | 14246.189 | 29 |
| Temporary .95 split, .9 damping | **14180.288** | **31** |

The pulse reduces baseline mean cost by **82.777 (0.58%)**, paired 95% bootstrap
interval **[-113.404, -55.466]**. Against fixed .8 the reduction is 65.900,
interval [-94.124, -41.405]. Each comparison has **25 wins, 7 ties, no losses**.
Every method's endpoint cost and assignment match at 2,000 and 10,000 updates.
This comparison concerns those checkpoints. All methods receive the same update budget; there is no branch
search, earlier-best-cost selection, learned model or DABP component.

The native `code/pulse.py::SplitPulseEngine` starts with .5/.5 cost splitting,
changes every pairwise factor to .95/.05 before zero-based step 64, and
restores .5/.5 before step 256. It preserves all Q/R messages, leaves unary
splits at .5/.5, and uses .9 damping throughout. `README.md` contains the native
execution example. The change affects the path to the final solution while
the original objective and final split representation are retained.

## Limits that matter

Strict stability combines constant last-100 assignments, small gauge-normalized
changes in all pairwise/unary Q/R, and a small undamped Q-map defect. Baseline
seed 6003 and pulse seed 6002 fail the message checks at both horizons despite
stable assignments. Therefore equal 31/32 counts do not imply per-instance
non-regression or a convergence theorem.

The small K4 experiments do not establish improvement with stable convergence.
For K4/domain10, 32 development inputs showed 8 wins, 22 ties and 2 losses,
but the cost interval includes zero and stability declines from 29/32 to 28/32.
The earlier learned controller also failed the longer-horizon comparison.
This result supports a simple splitting schedule on the tested sparse family,
not the claim that neural or online learning is beneficial.

The schedule is motivated by the paper's active-row and splitting mechanism;
its timings and .95 weight are empirical choices. Separate exact interval
helpers establish pattern-specific row-change/fixed-point facts, not a general
guarantee that the pulse improves cost. The saved September 9 Overleaf source
audit and local `publish/` draft differ; later local theorems are not attributed
to the submitted manuscript.

## Verification

The frozen confirmation protocol precedes seed execution. Every saved input,
source copy, pulse event and trajectory is retained in
`results/aaai_derived_control_20260915/paper_confirmation/`. All **960,000 costs**
were independently reconstructed, with maximum numerical discrepancy
2.55e-11. An independent audit verified all 192 checkpoint records, paired
statistics and 326 artifact hashes. A true-native seed-6000 replay matched all
2,000 baseline and pulse assignments and original costs exactly.

All **39 new tests pass**. Full pytest: 321 passed, 3 pre-existing Figure 5/8
reproduction failures, 2 skipped. `make ci` stops on seven pre-existing
formatting failures; no unrelated files were reformatted. The source changes
are experiment-local and no commit or publication was performed.

For detailed results, reproduction and limitations, read
`results/aaai_derived_control_20260915/paper_confirmation/README.md`.
