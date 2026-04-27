# Figure 5a / 5b / 8 — Dataset Generation

How the CSV datasets under `aij/data/figure_{5a,5b,8}_cycle_{3,6,9,12}_{examples,traces}.csv` were produced.

## Run

```bash
uv run python experiments/aij/code/generate_fig58_csv.py
```

This calls `experiments.aij.code.utils.fig58_repro` to generate **50 strictly-classified examples per (figure, cycle size)** — 600 examples total — and runs damped Min-Sum BP on each to record per-iteration belief traces.

## Parameters

| | Figure 5a | Figure 5b | Figure 8 |
|---|---|---|---|
| Case | consistent_no_tail | consistent_with_tail | inconsistent_no_tail |
| Strategy | random_full | random_full | motif_repeat + binary swap-mask variants (curated seeds first) |
| Iterations (trace) | 50 | 50 | 70 |

Common: `domain=2`, `low=0`, `high=10`, `damping_factor=0.9`, `normalize_messages=False`, `subtract_initial=False`, `classify_max_iter=140`. Cycle sizes: 3, 6, 9, 12. Each candidate is validated by an undamped-BP classifier (`classify_case_cycle`) before acceptance.

## Outputs

For each (figure, cycle):

- `figure_{id}_cycle_{N}_examples.csv` — 50 rows, one per generated graph (`seed`, `cost_tables`, `period`, `periodic_route`, `assignment_trace`, classification flags).
- `figure_{id}_cycle_{N}_traces.csv` — one row per `(example, series_key, iteration)`. fig5a/5b track the route value per variable (`xN_route`); fig8 tracks both domain values per variable (`xN_v0`, `xN_v1`).

## Reproduction

Any reviewer can reconstruct an exact graph from a row's `cost_tables`, re-run undamped BP to verify the classification, then re-run damped BP at the configured iteration count and check the traces. The replay path in `reproduce_figure.py --replay` does this for the per-figure plots.
