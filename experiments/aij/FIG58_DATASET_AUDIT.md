# Figure 5a/5b/8 Dataset Generation Audit

This document records the exact implementation and runtime procedure used to generate the CSV datasets under:

- `expiriments/generated_csv/figure_5a/cycle_{3,6,9,12}/`
- `expiriments/generated_csv/figure_5b/cycle_{3,6,9,12}/`
- `expiriments/generated_csv/figure_8/cycle_{3,6,9,12}/`

Each folder contains:

- `examples.csv` (50 examples)
- `traces.csv` (belief traces per iteration)

Total examples generated: `3 figures * 4 cycle sizes * 50 = 600`.

---

## 1. Scope and Fixed Parameters

The generation script is:

- `expiriments/generate_fig58_csv.py`

Main constants:

```python
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

DAMPING_FACTOR = 0.9
NORMALIZE_MESSAGES = False
SUBTRACT_INITIAL = False
CLASSIFY_MAX_ITER = 140
```

Figure mapping:

```python
FIGURE_CONFIGS = (
    {"figure_id": "figure_5a", "case_name": "consistent_no_tail", "generation_strategy": "random_full"},
    {"figure_id": "figure_5b", "case_name": "consistent_with_tail", "generation_strategy": "random_full"},
    {"figure_id": "figure_8",  "case_name": "inconsistent_no_tail", "generation_strategy": "motif_repeat"},
)
```

---

## 2. Core Graph Construction

General cycle graph construction is implemented in:

- `expiriments/utils/fig58_repro.py`

The cycle builder accepts any number of binary cost tables (one per factor edge around the cycle):

```python
def _build_cycle_graph_from_cost_tables(cost_tables):
    tables = _as_cycle_cost_tables(cost_tables)
    cycle_size = len(tables)
    domain = int(tables[0].shape[0])

    variables = [VariableAgent(f"x{i+1}", domain=domain) for i in range(cycle_size)]
    factors = []
    edges = {}

    for idx, ct in enumerate(tables):
        left_idx = idx + 1
        right_idx = (idx + 1) % cycle_size + 1
        factor = FactorAgent.create_from_cost_table(f"f{left_idx}{right_idx}", np.array(ct, dtype=float))
        factors.append(factor)
        edges[factor] = [variables[idx], variables[(idx + 1) % cycle_size]]

    return FGBuilder.build_from_edges(variables=variables, factors=factors, edges=edges)
```

---

## 3. Classification Logic (What Makes 5a/5b/8)

### 3.1 Assignment Trace Collection

For a candidate graph, undamped BP (`BPEngine`, `normalize_messages=False`) is stepped for `max_iter` and the assignment tuple is recorded at every iteration:

```python
engine = BPEngine(factor_graph=fg, normalize_messages=False)
assignments = []
for i in range(max_iter):
    engine.step(i)
    assignments.append(tuple(int(engine.assignments[f"x{j+1}"]) for j in range(cycle_size)))
```

### 3.2 Periodic Route Detection

The code finds a stable periodic suffix and earliest start index (`_find_route_info`).

- `period`: periodic route length
- `periodic_start`: first iteration where route repeats forever
- `no_tail`: `periodic_start == 0`

### 3.3 Route Value Sets by Variable

For each variable, collect all values it takes in the periodic route:

```python
route_values_by_var = tuple(
    tuple(sorted({assignment[var_idx] for assignment in info.route}))
    for var_idx in range(cycle_size)
)
```

### 3.4 Case Definitions Used

```python
consistent = all(len(values) == 1 for values in route_values_by_var)
inconsistent = not consistent
no_tail = info.start_index == 0
all_domain_values = tuple(range(domain))
inconsistent_all_domain_values = all(tuple(values) == all_domain_values for values in route_values_by_var)
```

Then:

- `consistent_no_tail` (Figure 5a): `consistent and no_tail`
- `consistent_with_tail` (Figure 5b): `consistent and not no_tail`
- `inconsistent_no_tail` (Figure 8): `inconsistent and no_tail and inconsistent_all_domain_values`

For domain 2, `inconsistent_all_domain_values` means every variable oscillates over both values `{0,1}` in the periodic route.

---

## 4. Example Generation Strategy by Figure

All examples are validated by the strict classifier above before acceptance.

### 4.1 Figure 5a / Figure 5b

Strategy: `random_full`

Per attempt, generate one table per cycle edge:

```python
tables = tuple(rng.randint(low, high, size=(domain, domain)).astype(float) for _ in range(cycle_size))
```

Find first valid candidate per seed (`find_case_cycle`), then collect unique examples across seeds (`find_cases_cycle`), deduplicating by flattened cost-table signature.

### 4.2 Figure 8

Target is strict `inconsistent_no_tail` for all cycle sizes.

Base strategy: `motif_repeat`

```python
motif = [rng.randint(low, high, size=(domain, domain)).astype(float) for _ in range(3)]
tables = tuple(np.array(motif[idx % 3], dtype=float) for idx in range(cycle_size))
```

Then apply binary label-symmetry variants via swap masks, and revalidate each candidate strictly:

```python
for mask_int in range(1 << cycle_size):
    candidate_tables = _swap_binary_tables(base_tables, mask)
    classification = classify_case_cycle(candidate_tables, max_iter=CLASSIFY_MAX_ITER)
    if classification.get(case_name, False):
        accept()
```

This keeps semantics strict while making generation practical for larger cycles.

Curated base seeds are used first for speed:

```python
CURATED_FIG8_BASE_SEEDS = {
    3: [488, 619, 781, 1079, 1100, 1244, 1472, 1876],
    6: [488],
    9: [488],
    12: [488],
}
```

If needed, fallback deterministic search continues from `seed_start`.

---

## 5. Trace Generation (What Numbers Are Saved)

Belief traces are generated with `DampingEngine`:

```python
engine = DampingEngine(
    factor_graph=fg,
    damping_factor=0.9,
    normalize_messages=False,
)
```

At every iteration:

```python
engine.step(i)
beliefs = engine.get_beliefs()
```

The saved value is the raw belief value (`subtract_initial=False`), not normalized-to-zero and not differenced.

### 5.1 Which series are tracked

- Figure 5a / 5b: one value per variable, the route value only:
  - series keys: `x1_route, x2_route, ...`
  - count per example: `cycle_size`
- Figure 8: all domain values for all variables:
  - series keys: `x1_v0, x2_v0, ..., xn_v0, x1_v1, ..., xn_v1`
  - count per example: `2 * cycle_size`

### 5.2 Iteration count

- Figure 5a / 5b: `50`
- Figure 8: `70`

---

## 6. CSV Schemas

## 6.1 `examples.csv`

One row per generated example graph.

Columns:

- `figure_id`
- `case_name`
- `cycle_size`
- `example_index`
- `seed`
- `generation_strategy`
- `period`
- `periodic_start`
- `consistent`
- `inconsistent`
- `no_tail`
- `unclassified`
- `route_values_by_var`
- `periodic_route`
- `assignment_trace`
- `cost_tables`

Notes:

- `route_values_by_var`, `periodic_route`, `assignment_trace`, `cost_tables` are JSON-encoded payloads inside CSV cells.

## 6.2 `traces.csv`

One row per `(example, series_key, iteration)`.

Columns:

- `figure_id`
- `case_name`
- `cycle_size`
- `example_index`
- `seed`
- `iteration`
- `series_key`
- `variable_name`
- `tracked_kind` (`route` or `value`)
- `tracked_value` (0 or 1 in this dataset)
- `belief` (float)

---

## 7. Row Count Formulas (Used for Verification)

For each figure and cycle size:

- `examples.csv` rows: `50`
- `traces.csv` rows:
  - Figure 5a/5b: `50 * cycle_size * 50`
  - Figure 8: `50 * (2 * cycle_size) * 70`

Concrete expected counts:

- Figure 5a:
  - cycle 3: 7,500
  - cycle 6: 15,000
  - cycle 9: 22,500
  - cycle 12: 30,000
- Figure 5b:
  - cycle 3: 7,500
  - cycle 6: 15,000
  - cycle 9: 22,500
  - cycle 12: 30,000
- Figure 8:
  - cycle 3: 21,000
  - cycle 6: 42,000
  - cycle 9: 63,000
  - cycle 12: 84,000

---

## 8. What Each Example Means

Each row in `examples.csv` is a fully specified BP experiment instance:

1. A specific cycle graph topology size (`cycle_size`).
2. A specific set of factor cost tables (`cost_tables`).
3. A strict dynamic classification under undamped BP:
   - 5a: immediate consistent periodic behavior.
   - 5b: consistent periodic behavior after a transient tail.
   - 8: immediate inconsistent periodic behavior where all variables cover both binary values.
4. Its exact assignment trajectory (`assignment_trace`) and periodic core (`periodic_route`).

This means any reviewer can:

- Reconstruct the exact graph from `cost_tables`.
- Re-run undamped BP and verify classification.
- Re-run damped BP and verify trace behavior in `traces.csv`.

---

## 9. Reproduction Commands

Generate all datasets:

```bash
cd /Users/or/Projects/Belief-Propagation-Simulator
.venv/bin/python -u expiriments/generate_fig58_csv.py
```

Run tests validating utilities + CSV pipeline:

```bash
cd /Users/or/Projects/Belief-Propagation-Simulator
.venv/bin/python -m pytest tests/test_fig58_repro_utils.py tests/test_fig58_csv_generation.py -q
```

---

## 10. Files Touched for This Pipeline

- `expiriments/utils/fig58_repro.py`
- `expiriments/generate_fig58_csv.py`
- `tests/test_fig58_csv_generation.py`
- `expiriments/FIG58_DATASET_AUDIT.md`

