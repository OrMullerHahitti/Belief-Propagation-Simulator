# Seed-0 Identity/Timing Diagnostic

This experiment instruments one fixed random graph instance:

- seed: `0`
- variables: `100`
- domain size: `2`
- density: `0.5`
- normalized messages: enabled
- max iterations: `200`
- snapshots: not written

It reuses the existing PropFlow engines and the existing
`experiments.non_convergence_chain.oscillation_detector` classifier. It does
not change update rules, normalization, tie-breaking, or classifier semantics.

Run:

```bash
.venv/bin/python -m experiments.seed0_identity_timing_diagnostic.run_diagnostic
```

Outputs are written to:

```text
artifacts/seed0_identity_timing_diagnostic/
```

The report uses diagnostic language only. It describes observed behavior in
this tested instance and does not claim a theorem.

