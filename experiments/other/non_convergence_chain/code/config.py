"""Configuration and graph builders for the non-convergence chain study."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from propflow import FactorAgent, FactorGraph, FGBuilder, VariableAgent


@dataclass(frozen=True)
class FactorTableConfig:
    """Explicit factor cost table and ordered variable scope."""

    name: str
    variables: list[str]
    table: list[Any] | None = None

    def as_array(self) -> np.ndarray:
        if self.table is None:
            raise ValueError(
                f"Cost table for factor '{self.name}' is missing. "
                "Paste the meeting example table into the YAML config."
            )
        arr = np.asarray(self.table, dtype=float)
        if arr.ndim != len(self.variables):
            raise ValueError(
                f"Factor '{self.name}' connects {len(self.variables)} variables "
                f"but table has ndim={arr.ndim}."
            )
        return arr


@dataclass(frozen=True)
class RandomGraphConfig:
    """Seeded random-graph experiment settings."""

    enabled: bool = False
    num_vars: int = 20
    domain_size: int = 2
    density: float = 0.25
    ct_factory: str = "random_int"
    ct_params: dict[str, Any] = field(default_factory=lambda: {"low": 0, "high": 10})
    max_iter: int | None = None
    split_at_iters: list[int] = field(default_factory=list)
    split_fraction: float = 1.0
    split_percentages: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    percentage_split_at_iter: int | None = None
    run_split_at_sweep: bool = True
    run_percentage_sweep: bool = True
    run_combined_sweep: bool = True
    split_transfer_modes: list[str] = field(default_factory=lambda: ["transfer"])


@dataclass(frozen=True)
class NonConvergenceConfig:
    """Full configuration for the chain diagnostics CLI."""

    graph_name: str
    run_chain: bool | None
    symmetric_chain_split: bool
    symmetric_chain_copies: int
    symmetric_chain_cost_scale: float
    variable_order: list[str]
    domain_values: list[Any]
    cost_tables: list[FactorTableConfig]
    max_iter: int = 200
    tolerance: float = 1e-9
    split_at_iters: list[int] = field(default_factory=lambda: [20, 40, 60, 80])
    split_ratio: float = 0.5
    split_targets: list[str] | None = None
    damping_factors: list[float] = field(default_factory=lambda: [0.5, 0.9])
    trace_every: int = 1
    save_snapshots: bool = True
    output_dir: str = "results/non_convergence_chain"
    seed: int = 0
    random_graph: RandomGraphConfig = field(default_factory=RandomGraphConfig)

    @property
    def domain_size(self) -> int:
        return len(self.domain_values)

    def require_cost_tables(self) -> None:
        if not self.cost_tables:
            raise ValueError(
                "No cost_tables were provided. Paste F12 and F23 into the YAML config."
            )
        names = {factor.name for factor in self.cost_tables}
        required = {"F12", "F23"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                f"Missing required chain factor(s): {missing}. "
                "The primary symmetric chain experiment expects F12 and F23."
            )
        for factor in self.cost_tables:
            factor.as_array()

    def has_complete_chain_cost_tables(self) -> bool:
        """Return whether the exact symmetric split chain can be built."""

        try:
            self.require_cost_tables()
        except ValueError:
            return False
        return True

    def should_run_chain(self) -> bool:
        """Resolve whether chain experiments should run.

        If ``run_chain`` is omitted, chain mode is enabled by default unless the
        config clearly describes a random-graph-only run with incomplete chain
        tables.
        """

        if self.run_chain is not None:
            return self.run_chain
        if self.random_graph.enabled and not self.has_complete_chain_cost_tables():
            return False
        return True


def _load_raw_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "YAML config loading requires PyYAML. Install pyyaml or use JSON."
        ) from exc
    loaded = yaml.safe_load(path.read_text())
    return dict(loaded or {})


def normalize_split_percentage(value: float) -> float:
    """Normalize a percentage value supplied as 0.1 or 10 into a fraction."""

    normalized = float(value)
    if 1.0 < normalized <= 100.0:
        normalized /= 100.0
    return normalized


def load_config(
    path: str | Path, *, output_dir: str | None = None
) -> NonConvergenceConfig:
    """Load a YAML/JSON study configuration into dataclasses."""

    raw = _load_raw_config(Path(path))
    factors = [
        FactorTableConfig(
            name=str(item["name"]),
            variables=[str(v) for v in item["variables"]],
            table=item.get("table"),
        )
        for item in raw.get("cost_tables", [])
    ]
    random_graph_raw = raw.get("random_graph") or {}
    split_transfer_modes = [
        str(mode) for mode in random_graph_raw.get("split_transfer_modes", ["transfer"])
    ]
    invalid_modes = sorted(set(split_transfer_modes) - {"reset", "transfer"})
    if invalid_modes:
        raise ValueError(
            "random_graph.split_transfer_modes must contain only "
            f"'transfer' and/or 'reset'; got {invalid_modes}."
        )
    if not split_transfer_modes:
        raise ValueError("random_graph.split_transfer_modes cannot be empty.")
    split_percentages = [
        normalize_split_percentage(value)
        for value in random_graph_raw.get(
            "split_percentages", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
    ]
    invalid_percentages = [
        value for value in split_percentages if not 0.0 < value <= 1.0
    ]
    if invalid_percentages:
        raise ValueError(
            "random_graph.split_percentages must be fractions in (0, 1]; "
            f"got {invalid_percentages}."
        )
    if not split_percentages:
        raise ValueError("random_graph.split_percentages cannot be empty.")
    cfg = NonConvergenceConfig(
        graph_name=str(raw.get("graph_name", "nonconverging_chain")),
        run_chain=(
            None if raw.get("run_chain") is None else bool(raw.get("run_chain"))
        ),
        symmetric_chain_split=bool(raw.get("symmetric_chain_split", True)),
        symmetric_chain_copies=int(raw.get("symmetric_chain_copies", 2)),
        symmetric_chain_cost_scale=float(raw.get("symmetric_chain_cost_scale", 1.0)),
        variable_order=[str(v) for v in raw.get("variable_order", ["X1", "X2", "X3"])],
        domain_values=list(raw.get("domain_values", [0, 1])),
        cost_tables=factors,
        max_iter=int(raw.get("max_iter", 200)),
        tolerance=float(raw.get("tolerance", 1e-9)),
        split_at_iters=[int(v) for v in raw.get("split_at_iters", [20, 40, 60, 80])],
        split_ratio=float(raw.get("split_ratio", 0.5)),
        split_targets=(
            None
            if raw.get("split_targets") is None
            else [str(v) for v in raw.get("split_targets", [])]
        ),
        damping_factors=[float(v) for v in raw.get("damping_factors", [0.5, 0.9])],
        trace_every=max(1, int(raw.get("trace_every", 1))),
        save_snapshots=bool(raw.get("save_snapshots", True)),
        output_dir=str(
            output_dir or raw.get("output_dir", "results/non_convergence_chain")
        ),
        seed=int(raw.get("seed", 0)),
        random_graph=RandomGraphConfig(
            enabled=bool(random_graph_raw.get("enabled", False)),
            num_vars=int(random_graph_raw.get("num_vars", 20)),
            domain_size=int(
                random_graph_raw.get(
                    "domain_size", len(raw.get("domain_values", [0, 1]))
                )
            ),
            density=float(random_graph_raw.get("density", 0.25)),
            ct_factory=str(random_graph_raw.get("ct_factory", "random_int")),
            ct_params=dict(random_graph_raw.get("ct_params", {"low": 0, "high": 10})),
            max_iter=random_graph_raw.get("max_iter"),
            split_at_iters=[int(v) for v in random_graph_raw.get("split_at_iters", [])],
            split_fraction=float(random_graph_raw.get("split_fraction", 1.0)),
            split_percentages=split_percentages,
            percentage_split_at_iter=(
                None
                if random_graph_raw.get("percentage_split_at_iter") is None
                else int(random_graph_raw["percentage_split_at_iter"])
            ),
            run_split_at_sweep=bool(random_graph_raw.get("run_split_at_sweep", True)),
            run_percentage_sweep=bool(
                random_graph_raw.get("run_percentage_sweep", True)
            ),
            run_combined_sweep=bool(random_graph_raw.get("run_combined_sweep", True)),
            split_transfer_modes=split_transfer_modes,
        ),
    )
    return cfg


def build_chain_graph(config: NonConvergenceConfig) -> FactorGraph:
    """Build the configured symmetric split X1-X2-X3 chain graph.

    The meeting structure uses two parallel copies of each pasted pairwise
    table. By default the copies keep the full pasted cost table, matching the
    diagrammed symmetric structure, rather than using the mid-run split policy's
    ``p*C`` and ``(1-p)*C`` scaling.
    """

    config.require_cost_tables()
    if config.symmetric_chain_copies < 1:
        raise ValueError("symmetric_chain_copies must be at least 1.")
    if config.symmetric_chain_cost_scale <= 0:
        raise ValueError("symmetric_chain_cost_scale must be positive.")

    variables = {
        name: VariableAgent(name=name, domain=config.domain_size)
        for name in config.variable_order
    }
    factors: list[FactorAgent] = []
    edges: dict[FactorAgent, list[VariableAgent]] = {}

    for factor_cfg in config.cost_tables:
        table = factor_cfg.as_array()
        expected_shape = (config.domain_size,) * len(factor_cfg.variables)
        if table.shape != expected_shape:
            raise ValueError(
                f"Factor '{factor_cfg.name}' table shape {table.shape} does not match "
                f"expected shape {expected_shape}."
            )
        try:
            factor_vars = [variables[name] for name in factor_cfg.variables]
        except KeyError as exc:
            raise ValueError(
                f"Factor '{factor_cfg.name}' references unknown variable {exc}."
            ) from exc

        copy_count = (
            config.symmetric_chain_copies if config.symmetric_chain_split else 1
        )
        for copy_index in range(copy_count):
            factor_name = (
                f"{factor_cfg.name}_{copy_index + 1}"
                if config.symmetric_chain_split
                else factor_cfg.name
            )
            factor = FactorAgent.create_from_cost_table(
                factor_name, table * config.symmetric_chain_cost_scale
            )
            factors.append(factor)
            edges[factor] = factor_vars

    return FGBuilder.build_from_edges(
        variables=[variables[name] for name in config.variable_order],
        factors=factors,
        edges=edges,
    )


def build_random_graph(config: NonConvergenceConfig) -> FactorGraph:
    """Build a seeded random graph for split-percentage experiments."""

    random_cfg = config.random_graph
    np.random.seed(config.seed)
    return FGBuilder.build_random_graph(
        num_vars=random_cfg.num_vars,
        domain_size=random_cfg.domain_size,
        ct_factory=random_cfg.ct_factory,
        ct_params=random_cfg.ct_params,
        density=random_cfg.density,
        seed=config.seed,
    )
