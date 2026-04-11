from propflow_server.models import EngineConfig, FactorSpec, GraphSpec, VariableSpec
from propflow_server.simulation import run_simulation


def test_engine_config_accepts_r_damping() -> None:
    config = EngineConfig(max_iters=1, damping=0.0, r_damping=0.3, engine_type="min_sum")
    spec = GraphSpec(
        variables=[
            VariableSpec(name="x1", domain_size=2),
            VariableSpec(name="x2", domain_size=2),
        ],
        factors=[
            FactorSpec(
                name="f1",
                neighbors=["x1", "x2"],
                cost_table=[[0.0, 1.0], [1.0, 0.0]],
            )
        ],
        config=config,
    )

    snapshots = run_simulation(spec)
    assert len(snapshots) == 1
    assert "f1->x1" in snapshots[0].R
