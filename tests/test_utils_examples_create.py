import numpy as np
import pytest

from propflow.configs import global_config_mapping as gcm
from propflow.utils import FGBuilder
from propflow.utils.examples import create_factor_graph
from propflow.utils.create.create_factor_graph_config import ConfigCreator
from propflow.utils.create.create_factor_graphs_from_config import FactorGraphBuilder

np.random.seed(42)


def test_create_factor_graph_cycle(monkeypatch):
    monkeypatch.setattr("propflow.utils.examples.build_cycle_graph", FGBuilder.build_cycle_graph)
    graph = create_factor_graph(
        graph_type="cycle",
        num_vars=4,
        domain_size=2,
        ct_params={"low": 0, "high": 3},
    )
    assert len(graph.variables) == 4
    assert graph.diameter >= 2


def test_create_factor_graph_unknown_type(monkeypatch):
    monkeypatch.setattr("propflow.utils.examples.build_cycle_graph", FGBuilder.build_cycle_graph)
    with pytest.raises(ValueError):
        create_factor_graph(graph_type="unknown")


def test_config_creator_and_factor_graph_builder(tmp_path, monkeypatch):
    monkeypatch.setitem(gcm.GRAPH_TYPES, "test-cycle", "propflow.utils.fg_utils.FGBuilder.build_cycle_graph")
    creator = ConfigCreator(base_dir=tmp_path)
    cfg_path = creator.create_graph_config(
        graph_type="test-cycle",
        num_variables=3,
        domain_size=3,
        ct_factory="random_int",
        ct_params={"low": 1, "high": 4},
    )
    builder = FactorGraphBuilder(output_dir=tmp_path / "graphs")
    graph_path = builder.build_and_save(cfg_path)
    assert graph_path.exists()
    loaded = FactorGraphBuilder.load_graph(graph_path)
    assert len(loaded.variables) == 3
    assert len(loaded.factors) == 3
