import os
import pickle
import numpy as np
import pytest

from bp_base.agents import VariableAgent, FactorAgent
from bp_base.factor_graph import FactorGraph
from utils.examples import create_factor_graph
from utils.create_factor_graphs_from_config import FactorGraphBuilder
from utils.create_factor_graph_config import ConfigCreator
from utils.path_utils import load_pickle


def test_factor_graph_direct_creation():
    v1 = VariableAgent(name="v1", domain=2)
    v2 = VariableAgent(name="v2", domain=2)
    ct_func = lambda n, d, **k: np.zeros((d, d))
    f = FactorAgent(name="f12", domain=2, ct_creation_func=ct_func, param={})
    fg = FactorGraph([v1, v2], [f], {f: [v1, v2]})

    assert len(fg.variables) == 2
    assert len(fg.factors) == 1
    assert fg.G.has_edge(f, v1)
    assert f.cost_table.shape == (2, 2)


def test_create_factor_graph_cycle():
    fg = create_factor_graph(graph_type="cycle", num_vars=3, domain_size=2)
    assert len(fg.variables) == 3
    assert len(fg.factors) == 3
    # for a cycle of 3 variables there should be 6 edges (factor-variable)
    assert len(fg.G.edges()) == 6


def test_factor_graph_builder_cycle(tmp_path):
    creator = ConfigCreator(base_dir=tmp_path)
    cfg = creator.create_graph_config(
        graph_type="cycle",
        num_variables=3,
        domain_size=2,
        ct_factory="random_int",
        ct_params={"low": 0, "high": 5},
        density=1.0,
    )
    builder = FactorGraphBuilder(output_dir=tmp_path)
    fg_path = builder.build_and_save(cfg)
    assert os.path.exists(fg_path)
    fg = builder.load_graph(fg_path)
    assert isinstance(fg, FactorGraph)
    assert len(fg.variables) == 3


def test_factor_graph_pickle_load(tmp_path):
    fg = create_factor_graph(graph_type="cycle", num_vars=3, domain_size=2)
    pkl = tmp_path / "fg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(fg, f)
    loaded = load_pickle(pkl)
    assert isinstance(loaded, FactorGraph)
    assert len(loaded.variables) == 3
