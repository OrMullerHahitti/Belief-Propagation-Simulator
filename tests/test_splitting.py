import os

import pytest
import numpy as np
from bp_base.agents import VariableAgent, FactorAgent, BPAgent  # BPAgent for mailer
from bp_base.components import Message
from bp_base.computators import MaxSumComputator
from bp_base.factor_graph import FactorGraph
from configs.global_config_mapping import PROJECT_ROOT
from configs.loggers import Logger
from utils.path_utils import load_pickle
from utils.splitting import split_all_factors

logger = Logger(__name__, file=False)
logger.setLevel(10)

@pytest.fixture
def factor_graph():
    pickle_path = os.path.join(
        PROJECT_ROOT,
        "configs",
        "factor_graphs",
        "test-factor.pkl",
    )
    fg = load_pickle(pickle_path)
    return fg
def test_splitting(factor_graph,p:float=0.4):
    fg:FactorGraph = factor_graph
    num_of_factors = len(fg.get_factor_agents())
    split_all_factors(fg,p=p)
    logger.debug(f"Number of factors before splitting: {num_of_factors}")
    assert len(fg.get_factor_agents()) == num_of_factors*2 , "did not double the factors"

