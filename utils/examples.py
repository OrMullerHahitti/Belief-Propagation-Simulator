# create simple factorgraph cycle domain 3 3 variables
from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from typing import List, Tuple, Iterable
import networkx as nx
import os
from pathlib import Path

from utils.create_factor_graphs_from_config import FactorGraphBuilder
from utils.path_utils import find_project_root
import sys


# create simple factorgraph cycle domain 3 3 variables
def create_simple_factor_graph_cycle() -> FactorGraph:
    project_root = find_project_root()
    sys.path.append(str(project_root))
    cfg = os.path.join(
        project_root, "configs", "factor_graph_configs", "simple_example.pkl"
    )

    fg = FactorGraphBuilder().build_and_return(cfg)

    return fg


fg = create_simple_factor_graph_cycle()
