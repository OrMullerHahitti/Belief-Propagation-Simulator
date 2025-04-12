from dataclasses import dataclass
from typing import Dict, List

from bp_base.factor_graph import FactorGraph
from policies.abstract import MessagePolicy, FactorPolicy
from policies.pipelines import message_pipline

GraphConfig:Dict[str,str]

@dataclass
class BPConfig:
    """
    Configuration class for the BPBase project.
    This class holds configuration options and settings.
    """

    # Simulation parameters
    def __init__(self, factorgraphConfig:GraphConfig,message_pipline:List[MessagePolicy],factor_pipeline:List[FactorPolicy]):
        pass

