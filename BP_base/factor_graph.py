# implementation of factor graph given everything in BP_base
from abc import ABC
from typing import List, Dict

from networkx import Graph
from networkx.linalg.spectrum import adjacency_spectrum

from BP_base.agents import VariableNode, FactorNode
from DCOP_base import Agent


def _strip_name(agent:Agent)->str:
    return agent.name[1:]

def _create_edges(variable_li:List[VariableNode],factor_li:List[FactorNode])->Dict[FactorNode,List[VariableNode]]:#adj list
    adjacency = {}
    for f in factor_li:
        adjacency[f]=[v for v in variable_li if _strip_name(f) in _strip_name(v)]
    return adjacency







class FactorGraph(Graph):
    def __init__(self,variable_li:List[VariableNode],factor_li:List[FactorNode]):
        self.add_nodes_from(variable_li)
        self.add_nodes_from(factor_li)
        adjacency= _create_edges(variable_li,factor_li)
        self.add_edges_from((f, v) for f, neighbors in adjacency.items() for v in neighbors)



