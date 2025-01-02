# nodes.py


from abstract_base.Node import Node


class VariableNode(Node):
    def __init__(self, name, domain):
        super().__init__(name)
        self.domain = domain





class FactorNode(Node):
    def __init__(self, name, potential_table, var_names):
        super().__init__(name)
        self.potential_table = potential_table  # numpy array
        self.var_names = var_names


