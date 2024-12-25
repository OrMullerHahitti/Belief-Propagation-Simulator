import pandas as pd
import numpy as np
from numpy.ma.core import min_val


def num_to_letter(n: int) -> str:
    # 1 -> 'a', 2 -> 'b', ...
    return chr(ord('a') + (n - 1))

def letter_to_num(c: str) -> int:
    # 'a' -> 1, 'b' -> 2, ...
    return ord(c) - ord('a') + 1


class CostTable:
    def __init__(self, data:pd.DataFrame|None = None, domain_size=None):
        '''
        data: optional DataFrame of cost values
        domain_size: if data is None, create a random domain_size x domain_size DataFrame


        '''
        if data is None and domain_size is not None:
            letters = [chr(ord('a') + i) for i in range(domain_size)]
            data = np.random.rand(domain_size, domain_size)
            self._df = pd.DataFrame(data, index=letters, columns=letters)
        elif data is not None:
            self._df = pd.DataFrame(data)
        else:
            raise ValueError("Must provide either data or domain size")
    @staticmethod
    def to_letter(self, x):
        if isinstance(x, int):
            return chr(ord('a') + x-1)
        elif isinstance(x, str):
            return x
        else:
            raise TypeError("Index must be int or str")

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(row_key, int):
                row_key = self.to_letter(row_key)
            if isinstance(col_key, int):
                col_key = self.to_letter(col_key)
            return self._df.loc[row_key, col_key]
        elif isinstance(key, int):
            key = self.to_letter(key)
            return self._df[key]
        else:
            return self._df[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(row_key, int):
                row_key = self.to_letter(row_key)
            if isinstance(col_key, int):
                col_key = self.to_letter(col_key)
            self._df.loc[row_key, col_key] = value
        elif isinstance(key, int):
            key = self.to_letter(key)
            self._df[key] = value
        else:
            self._df[key] = value

    def __str__(self):
        return str(self._df)

    def to_dataframe(self):
        return self._df.copy()
class FactorNode:
    def __init__(self, name: str, var_nodes, cost_table: CostTable = None):
        """
        var_nodes: tuple/list of the two VariableNodes this factor connects
        cost_table: optional CostTable. If None, create automatically.
        """
        self.name = name
        self.var_nodes = var_nodes
        if cost_table is None:
            # Assuming both var_nodes have the same domain_size
            domain_size = var_nodes[0].domain_size
            self.cost_table = CostTable(domain_size=domain_size)
        else:
            self.cost_table = cost_table
    def __getitem__(self, index):
        return self.cost_table[index]


        # Optionally, you may want to keep track of dimension mapping
        # e.g., which variable corresponds to rows or columns.




class Q:
    def __init__(self, domain_size):
        self.domain_size = domain_size
        # Random initialization of Q values
        letters = [num_to_letter(i) for i in range(1, domain_size + 1)]
        random_values = np.random.rand(domain_size)  # random array of length domain_size
        self.values = {letter: val for letter, val in zip(letters, random_values)}

    def __getitem__(self, index):
        if isinstance(index, int):
            letter = num_to_letter(index)
            return self.values[letter]
        elif isinstance(index, str):
            return self.values[index]
        else:
            raise TypeError("Index must be int or str")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            letter = num_to_letter(index)
            self.values[letter] = value
        elif isinstance(index, str):
            self.values[index] = value
        else:
            raise TypeError("Index must be int or str")

    def __str__(self):
        return str(self.values)


class R:
    def __init__(self, q: Q, cost_table: CostTable, direction: str = 'rows'):
        """
        direction: 'rows' or 'columns', indicating how to align Q.
        If 'rows', Q[i] is added to the i-th row.
        If 'columns', Q[i] is added to the i-th column.
        """
        df = cost_table.to_dataframe().copy()

        if direction == 'rows':
            # Add Q to each row
            for i in range(1, q.domain_size+1):
                letter = num_to_letter(i)
                df.loc[letter, :] = df.loc[letter, :] + q[i]
            # Minimize across columns
            self.mins = {
                num_to_letter(i): df.loc[num_to_letter(i), :].min()
                for i in range(1, q.domain_size+1)
            }

        elif direction == 'columns':
            # Add Q to each column
            for i in range(1, q.domain_size+1):
                letter = num_to_letter(i)
                df.loc[:, letter] = df.loc[:, letter] + q[i]
            # Minimize across rows
            self.mins = {
                num_to_letter(i): df.loc[:, num_to_letter(i)].min()
                for i in range(1, q.domain_size+1)
            }

        self.after = CostTable(df=df)
        self.minimizer = min(self.mins.values())

    def __getitem__(self, index):
        if isinstance(index, int):
            letter = num_to_letter(index)
            return self.mins[letter]
        elif isinstance(index, str):
            return self.mins[index]
        else:
            raise TypeError("Index must be int or str")

    def __str__(self):
        return f"R mins: {self.mins}\nR.after:\n{self.after}\nMinimizer: {self.minimizer}"


class VariableNode:
    def __init__(self, name: str, domain_size: int):
        self.name = name
        self.domain_size = domain_size
        self.connected_factors = []
        self.sent_messages = {}    # {factor_name: [Q_message_history]}
        self.received_messages = {} # {factor_name: [R_message_history]}

    def connect_factor(self, factor_node: FactorNode):
        self.connected_factors.append(factor_node)
        self.sent_messages[factor_node.name] = []
        self.received_messages[factor_node.name] = []

    def send_message(self, factor_node: FactorNode, q_message: Q):
        self.sent_messages[factor_node.name].append(q_message)

    def receive_message(self, factor_node: FactorNode, r_message: R):
        self.received_messages[factor_node.name].append(r_message)




