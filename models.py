import pandas as pd
import numpy as np

def num_to_letter(n: int) -> str:
    # 1 -> 'a', 2 -> 'b', ...
    return chr(ord('a') + (n - 1))

def letter_to_num(c: str) -> int:
    # 'a' -> 1, 'b' -> 2, ...
    return ord(c) - ord('a') + 1


class CostTableIndexer:
    def __init__(self, ct, row_index):
        self.ct = ct  # reference to CostTable instance
        self.row = row_index

    def __getitem__(self, col_index):
        # Convert row and col indexes (could be letter or number)
        row_letter = self.ct._to_letter(self.row)
        col_letter = self.ct._to_letter(col_index)
        return self.ct._df.loc[row_letter, col_letter]


class CostTable:
    def __init__(self, df=None, domain_size=None):
        if df is not None:
            self._df = df.copy()
        else:
            if domain_size is None:
                raise ValueError("Must provide either df or domain size")
            letters = [num_to_letter(i) for i in range(1, domain_size+1)]
            # Create random NxN cost table
            data = np.random.rand(domain_size, domain_size)  # random NxN in [0,1)
            self._df = pd.DataFrame(data, index=letters, columns=letters)

    def __getitem__(self, row_index):
        # This returns a row accessor
        return CostTableIndexer(self, row_index)

    def _to_letter(self, x):
        if isinstance(x, int):
            return num_to_letter(x)
        elif isinstance(x, str):
            # assume already a letter
            return x
        else:
            raise TypeError("Index must be int or str")

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
        letters = [num_to_letter(i) for i in range(1, domain_size+1)]
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


#example usage:
# Step 1: Create Variable Nodes
X1 = VariableNode(name="X1", domain_size=3)
X2 = VariableNode(name="X2", domain_size=3)

# Step 2: Create Factor Node
F12 = FactorNode(name="F12", var_nodes=(X1, X2))

# Step 3: Print the randomly generated cost table of F12
print(f"CostTable of F12:\n{F12.cost_table}\n")

# Step 4: Create a random Q message from X1 to F12
Q_X1_F12 = Q(domain_size=X1.domain_size)
print(f"Random Q message from X1 to F12:\n{Q_X1_F12}\n")

# Step 5: Compute the R message sent from F12 back to X1
R_F12_X1 = R(q=Q_X1_F12, cost_table=F12.cost_table, direction="columns")  # Assume X1 corresponds to columns
print(f"R message from F12 to X1:\n{R_F12_X1}\n")

# Step 6: Display R.after for debugging
print(f"Processed CostTable (R.after):\n{R_F12_X1.after}\n")
print(f"Minimum values in R (mins): {R_F12_X1.mins}")
print(f"Global minimizer in R: {R_F12_X1.minimizer}")

