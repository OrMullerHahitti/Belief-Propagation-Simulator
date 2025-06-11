\
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx
from IPython.display import display, clear_output
import ipywidgets as widgets

# Heuristic constants for "simple graph"
MAX_VARS_SIMPLE = 6
MAX_FACTORS_SIMPLE = 6
MAX_DOMAIN_SIMPLE = 4
MAX_FACTOR_DEGREE_FOR_TABLE = 2

class EnhancedBPVisualizer:
    """Enhanced step-by-step BP visualizer with detailed table views."""

    def __init__(self, engine):
        self.engine = engine
        self.graph = engine.graph
        self.step_count = 0
        
        self.is_simple_graph = self._check_graph_simplicity()

        # Create layout
        self.pos = self._get_layout()

        # UI elements
        self.output = widgets.Output()
        self.btn_step = widgets.Button(description="Next Step", button_style='primary')
        self.btn_reset = widgets.Button(description="Reset", button_style='danger')
        self.info_label = widgets.Label(value=f"Step: 0 | Simple Graph: {self.is_simple_graph}")

        self.btn_step.on_click(self.on_step)
        self.btn_reset.on_click(self.on_reset)

        self.controls = widgets.VBox([
            widgets.HBox([self.btn_step, self.btn_reset, self.info_label]),
        ])
        
        # Store highlighted cells info: {factor_name: {message_to_var_name: [(row,col), ...]}}
        self.highlighted_cells_info = {}


    def _get_layout(self):
        # Filter out nodes that are not in the graph G before passing to layout
        nodes_in_g = [n for n in self.graph.variables if n in self.graph.G]
        if not nodes_in_g and self.graph.variables: # if all variables are somehow not in G, pick first one if any
            nodes_in_g = [self.graph.variables[0]] if self.graph.variables[0] in self.graph.G else []

        if not self.graph.G or not self.graph.variables or not nodes_in_g:
             # Handle empty or improperly initialized graph for layout
            return {}

        # Ensure all variable nodes exist in G for bipartite layout
        valid_variable_nodes = [v for v in self.graph.variables if v in self.graph.G]
        
        if not valid_variable_nodes: # If no variable nodes are in G, cannot compute bipartite layout
            # Fallback to a generic layout if bipartite is not possible
            return nx.spring_layout(self.graph.G, scale=2.0, center=[0,0], seed=42)

        try:
            pos = nx.bipartite_layout(
                self.graph.G,
                nodes=valid_variable_nodes, # Use only valid variable nodes for one set of bipartite
                scale=2.0, # Increased scale
                center=[0, 0]
            )
            # Spread out nodes more
            for k in pos:
                pos[k] = pos[k] * np.array([1.5, 1.2]) # Stretch x and y
            return pos

        except Exception as e:
            # Fallback to a spring layout if bipartite layout fails
            # print(f"Bipartite layout failed: {e}. Falling back to spring layout.")
            return nx.spring_layout(self.graph.G, scale=2.0, center=[0,0], seed=42)


    def _check_graph_simplicity(self):
        if not self.graph or not self.graph.G:
            return False
        num_vars = len(self.graph.variables)
        num_factors = len(self.graph.factors)
        
        max_domain = 0
        for var in self.graph.variables:
            if hasattr(var, 'domain_size'): # Assuming domain_size attribute
                 max_domain = max(max_domain, var.domain_size)
            elif hasattr(var, 'domain'): # Assuming domain is an int for size
                 max_domain = max(max_domain, var.domain)


        max_degree = 0
        for factor in self.graph.factors:
            degree = self.graph.G.degree(factor)
            max_degree = max(max_degree, degree)
            if degree > MAX_FACTOR_DEGREE_FOR_TABLE: # If any factor is too complex for table display
                 # This check is more about individual factor displayability
                 pass


        return (num_vars <= MAX_VARS_SIMPLE and
                num_factors <= MAX_FACTORS_SIMPLE and
                max_domain <= MAX_DOMAIN_SIMPLE)

    def _get_node_display_name(self, node):
        return node.name if hasattr(node, 'name') else str(node)

    def visualize_state(self):
        with self.output:
            clear_output(wait=True)
            
            fig_height = 8
            if self.is_simple_graph:
                fig_height = 10 # More space for tables

            # One axis for graph, one for belief tables
            fig, (ax_graph, ax_beliefs) = plt.subplots(2, 1, figsize=(14, fig_height + 4), gridspec_kw={'height_ratios': [3, 1]})
            fig.set_constrained_layout(True)

            self.draw_graph_representation(ax_graph)
            self.draw_belief_tables(ax_beliefs)

            plt.show()

    def draw_graph_representation(self, ax):
        ax.clear()
        ax.axis('off')
        ax.set_title(f"Factor Graph - Step {self.step_count}", fontsize=14)
        
        if not self.graph.G or not self.pos:
            ax.text(0.5, 0.5, "Graph not available or layout failed.", ha='center', va='center')
            return

        # Draw edges first
        nx.draw_networkx_edges(self.graph.G, self.pos, ax=ax, alpha=0.3, width=1.5)

        # Draw variable nodes
        var_nodes = self.graph.variables
        var_labels = {n: self._get_node_display_name(n) for n in var_nodes}
        nx.draw_networkx_nodes(self.graph.G, self.pos, nodelist=var_nodes,
                               node_shape='o', node_color='lightblue', node_size=1500, ax=ax, alpha=0.8)
        nx.draw_networkx_labels(self.graph.G, self.pos, labels=var_labels, ax=ax, font_size=9)

        # Draw factor nodes
        for factor in self.graph.factors:
            if factor not in self.graph.G: continue # Skip if factor somehow not in G

            factor_pos = self.pos.get(factor)
            if factor_pos is None: continue # Skip if no position

            is_factor_simple_enough_for_table = (hasattr(factor, 'cost_table') and
                                                 factor.cost_table is not None and
                                                 factor.cost_table.ndim <= MAX_FACTOR_DEGREE_FOR_TABLE and
                                                 self.graph.G.degree(factor) <= MAX_FACTOR_DEGREE_FOR_TABLE and
                                                 all(v.domain <= MAX_DOMAIN_SIMPLE for v in self.graph.G.neighbors(factor) if hasattr(v, 'domain')))


            if self.is_simple_graph and is_factor_simple_enough_for_table:
                self._draw_factor_as_table(ax, factor, factor_pos)
            else:
                nx.draw_networkx_nodes(self.graph.G, self.pos, nodelist=[factor],
                                       node_shape='s', node_color='lightgreen', node_size=1500, ax=ax, alpha=0.8)
                nx.draw_networkx_labels(self.graph.G, self.pos, labels={factor: self._get_node_display_name(factor)}, ax=ax, font_size=9)
        
        # Draw messages
        self._draw_messages(ax)
        ax.axis('equal')


    def _draw_factor_as_table(self, ax, factor, factor_pos):
        cost_table_data = factor.cost_table
        factor_name = self._get_node_display_name(factor)

        if cost_table_data is None: return

        # Determine connected variables for row/col labels
        connected_vars = list(self.graph.G.neighbors(factor))
        
        row_labels = []
        col_labels = []
        table_data_str = []

        if cost_table_data.ndim == 1:
            # For 1D cost table (factor connected to one variable)
            var = connected_vars[0] if connected_vars else None
            if var:
                row_labels = [f"{self._get_node_display_name(var)}_{i}" for i in range(cost_table_data.shape[0])]
            else: # Fallback if var name not found
                row_labels = [str(i) for i in range(cost_table_data.shape[0])]
            col_labels = ["Cost"]
            table_data_str = [[f"{val:.2f}"] for val in cost_table_data]

        elif cost_table_data.ndim == 2:
            # For 2D cost table
            var1 = connected_vars[0] if len(connected_vars) > 0 else None
            var2 = connected_vars[1] if len(connected_vars) > 1 else None

            # Use factor.connection_number if available, otherwise assume order
            var_names_ordered = ["VarA", "VarB"] # Fallback names
            if hasattr(factor, 'connection_number'):
                # Sort var names by their dimension index in cost_table
                sorted_conn_vars = sorted(factor.connection_number.items(), key=lambda item: item[1])
                var_names_ordered = [name for name, _ in sorted_conn_vars]


            row_var_name = self._get_node_display_name(var1) if var1 else var_names_ordered[0]
            col_var_name = self._get_node_display_name(var2) if var2 else var_names_ordered[1]
            
            row_labels = [f"{row_var_name}_{i}" for i in range(cost_table_data.shape[0])]
            col_labels = [f"{col_var_name}_{j}" for j in range(cost_table_data.shape[1])]
            table_data_str = [[f"{val:.2f}" for val in row] for row in cost_table_data]
        else: # Not 1D or 2D
            nx.draw_networkx_nodes(self.graph.G, self.pos, nodelist=[factor],
                                   node_shape='s', node_color='lightcoral', node_size=1500, ax=ax, alpha=0.8) # Indicate complex factor
            nx.draw_networkx_labels(self.graph.G, self.pos, labels={factor: factor_name}, ax=ax, font_size=9)
            return

        # Create the table
        # Note: Matplotlib's table is drawn in data coordinates if no transform is given.
        # We need to position it carefully. This is complex.
        # A simpler approach for now: draw a box and text, not a full matplotlib table object in graph.
        # For true table drawing, one would typically use a dedicated axis or complex transforms.
        
        # Simplified representation: a rectangle with the factor name
        rect_width = 0.2 * len(col_labels) + 0.2
        rect_height = 0.1 * len(row_labels) + 0.1
        rect = patches.Rectangle((factor_pos[0] - rect_width/2, factor_pos[1] - rect_height/2), 
                                 rect_width, rect_height, linewidth=1, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.8)
        ax.add_patch(rect)
        ax.text(factor_pos[0], factor_pos[1] + rect_height/2 + 0.05, factor_name, 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add cell highlighting if info is available
        # This part is complex due to mapping table cells to screen space and matching with message computation.
        # The self.highlighted_cells_info should be populated during/after engine.step()
        # For now, this is a placeholder for the highlighting logic on the table drawn on graph.
        # A more robust way is to show this detail in a separate panel.

        # Let's try to draw a mini-table using text annotations if very small
        if cost_table_data.ndim <= 2 and cost_table_data.size <= 9: # e.g. up to 3x3
            cell_width = rect_width / (len(col_labels) or 1)
            cell_height = rect_height / (len(row_labels) or 1)
            
            current_highlights = self.highlighted_cells_info.get(self._get_node_display_name(factor), {})

            for r, row_data in enumerate(table_data_str):
                for c, cell_val_str in enumerate(row_data):
                    cell_x = factor_pos[0] - rect_width/2 + (c + 0.5) * cell_width
                    cell_y = factor_pos[1] + rect_height/2 - (r + 0.5) * cell_height
                    
                    cell_color = 'black'
                    bg_props = None

                    # Check if this cell should be highlighted for any outgoing message
                    is_highlighted = False
                    for msg_to_var, highlighted_coords_list in current_highlights.items():
                        if (r,c) in highlighted_coords_list:
                            is_highlighted = True
                            break
                    
                    if is_highlighted:
                        bg_props = dict(boxstyle="square,pad=0.3", facecolor='yellow', alpha=0.7, ec='orange')

                    ax.text(cell_x, cell_y, cell_val_str, ha='center', va='center', fontsize=7, color=cell_color, bbox=bg_props)


    def _draw_messages(self, ax):
        for node in self.graph.G.nodes():
            if not hasattr(node, 'mailer') or not hasattr(node.mailer, 'outbox'):
                continue
            
            for msg in node.mailer.outbox:
                if not hasattr(msg, 'sender') or not hasattr(msg, 'recipient') or \
                   msg.sender not in self.pos or msg.recipient not in self.pos:
                    continue

                start_node = msg.sender
                end_node = msg.recipient
                
                start_pos = np.array(self.pos[start_node])
                end_pos = np.array(self.pos[end_node])

                # Adjust start/end points if one is a table-factor
                # This is a rough adjustment; precise connection to table borders is harder.
                if self.is_simple_graph and isinstance(start_node, tuple(self.graph.factors)) and \
                   hasattr(start_node, 'cost_table') and start_node.cost_table.ndim <= MAX_FACTOR_DEGREE_FOR_TABLE:
                    # Heuristic: aim for edge of a conceptual bounding box
                    direction = (end_pos - start_pos)
                    direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
                    # Assume table drawn for factor is roughly 0.2 units wide/high
                    start_pos = start_pos + direction_norm * 0.2 


                if self.is_simple_graph and isinstance(end_node, tuple(self.graph.factors)) and \
                   hasattr(end_node, 'cost_table') and end_node.cost_table.ndim <= MAX_FACTOR_DEGREE_FOR_TABLE:
                    direction = (start_pos - end_pos)
                    direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
                    end_pos = end_pos + direction_norm * 0.2


                ax.annotate("", xy=end_pos, xytext=start_pos,
                            arrowprops=dict(arrowstyle="->", lw=1.5, color='red', shrinkA=15, shrinkB=15,
                                            connectionstyle="arc3,rad=0.1")) # Added shrink and arc

                mid_point = (start_pos + end_pos) / 2
                msg_text_data = msg.data
                if isinstance(msg_text_data, np.ndarray):
                    msg_text = f"{msg_text_data[:2].round(1)}..." if len(msg_text_data) > 3 else str(msg_text_data.round(1))
                else: # if it's a scalar or other type
                    msg_text = str(msg_text_data)

                ax.text(mid_point[0], mid_point[1], msg_text, fontsize=7, color='darkred',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, ec='gray'))


    def draw_belief_tables(self, ax):
        ax.clear()
        ax.axis('off') # No frame, ticks, etc.
        
        var_agents = self.graph.variables
        if not var_agents:
            ax.text(0.5, 0.95, "No variables to display beliefs for.", ha='center', va='top')
            return

        ax.set_title("Variable Beliefs", fontsize=14, loc='left', pad=10)

        num_vars = len(var_agents)
        # Attempt to lay out tables in a grid
        # Max 3 tables per row for readability
        max_cols = 3 
        num_cols = min(num_vars, max_cols)
        num_rows = (num_vars + num_cols - 1) // num_cols

        table_width = 1.0 / num_cols
        table_height = 1.0 / num_rows
        
        for i, var in enumerate(var_agents):
            if not hasattr(var, 'belief') or var.belief is None:
                continue

            row_idx = i // num_cols
            col_idx = i % num_cols

            belief_data = var.belief
            domain_size = len(belief_data)
            
            cell_text = [["Value", "Belief"]]
            cell_text.extend([[str(j), f"{belief_data[j]:.3f}"] for j in range(domain_size)])
            
            # Position for the table title (variable name)
            title_x = col_idx * table_width + table_width / 2
            title_y = 1.0 - (row_idx * table_height) - 0.02 # Slightly above table

            ax.text(title_x, title_y, self._get_node_display_name(var), ha='center', va='top', fontsize=10, fontweight='bold', transform=ax.transAxes)

            # Create the table itself
            # The position [left, bottom, width, height] for the table within the axis `ax`
            # These are relative to the axis `ax` if `ax.transAxes` is used implicitly by table.
            # Matplotlib tables are tricky with subplot layouts.
            # For simplicity, we'll use ax.table which places it within the axis.
            # We need to calculate absolute positions if using fig.add_axes for each table.
            
            # This places one table; for multiple, need to adjust x,y offsets.
            # The table will be placed relative to (0,0) of the axis `ax`.
            # We need to manually shift it.
            # A simpler way: create sub-axes for each table.
            
            # Let's use a simpler text representation if sub-axes are too complex for now.
            # For now, just print text for each var belief. A proper table layout is more involved.
            
            table_content = f"{self._get_node_display_name(var)}:\n"
            max_val_idx = np.argmax(belief_data)
            for j in range(domain_size):
                is_max = " (max)" if j == max_val_idx else ""
                table_content += f"  {j}: {belief_data[j]:.3f}{is_max}\n"
            
            # Position text block
            text_x = (col_idx + 0.05) * table_width 
            text_y = 1.0 - (row_idx + 0.1) * table_height # Adjust y to be below title
            ax.text(text_x, text_y, table_content, ha='left', va='top', fontsize=8, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue", alpha=0.7, ec='lightsteelblue'))


    def _update_highlight_info(self):
        self.highlighted_cells_info.clear()
        if not self.is_simple_graph:
            return

        for factor in self.graph.factors:
            if not (hasattr(factor, 'cost_table') and factor.cost_table is not None and
                    factor.cost_table.ndim <= MAX_FACTOR_DEGREE_FOR_TABLE and
                    hasattr(factor, 'mailer') and hasattr(factor.mailer, 'outbox')):
                continue

            factor_name = self._get_node_display_name(factor)
            self.highlighted_cells_info[factor_name] = {}

            for msg in factor.mailer.outbox: # These are R-messages: factor -> variable
                recipient_var = msg.recipient
                recipient_var_name = self._get_node_display_name(recipient_var)
                
                # This is where the complex logic to find *which* original cost table cells
                # contributed to this message would go. It requires knowledge of the
                # BP computator's internal workings (e.g., min_sum).

                # Simplified placeholder: if a message is sent, we'd need to determine
                # the specific cells. For now, this is not fully implemented due to complexity
                # of introspecting the computator.
                # Example: if R_message[k] was min_j (Cost[k,j] + Q[j]), highlight (k,j_min).
                
                # This requires:
                # 1. Factor's cost table.
                # 2. Incoming Q messages to the factor (from factor.mailer.inbox, excluding Q from recipient_var).
                # 3. The recipient_var to know which dimension of cost_table corresponds to it.
                # 4. The computator logic (e.g., min_sum).

                # For a factor f connected to v_target and v_other (2D cost table C[x_target, x_other])
                # R_{f->v_target}(x_target) = min_{x_other} ( C(x_target, x_other) + Q_{v_other->f}(x_other) )
                # We need to find, for each x_target_i, which x_other_j minimized the sum.
                
                # This is a complex calculation that ideally should be provided by the engine or computator
                # or re-calculated here if necessary.
                # For now, let's assume no specific highlighting is computed yet.
                self.highlighted_cells_info[factor_name][recipient_var_name] = [] # Empty list of (r,c) tuples


    def on_step(self, b):
        self.engine.step(self.step_count) # engine.step should update messages in mailers
        self.step_count += 1
        self.info_label.value = f"Step: {self.step_count} | Simple Graph: {self.is_simple_graph}"
        self._update_highlight_info() # Update highlights based on new messages
        self.visualize_state()

    def on_reset(self, b):
        self.step_count = 0
        self.info_label.value = f"Step: 0 | Simple Graph: {self.is_simple_graph}"
        
        # Reset engine state (clear messages, reinitialize if method exists)
        if hasattr(self.engine, 'reset'):
            self.engine.reset() 
        else: # Manual reset based on SimpleBPVisualizer
            for node in self.graph.G.nodes():
                if hasattr(node, 'empty_mailbox'): node.empty_mailbox()
                if hasattr(node, 'empty_outgoing'): node.empty_outgoing()
                if hasattr(node, 'mailer'):
                    if hasattr(node.mailer, '_incoming'): node.mailer._incoming.clear()
                    if hasattr(node.mailer, '_outgoing'): node.mailer._outgoing.clear()
                    if hasattr(node.mailer, 'outbox'): node.mailer.outbox.clear() # Clear outbox too
                    if hasattr(node.mailer, 'inbox'): node.mailer.inbox.clear()   # Clear inbox

            # Reinitialize first messages for variables if applicable
            for var in self.graph.variables:
                if hasattr(var, 'mailer') and hasattr(var.mailer, 'set_first_message'):
                    for neighbor in self.graph.G.neighbors(var):
                         if hasattr(neighbor, 'is_factor_agent') and neighbor.is_factor_agent: # Check if neighbor is factor
                            var.mailer.set_first_message(var, neighbor) # Assuming this method exists

        self.highlighted_cells_info.clear()
        self.visualize_state()

    def run(self):
        display(self.controls)
        display(self.output)
        self.visualize_state() # Initial visualization

# Example usage (for testing, normally called from notebook)
def demo_enhanced_visualization():
    import sys # Add sys import
    import os # Add os import
    # Add workspace root to path for imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from bp_base.factor_graph import FactorGraph
    from bp_base.bp_computators import MinSumComputator
    from bp_base.bp_engine_base import BPEngine
    from base_all.agents import VariableAgent, FactorAgent # Ensure these are correctly importable

    # Create a simple graph
    v0 = VariableAgent("v0", domain=2)
    v1 = VariableAgent("v1", domain=2)
    f0 = FactorAgent("f0", domain=2, ct_creation_func=lambda n,d: np.array([[0.1,1.0],[1.0,0.2]]), param={}) # Modified costs for clarity
    
    variables = [v0, v1]
    factors = [f0]
    edges = {f0: [v0,v1]}

    fg = FactorGraph(variables, factors, edges)
    engine = BPEngine(fg, computator=MinSumComputator())

    viz = EnhancedBPVisualizer(engine)
    viz.run()

if __name__ == "__main__":
    # This demo might fail if imports are not set up for direct script execution
    print("Running demo visualization... (ensure imports are correct)")
    demo_enhanced_visualization()
    # pass
