import dash
from dash import dcc, html
import dash_cytoscape as cyto
import networkx as nx
import random
from dash.dependencies import Input, Output, State

# Create a sample graph
G = nx.Graph()
nodes = list(range(1, 6))
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]

# Add nodes and edges to the graph
for node in nodes:
    G.add_node(node)

for edge in edges:
    G.add_edge(*edge)


# Convert NetworkX graph to Dash Cytoscape format
def generate_elements(G):
    elements = [{"data": {"id": str(node), "label": str(node)}} for node in G.nodes()]
    elements += [
        {"data": {"source": str(edge[0]), "target": str(edge[1])}} for edge in G.edges()
    ]
    return elements


# Initialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H2("Interactive Graph Visualization"),
        cyto.Cytoscape(
            id="graph",
            layout={"name": "cose"},  # Layout type
            style={"width": "100%", "height": "500px"},
            elements=generate_elements(G),
            stylesheet=[
                {
                    "selector": "node",
                    "style": {
                        "content": "data(label)",
                        "background-color": "#3498db",
                        "width": 25,
                        "height": 25,
                    },
                },
                {"selector": "edge", "style": {"line-color": "#95a5a6", "width": 2}},
            ],
        ),
        html.Button("Add Node", id="add-node-btn", n_clicks=0),
        html.Button("Add Edge", id="add-edge-btn", n_clicks=0),
    ]
)


@app.callback(
    Output("graph", "elements"),
    Input("add-node-btn", "n_clicks"),
    Input("add-edge-btn", "n_clicks"),
    State("graph", "elements"),
)
def update_graph(n_nodes, n_edges, elements):
    G_dynamic = nx.Graph()

    # Convert elements back to NetworkX format
    for element in elements:
        if "source" in element["data"]:  # It's an edge
            G_dynamic.add_edge(
                int(element["data"]["source"]), int(element["data"]["target"])
            )
        else:  # It's a node
            G_dynamic.add_node(int(element["data"]["id"]))

    # Add a new node if the button is clicked
    if n_nodes > len(G_dynamic.nodes):
        new_node = max(G_dynamic.nodes) + 1 if G_dynamic.nodes else 1
        G_dynamic.add_node(new_node)

    # Add a random edge if the button is clicked
    if n_edges > len(G_dynamic.edges):
        node_list = list(G_dynamic.nodes)
        if len(node_list) > 1:
            new_edge = (random.choice(node_list), random.choice(node_list))
            while new_edge[0] == new_edge[1] or new_edge in G_dynamic.edges:
                new_edge = (random.choice(node_list), random.choice(node_list))
            G_dynamic.add_edge(*new_edge)

    return generate_elements(G_dynamic)


if __name__ == "__main__":
    app.run_server(debug=True)
