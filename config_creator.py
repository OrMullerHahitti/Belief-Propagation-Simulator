from utils.create_factor_graphs_from_config import FactorGraphBuilder
from utils.path_utils import find_project_root
from utils.create_factor_graph_config import ConfigCreator, GraphConfig
if __name__ == "__main__":
    from utils.create_factor_graph_config import ConfigCreator
    # 1. Create a FactorGraphBuilder instance
    config_path = find_project_root() / "configs/factor_graph_configs"
    ConfigCreator(config_path).create_graph_config(graph_type="random",
                                             domain_size=5,
                                             num_variables=10,
                                             ct_factory="random_int",
                                             ct_params={"low": 1,
                                                        'high': 100},
                                                density=0.4)

    builder = FactorGraphBuilder()

    # 2. Build and save a factor graph from a config file
    for i in range(3):
        cfg_path = f"{find_project_root()}\\configs/factor_graph_configs/random-10-random_intlow1,high1000.4.pkl"
        out_path = builder.build_and_save(cfg_path)



        print(f"Factor graph saved to: {out_path}")
        # 3. Load the factor graph from the saved file
        loaded_graph = builder.load_graph(out_path)
        loaded_graph.visualize()
