from utils.create_factor_graphs_from_config import FactorGraphBuilder, get_project_root
from utils.create_factor_graph_config import ConfigCreator, GraphConfig
if __name__ == "__main__":
    from utils.create_factor_graph_config import ConfigCreator
    # 1. Create a FactorGraphBuilder instance
    config_path = get_project_root() / "configs/factor_graph_configs"
    ConfigCreator(config_path).create_config(graph_type="random",
                                             domain_size=3,
                                             num_variables=50,
                                             ct_factory="random_int",
                                             ct_params={"low": 1,
                                                        'high': 100},
                                                density=0.2)

    builder = FactorGraphBuilder()

    # 2. Build and save a factor graph from a config file
    for i in range(50):
        cfg_path = f"{get_project_root()}\\configs/factor_graph_configs/random-50-random_intlow1,high1000.2.pkl"
        out_path = builder.build_and_save(cfg_path)

        print(f"Factor graph saved to: {out_path}")
