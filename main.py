from utils.create_factor_graphs_from_config import FactorGraphBuilder, get_project_root

if __name__ == "__main__":
    # 1. Create a FactorGraphBuilder instance
    builder = FactorGraphBuilder()

    # 2. Build and save a factor graph from a config file
    cfg_path = f"{get_project_root()}\\configs/factor_graph_configs/configs/factor_graph_configs/max-sum-cycle-3-random_intlow2,high100.pkl"
    out_path = builder.build_and_save(cfg_path)

    print(f"Factor graph saved to: {out_path}")
