from types import SimpleNamespace

from propflow.utils import experiments


def test_run_baseline_comparison(monkeypatch):
    class DummyPerformanceMonitor:
        def __init__(self):
            self.summary = {"total_messages": 10, "total_time": 1.0, "avg_memory_mb": 0.1}

        def get_summary(self):
            return self.summary

    class DummyPruningPolicy:
        def get_stats(self):
            return {"pruned_messages": 2, "pruning_rate": 0.2}

    class DummyEngine:
        def __init__(self, factor_graph, computator, monitor_performance=True, **kwargs):
            self.factor_graph = factor_graph
            self.computator = computator
            self.monitor_performance = monitor_performance
            self.history = SimpleNamespace(costs=[5.0, 3.0])
            self.performance_monitor = DummyPerformanceMonitor()
            self.pruning_policy = DummyPruningPolicy()

        def run(self, max_iter, save_csv):
            return None

    class DummyBuilder:
        def build_and_save(self, path):
            return "graph.pkl"

        def load_graph(self, path):
            return SimpleNamespace(name="graph")

    class DummyCreator:
        def create_graph_config(self, **kwargs):
            return "config.pkl"

    monkeypatch.setattr(experiments, "ConfigCreator", DummyCreator)
    monkeypatch.setattr(experiments, "FactorGraphBuilder", DummyBuilder)
    monkeypatch.setattr(experiments, "BPEngine", DummyEngine)
    monkeypatch.setattr(experiments, "MessagePruningEngine", DummyEngine)
    monkeypatch.setattr(experiments, "MinSumComputator", lambda: "computator")

    results = experiments.run_baseline_comparison()
    assert "regular" in results and "pruning" in results
    assert results["pruning"]["pruning_rate"] == 0.2
