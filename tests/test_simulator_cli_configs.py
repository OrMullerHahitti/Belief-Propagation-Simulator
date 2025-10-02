import pickle
import sys

import pytest

from propflow.cli import main as cli_main
from propflow.computators import COMPUTATORS
from propflow.configs import (
    CTFactory,
    ENGINE_DEFAULTS,
    LOG_LEVELS,
    Logger,
    get_ct_factory,
    validate_engine_config,
    validate_policy_config,
)
from propflow.engines import ENGINES
from propflow.simulator import Simulator, _setup_logger


class StubEngine:
    def __init__(self, factor_graph, convergence_config, **kwargs):
        self.factor_graph = factor_graph
        self.history = type("History", (), {"costs": [10.0, 5.0]})()

    def run(self, max_iter):
        return None


def test_simulator_logger_level():
    logger = _setup_logger("INFO")
    assert logger.level == LOG_LEVELS["INFORMATIVE"]


def test_simulator_run_simulations(monkeypatch):
    configs = {"engineA": {"class": StubEngine, "param": 1}}
    sim = Simulator(configs)

    def fake_run_batch(args, max_workers):
        assert len(args) == 1
        return [(0, "engineA", [1.0, 0.5, 0.25])]

    monkeypatch.setattr(sim, "_run_batch_safe", fake_run_batch)
    graphs = [{"nodes": [1, 2]}]
    results = sim.run_simulations(graphs, max_iter=3)
    assert results["engineA"][0][-1] == 0.25


def test_run_single_simulation_success():
    graph = {"nodes": [1]}
    args = (0, "engineA", {"class": StubEngine}, pickle.dumps(graph), 3, LOG_LEVELS["INFORMATIVE"])
    index, name, costs = Simulator._run_single_simulation(args)
    assert index == 0 and name == "engineA" and costs[-1] == 5.0


def test_sequential_fallback(monkeypatch):
    sim = Simulator({"engineA": {"class": StubEngine}})
    monkeypatch.setattr(Simulator, "_run_single_simulation", staticmethod(lambda args: (args[0], args[1], [42])))
    results = sim._sequential_fallback([(0, "engineA", {}, b"", 1, 0)])
    assert results[0][2] == [42]


def test_set_log_level_and_invalid_value():
    sim = Simulator({"engineA": {"class": StubEngine}})
    sim.set_log_level("HIGH")
    assert sim.logger.level == LOG_LEVELS["HIGH"]
    previous = sim.logger.level
    sim.set_log_level("unknown")
    assert sim.logger.level == previous


def test_cli_main_version_output(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["propflow", "--version"])
    cli_main()
    out = capsys.readouterr().out
    assert "PropFlow Version" in out
    monkeypatch.setattr(sys, "argv", ["propflow"])
    cli_main()
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_config_validators_and_factories():
    config = ENGINE_DEFAULTS.copy()
    assert validate_engine_config(config)
    policy = {"damping_factor": 0.5, "split_factor": 0.4, "pruning_threshold": 0.1}
    assert validate_policy_config(policy)
    factory_fn = get_ct_factory(CTFactory.random_int)
    arr = factory_fn(2, 3, low=0, high=2)
    assert arr.shape == (3, 3)


def test_global_logger_and_registries():
    logger = Logger("test", file=False)
    logger.info("hello")
    assert "min-sum" in COMPUTATORS
    assert "BPEngine" in ENGINES
