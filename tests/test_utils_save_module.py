import json
from dataclasses import dataclass

import numpy as np
import pytest

from propflow.utils.save import (
    EnhancedSaveModule,
    _serialize_numpy_array,
    save_enhanced_engine_analysis,
    save_simulation_data,
    save_simulation_result,
    save_simulator_comprehensive_analysis,
)

np.random.seed(42)


@dataclass
class DummyVariable:
    name: str
    domain: int


@dataclass
class DummyFactor:
    name: str
    connection_number: dict
    cost_table: np.ndarray


@dataclass
class DummyMessage:
    sender: str
    recipient: str


class DummyHistory:
    def __init__(self) -> None:
        self.costs = [10.0, 5.0, 5.0]
        self.beliefs = {0: {"x1": np.array([0.2, 0.8])}}
        self.assignments = {0: {"x1": 1}}
        self.use_bct_history = True
        self.step_messages = {0: [DummyMessage("f1", "x1")], 1: [DummyMessage("x1", "f1")]}


class DummyConvergenceMonitor:
    def __init__(self) -> None:
        self.convergence_history = [
            {"belief_converged": False, "assignment_converged": False},
            {"belief_converged": True, "assignment_converged": True},
        ]


class DummyPerformanceMonitor:
    def __init__(self) -> None:
        self._summary = {"total_steps": 2, "avg_step_time": 0.1}

    def get_summary(self) -> dict:
        return self._summary


class DummyGraph:
    def __init__(self, variables, factors) -> None:
        self.variables = variables
        self.factors = factors


class DummyEngine:
    def __init__(self) -> None:
        self.graph = DummyGraph(
            variables=[DummyVariable("x1", 3)],
            factors=[
                DummyFactor(
                    name="f1",
                    connection_number={"x1": 0},
                    cost_table=np.array([[1.0, 2.0, 3.0]]),
                )
            ],
        )
        self.history = DummyHistory()
        self.convergence_monitor = DummyConvergenceMonitor()
        self.performance_monitor = DummyPerformanceMonitor()
        self.name = "dummy"


class DummySimulator:
    def __init__(self) -> None:
        self.results = {"engineA": [[10.0, 5.0, 2.5], [9.0, 4.5]]}
        self.engine_configs = {"engineA": {"class": DummyEngine}}
        self.logger = type("L", (), {"level": 20})()


@pytest.fixture
def dummy_engine():
    return DummyEngine()


@pytest.fixture
def tmp_chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_serialize_numpy_array_handles_nested():
    data = {"a": np.array([1, 2]), "b": np.array([[3.0], [4.0]])}
    serialized = _serialize_numpy_array(data)
    assert serialized == {"a": [1, 2], "b": [[3.0], [4.0]]}


def test_save_simulation_data_writes_expected(tmp_path, dummy_engine):
    out_path = tmp_path / "simulation.json"
    save_simulation_data(dummy_engine, str(out_path))
    with out_path.open() as fh:
        payload = json.load(fh)
    assert payload["agents"][0]["name"] == "x1"
    assert payload["history"]["costs"] == [10.0, 5.0, 5.0]


def test_save_simulation_result_structure(tmp_path, dummy_engine, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 123.456)
    out_path = tmp_path / "result.json"
    save_simulation_result(dummy_engine, str(out_path))
    with out_path.open() as fh:
        data = json.load(fh)
    assert data["simulationResult"]["totalIterations"] == 3
    assert data["simulationResult"]["steps"][0]["timestamp"] == 123456


def test_enhanced_save_module_for_simulator(tmp_chdir, dummy_engine):
    simulator = DummySimulator()
    saver = EnhancedSaveModule()
    json_path = tmp_chdir / "analysis.json"
    saver.save_simulator_analysis(simulator, str(json_path), save_csv=True)
    assert json_path.exists()
    csv_path = tmp_chdir / "analysis_summary.csv"
    assert csv_path.exists()


def test_enhanced_save_module_for_engine(tmp_chdir, dummy_engine, monkeypatch):
    saver = EnhancedSaveModule()
    monkeypatch.setattr(saver, "_get_base_engine_data", lambda engine: {"agents": []})
    analysis_path = tmp_chdir / "engine.json"
    saver.save_enhanced_engine_data(dummy_engine, str(analysis_path))
    assert analysis_path.exists()
    with analysis_path.open() as fh:
        payload = json.load(fh)
    assert payload["analysis_metadata"]["engine_name"] == "dummy"
    assert payload["performance_analysis"]["has_detailed_metrics"]


def test_convenience_wrappers(tmp_chdir, dummy_engine, monkeypatch):
    simulator = DummySimulator()
    path_sim = save_simulator_comprehensive_analysis(simulator, str(tmp_chdir / "sim.json"), save_csv=False)
    assert path_sim.endswith("sim.json")
    monkeypatch.setattr(EnhancedSaveModule, "_get_base_engine_data", lambda self, engine: {"agents": []})
    path_engine = save_enhanced_engine_analysis(dummy_engine, str(tmp_chdir / "eng.json"), include_performance=False, include_convergence_detail=False)
    assert path_engine.endswith("eng.json")
