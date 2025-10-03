import numpy as np
import pytest

from propflow.configs.global_config_mapping import (
    CT_FACTORIES,
    CTFactory,
    ENGINE_DEFAULTS,
    get_ct_factory,
    get_validated_config,
    validate_convergence_config,
    validate_engine_config,
    validate_policy_config,
)


def test_get_ct_factory_returns_callable():
    factory_fn = get_ct_factory(CTFactory.random_int)
    table = factory_fn(2, 3, low=0, high=2)
    assert isinstance(table, np.ndarray)
    assert table.shape == (3, 3)

    named_factory = get_ct_factory("uniform_float")
    assert callable(named_factory)

    with pytest.raises(TypeError):
        get_ct_factory(123)  # type: ignore[arg-type]


def test_validation_helpers_enforce_constraints():
    valid_engine = ENGINE_DEFAULTS.copy()
    assert validate_engine_config(valid_engine)

    invalid_engine = valid_engine.copy()
    invalid_engine["max_iterations"] = 0
    with pytest.raises(ValueError):
        validate_engine_config(invalid_engine)

    valid_policy = {"damping_factor": 0.5, "split_factor": 0.4, "pruning_threshold": 0.1}
    assert validate_policy_config(valid_policy)

    with pytest.raises(ValueError):
        validate_policy_config({"damping_factor": 1.5})

    valid_convergence = {"belief_threshold": 1e-3, "min_iterations": 0, "patience": 2}
    assert validate_convergence_config(valid_convergence)

    with pytest.raises(ValueError):
        validate_convergence_config({"belief_threshold": 0})


def test_get_validated_config_merges_defaults():
    config = get_validated_config("engine", {"max_iterations": 25})
    assert config["max_iterations"] == 25
    assert config["normalize_messages"] == ENGINE_DEFAULTS["normalize_messages"]

    simulator_config = get_validated_config("simulator")
    assert "default_max_iter" in simulator_config

    with pytest.raises(ValueError):
        get_validated_config("unknown")
