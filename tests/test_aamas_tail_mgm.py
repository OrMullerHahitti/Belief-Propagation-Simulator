"""All observed tail values remain available to coordinated MGM local search."""

import numpy as np
import pytest

from experiments.aamas.late_split.mgm import mgm_menu_search, observed_menus
from experiments.aaai.code.merge import mgm1_binary_merge, score_assignment


def test_third_tail_value_is_available_even_when_absent_from_final_pair():
    trace = {
        "variable_names": np.array(["x1"]),
        "assignments": np.array([[2], [0], [1], [2], [0], [1]]),
    }
    menus, indices = observed_menus(trace, 6)
    assert menus == {"x1": [0, 1, 2]}
    assert indices == [0, 1, 2]
    names, axes, tables = ["x1"], {"u1": ["x1"]}, {"u1": np.array([10.0, 5.0, 1.0])}
    result = mgm_menu_search({"x1": 0}, menus, names, axes, tables)
    assert result["assignment"] == {"x1": 2}
    assert result["costs"] == [10.0, 1.0]
    _, old_costs, _ = mgm1_binary_merge(
        {"x1": 0}, {"x1": 1}, "branch1", names, axes, tables
    )
    assert old_costs[-1] == 5.0


@pytest.mark.parametrize("start", ["branch1", "branch2"])
def test_two_value_menus_match_existing_mgm(start):
    rng = np.random.default_rng(7)
    names = ["x1", "x2", "x3"]
    axes = {"f12": names[:2], "f23": names[1:], "u1": names[:1]}
    tables = {"f12": rng.random((3, 3)), "f23": rng.random((3, 3)), "u1": rng.random(3)}
    a, b = dict.fromkeys(names, 0), dict.fromkeys(names, 2)
    expected, costs, moves = mgm1_binary_merge(a, b, start, names, axes, tables)
    result = mgm_menu_search(
        a if start == "branch1" else b, {v: [0, 2] for v in names}, names, axes, tables
    )
    assert result["assignment"] == expected
    np.testing.assert_allclose(result["costs"], costs, atol=1e-10, rtol=0)
    assert result["moves_per_round"] == moves


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_arbitrary_menus_finish_at_menu_local_minimum(seed):
    rng = np.random.default_rng(seed)
    names = ["x1", "x2", "x3"]
    axes = {"f12": names[:2], "f23": names[1:]}
    tables = {f: rng.uniform(1, 100, (4, 4)) for f in axes}
    menus = {"x1": [0, 1, 3], "x2": [0, 2, 3], "x3": [0, 1, 2, 3]}
    result = mgm_menu_search(dict.fromkeys(names, 0), menus, names, axes, tables)
    assert not result["hit_round_cap"]
    assert np.all(np.diff(result["costs"]) < 0)
    for v in names:
        for value in menus[v]:
            alternate = {**result["assignment"], v: value}
            assert score_assignment(alternate, tables, axes) >= result["cost"] - 1e-8


def test_invalid_menu_is_rejected():
    with pytest.raises(ValueError, match="nonempty menus"):
        mgm_menu_search(
            {"x1": 2},
            {"x1": [0, 1]},
            ["x1"],
            {"u1": ["x1"]},
            {"u1": np.array([1.0, 2.0, 3.0])},
        )
