from experiments.non_convergence_chain.diagonal_analyzer import (
    binary_diagonal_orientation,
    generalized_selected_entry_signature,
)


def test_binary_main_diagonal():
    assert binary_diagonal_orientation([[0, 0], [1, 1]]) == "main"


def test_binary_anti_diagonal():
    assert binary_diagonal_orientation([[0, 1], [1, 0]]) == "anti"


def test_binary_mixed_selected_entries():
    assert binary_diagonal_orientation([[0, 0], [0, 1]]) == "mixed"


def test_generalized_non_binary_signature():
    signature = generalized_selected_entry_signature(
        [[0, 2], [1, 1], [0, 2]],
        row_minimizer_map={"0": [2], "1": [1]},
        column_minimizer_map={"1": [1], "2": [0]},
    )

    assert signature["selected_cell_coordinates"] == [[0, 2], [1, 1], [0, 2]]
    assert signature["row_minimizer_map"]["0"] == [2]
    assert signature["active_minimizer_transition_signature"]["[0, 2]"] == 2
