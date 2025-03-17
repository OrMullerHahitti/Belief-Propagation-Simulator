import numpy as np
import pytest

from saved_for_later.PotentialTable import CostTable


def test_has_default_random_table():
    table = CostTable(2)
    print(f"Random table shape: {table.table.shape}")
    print(f"Random table: \n{table.table}")
    assert isinstance(table.table, np.ndarray)
    assert table.table.shape == (2, 2)

def test_initializes_with_dict():
    data = {0: [1, 2], 1: [3, 4]}
    table = CostTable(2, data)
    expected = np.array([[1, 2], [3, 4]])
    print(f"Input dict: {data}")
    print(f"Converted table: \n{table.table}")
    assert np.array_equal(table.table, expected)

def test_calculates_mean_columns():
    data = np.array([[1, 2], [3, 4]])
    table = CostTable(2, data)
    print(f"Table: \n{table.table}")
    print(f"Column means: {table.mean_cols}")
    assert np.array_equal(table.mean_cols, np.array([2., 3.]))

def test_calculates_mean_rows():
    data = np.array([[1, 2], [3, 4]])
    table = CostTable(2, data)
    print(f"Table: \n{table.table}")
    print(f"Row means: {table.mean_rows}")
    assert np.array_equal(table.mean_rows, np.array([1.5, 3.5]))

def test_calculates_mean_std():
    data = np.array([[1, 2], [3, 4]])
    table = CostTable(2, data)
    print(f"Table: \n{table.table}")
    print(f"Standard deviation: {table.mean_std}")
    assert table.mean_std == pytest.approx(1.118033988749895, rel=1e-3)