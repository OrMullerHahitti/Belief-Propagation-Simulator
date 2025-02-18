import numpy as np
import pytest

from utils.randomes import create_random_table, create_random_message, uniform_random, normal_random, integer_random

def test_uniform_random_table_has_correct_shape():
    result = create_random_table((2, 3), uniform_random)
    print(f"\nGenerated uniform random table:\n{result}")
    assert result.shape == (2, 3)

def test_normal_random_table_has_correct_shape():
    result = create_random_table((3, 4), normal_random)
    print(f"\nGenerated normal random table:\n{result}")
    assert result.shape == (3, 4)

def test_integer_random_table_has_correct_shape():
    result = create_random_table((2, 2), lambda s: integer_random(s, 0, 5))
    print(f"\nGenerated integer random table:\n{result}")
    assert result.shape == (2, 2)

def test_uniform_random_message_has_correct_shape():
    result = create_random_message(5, uniform_random)
    print(f"\nGenerated uniform random message:\n{result}")
    assert result.shape == (5,)

def test_normal_random_message_has_correct_shape():
    result = create_random_message(3, normal_random)
    print(f"\nGenerated normal random message:\n{result}")
    assert result.shape == (3,)

def test_integer_random_values_within_bounds():
    low, high = 0, 5
    result = integer_random((100,), low, high)
    print(f"\nGenerated integer random values:\n{result}")
    assert np.all(result >= low)
    assert np.all(result < high)

def test_uniform_random_values_within_bounds():
    result = uniform_random((100,))
    print(f"\nGenerated uniform random values:\n{result}")
    assert np.all(result >= 0)
    assert np.all(result <= 1)

def test_empty_shape_creates_empty_array():
    result = create_random_table((), uniform_random)
    print(f"\nGenerated empty shape array:\n{result}")
    result = np.array(result)
    assert result.shape == ()


def test_zero_dimension_creates_empty_array():
    result = create_random_table((0,), uniform_random)
    print(f"\nGenerated zero dimension array:\n{result}")
    assert result.shape == (0,)