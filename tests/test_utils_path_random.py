import os
import pickle
import shutil
import uuid
from pathlib import Path

import numpy as np

from propflow.utils import find_project_root
from propflow.utils.path_utils import (
    create_directory,
    create_file,
    find_project_root as direct_find_project_root,
    generate_unique_name,
    get_absolute_path,
    list_files_in_directory,
    load_pickle,
)
from propflow.utils.randomes import (
    create_random_message,
    create_random_table,
    integer_random,
    normal_random,
    uniform_random,
)

np.random.seed(42)


def test_find_project_root_matches_pyproject():
    root = find_project_root()
    assert (root / "pyproject.toml").exists()
    assert root == direct_find_project_root()


def test_directory_and_file_helpers(tmp_path):
    root = find_project_root()
    unique_dir = f"tmp_propflow_test_{uuid.uuid4().hex}"
    create_directory(unique_dir)
    try:
        target = root / unique_dir
        assert target.exists()
        nested_file = tmp_path / "example.txt"
        create_file(str(nested_file), "hello")
        assert nested_file.read_text() == "hello"
    finally:
        shutil.rmtree(target, ignore_errors=True)


def test_pickle_helpers(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3]}
    pickle_path = tmp_path / "data.pkl"
    with pickle_path.open("wb") as fh:
        pickle.dump(payload, fh)
    loaded = load_pickle(str(pickle_path))
    assert loaded == payload


def test_path_utilities(tmp_path):
    file_path = tmp_path / "item.txt"
    file_path.write_text("42")
    abs_path = get_absolute_path(str(file_path))
    assert Path(abs_path).exists()
    unique = generate_unique_name("report", ".json")
    assert unique.startswith("report_") and unique.endswith(".json")
    files = list_files_in_directory(tmp_path)
    assert str(file_path) in files
    filtered = list_files_in_directory(tmp_path, "txt")
    assert filtered == [str(file_path)]


def test_random_generators_produce_expected_shapes():
    shape = (2, 3)
    uniform = uniform_random(shape)
    assert uniform.shape == shape
    assert np.all((uniform >= 0.0) & (uniform < 1.0))
    normal = normal_random(shape, mean=0.0, std=1.0)
    assert normal.shape == shape
    integers = integer_random(shape, low=0, high=5)
    assert integers.shape == shape
    assert np.all((integers >= 0) & (integers < 5))
    message = create_random_message(4, randomness_policy=uniform_random)
    assert message.shape == (4,)
    table = create_random_table((2, 2), randomness_policy=integer_random)
    assert table.shape == (2, 2)
