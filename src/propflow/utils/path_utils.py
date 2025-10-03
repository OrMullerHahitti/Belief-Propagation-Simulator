"""Utility functions for file system and path manipulation.

This module provides a set of helper functions for common tasks related to
file and directory operations, such as finding the project root, creating
directories, and handling file paths.
"""
import pickle
from pathlib import Path
from typing import Any, List, Optional


def find_project_root() -> Path:
    """Finds the project root directory by searching for common markers.

    This function starts from the current working directory and traverses up
    the directory tree, looking for files or directories like `.git`,
    `pyproject.toml`, or `setup.py` that typically indicate the root of a project.

    Returns:
        A `Path` object representing the project root directory, or the current
        working directory if no project root markers are found (e.g., when
        installed as a package).
    """
    current_dir = Path.cwd()
    while True:
        if any((current_dir / marker).exists() for marker in [".git", "pyproject.toml", "setup.py"]):
            return current_dir
        if current_dir == current_dir.parent:
            # When installed as a package, no project root exists - use cwd
            return Path.cwd()
        current_dir = current_dir.parent


def create_directory(path: str) -> None:
    """Creates a directory relative to the project root, if it doesn't exist.

    Args:
        path: The path of the directory to create, relative to the project root.
    """
    Path(find_project_root() / path).mkdir(parents=True, exist_ok=True)


def create_file(path: str, content: str = "") -> None:
    """Creates a file with optional content, overwriting it if it exists.

    Args:
        path: The path of the file to create.
        content: The string content to write to the file. Defaults to "".
    """
    with open(path, "w") as file:
        file.write(content)


def load_pickle(file_path: str) -> Any:
    """Safely loads a pickle file.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The deserialized object from the pickle file, or `None` if an error occurs.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None


def get_absolute_path(relative_path: str) -> str:
    """Converts a relative path to an absolute path string.

    Args:
        relative_path: The relative path to convert.

    Returns:
        The absolute path as a string.
    """
    return str(Path(relative_path).resolve())


def generate_unique_name(base_name: str, extension: str = "") -> str:
    """Generates a unique name by appending a timestamp to a base name.

    Args:
        base_name: The base string for the name.
        extension: The file extension to append (e.g., '.json', '.pkl').

    Returns:
        A unique string in the format `base_name_YYYYMMDDHHMMSS.extension`.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def list_files_in_directory(directory: str, extension_filter: Optional[str] = None) -> List[str]:
    """Lists all files in a directory, with an optional filter for file extension.

    Args:
        directory: The path to the directory.
        extension_filter: An optional file extension to filter by (without the dot).

    Returns:
        A list of file paths as strings.
    """
    path = Path(directory)
    if extension_filter:
        return [str(file) for file in path.glob(f"*.{extension_filter}")]
    return [str(file) for file in path.iterdir() if file.is_file()]
