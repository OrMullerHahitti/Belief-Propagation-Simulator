from pathlib import Path

#TODO : make this project root
# finder BEETER (for example check if the root that im currently in OR the parent of the current file is not in the list of possible directories
# which will be made by me in the config (some kind of mapping or a list of strings)
def get_project_root() -> Path:
    """Return the path to the project root directory."""
    # Try to find the project root by looking for a specific directory/file
    current_path = Path(__file__).resolve().parent
    while current_path.name != "Belief_propagation_simulator_" and current_path != current_path.parent:
        current_path = current_path.parent

    # If we didn't find the project root, use the current file's parent
    if current_path == current_path.parent:
        # Fallback to assuming we're running from somewhere within the project
        current_path = Path(__file__).resolve().parent.parent

    return current_path

import os


def find_project_root(start_dir=None):
    """Find the project root by looking for a specific marker file/directory."""
    if start_dir is None:
        start_dir = os.getcwd()

    current_dir = os.path.abspath(start_dir)

    # Keep going up until we find a marker or hit the filesystem root
    while True:
        # Check for a marker (e.g., .git directory, pyproject.toml, etc.)
        if os.path.exists(os.path.join(current_dir, '.git')) or \
                os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            return current_dir

        # Get the parent directory
        parent_dir = os.path.dirname(current_dir)

        # If we've reached the filesystem root, stop
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find project root")

        current_dir = parent_dir


