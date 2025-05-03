from pathlib import Path

#TODO : make this project root
# finder BEETER (for example check if the root that im currently in OR the parent of the current file is not in the list of possible directories
# which will be made by me in the config (some kind of mapping or a list of strings)
def find_project_root():
    """Find the project root directory by looking for a common marker like .git or a specific file"""
    current_dir = Path.cwd()
    while True:
        # Check if this is the project root (containing typical root markers)
        if any((current_dir / marker).exists() for marker in ['.git', 'setup.py', 'pyproject.toml','.root']):
            return current_dir

        # Check if we've reached the filesystem root
        if current_dir == current_dir.parent:
            raise FileNotFoundError("Project root not found")

        # Move up one directory
        current_dir = current_dir.parent

import os


# def find_project_root(start_dir=None):
#     """Find the project root by looking for a specific marker file/directory."""
#     if start_dir is None:
#         start_dir = os.getcwd()
#
#     current_dir = os.path.abspath(start_dir)
#
#     # Keep going up until we find a marker or hit the filesystem root
#     while True:
#         # Check for a marker (e.g., .git directory, pyproject.toml, etc.)
#         if os.path.exists(os.path.join(current_dir, '.git')) or \
#                 os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
#             return current_dir
#
#         # Get the parent directory
#         parent_dir = os.path.dirname(current_dir)
#
#         # If we've reached the filesystem root, stop
#         if parent_dir == current_dir:
#             raise FileNotFoundError("Could not find project root")
#
#         current_dir = parent_dir


