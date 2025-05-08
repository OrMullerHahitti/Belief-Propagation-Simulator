# Function to find the project root directory
import pickle
import sys
from pathlib import Path


def find_project_root():
    """Find the project root directory by looking for a common marker like .git or a specific file"""
    current_dir = Path.cwd()
    while True:
        # Check if this is the project root (containing typical root markers)
        if any(
            (current_dir / marker).exists()
            for marker in [".git", "setup.py", "pyproject.toml", ".root"]
        ):
            return current_dir

        # Check if we've reached the filesystem root
        if current_dir == current_dir.parent:
            raise FileNotFoundError("Project root not found")

        # Move up one directory
        current_dir = current_dir.parent


# Make sure your project root is in the Python path
project_root = find_project_root()
sys.path.append(str(project_root))


# Safely load pickle by handling errors - MOVED OUTSIDE TRY BLOCK
def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None
