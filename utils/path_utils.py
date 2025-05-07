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

def create_directory(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def create_file(path: str, content: str = "") -> None:
    """Create a file with optional content. Overwrites if it already exists."""
    with open(path, 'w') as file:
        file.write(content)

def get_absolute_path(relative_path: str) -> str:
    """Convert a relative path to an absolute path."""
    return str(Path(relative_path).resolve())

def generate_unique_name(base_name: str, extension: str = "") -> str:
    """Generate a unique name by appending a timestamp to the base name."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base_name}_{timestamp}{extension}"

def list_files_in_directory(directory: str, extension_filter: str = None) -> list:
    """List all files in a directory, optionally filtering by extension."""
    path = Path(directory)
    if extension_filter:
        return [str(file) for file in path.glob(f"*.{extension_filter}")]
    return [str(file) for file in path.iterdir() if file.is_file()]


