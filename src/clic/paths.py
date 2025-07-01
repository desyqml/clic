import os
from pathlib import Path
from typing import Union

# Default location (e.g., user cache dir, current dir, or custom env)
_default_data_dir = Path(os.getenv("CLIC_DATA_DIR", Path.home() / ".cache" / "clic"))


def get_data_dir() -> Path:
    """Return the base directory where CLIC files are stored."""
    return _default_data_dir


def set_data_dir(path: Union[str, Path]):
    """Set a new base directory where CLIC should save files."""
    global _default_data_dir
    _default_data_dir = Path(path)


def get_file_path(name: str) -> Path:
    """Return full path to a named file under the CLIC data dir."""
    return get_data_dir() / name
