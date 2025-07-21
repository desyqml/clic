from pathlib import Path
from typing import Tuple, Union

from clic.paths import get_data_dir
from clic.utils import read_path


def load(num_pixels: int = 8, path: Union[str, None] = None) -> Tuple[dict, dict, dict]:
    _path = get_data_dir() if path is None else Path(path)
    return (
        read_path(_path / f"clic_{num_pixels}px_train.h5", xlabel="x", ylabel="y"),
        read_path(_path / f"clic_{num_pixels}px_val.h5", xlabel="x", ylabel="y"),
        read_path(_path / f"clic_{num_pixels}px_test.h5", xlabel="x", ylabel="y"),
    )
