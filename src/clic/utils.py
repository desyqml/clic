from pathlib import Path
from typing import Optional

import h5py


def split_batch(batch, cutoff):
    """
    Splits a batch of data into two parts at the specified cutoff index.

    Parameters:
    batch (dict): A dictionary where the keys are data labels and the values are tensors.
    cutoff (int): The index at which to split the batch.

    Returns:
    tuple: Two dictionaries containing the split data.
    """
    return {k: v[:cutoff] for k, v in batch.items()}, {
        k: v[cutoff:] for k, v in batch.items()
    }


def join_batches(batches):
    """
    Joins a list of batches into a single batch by concatenating tensors along the first dimension.

    Parameters:
    batches (list): A list of dictionaries, where each dictionary represents a batch.

    Returns:
    dict: A single dictionary containing concatenated data from all input batches.
    """
    joined_batch = {
        k: [item for batch in batches for item in batch[k]] for k in batches[0].keys()
    }
    return joined_batch


def read_path(
    fn: Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
    xlabel: str = "ECAL",
    ylabel: str = "energy",
) -> dict:
    """
    Reads data from an HDF5 file and returns it as a dictionary of tensors.

    Parameters:
    fn (Path): The path to the HDF5 file.
    start (int, optional): The starting index for reading data. Defaults to None.
    end (int, optional): The ending index for reading data. Defaults to None.

    Returns:
    dict: A dictionary containing the read data as tensors.
    """
    with h5py.File(fn, "r") as f:
        res = {
            "x": f[xlabel][start:end],
            "y": f[ylabel][start:end],
        }
    return res
