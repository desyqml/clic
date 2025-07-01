import argparse
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tqdm


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
            "x": f["ECAL"][start:end],
            "y": f["energy"][start:end],
        }
    return res


class Preprocessor:
    MAX_ENERGY = 0.5
    ENERGY_AVG = 250  # Average energy for learning
    ENERGY_SHIFT = 25  # Energy range around the average to focus on

    @staticmethod
    def downsample_3d_to_1d(X: np.ndarray, num_qubits: int) -> np.ndarray:
        """Downsample data from:
        X  : (num_images, side, side, side)
        |              |----------------|
        V                    3D Image
        dX : (num_images, num_qubits)
        """
        num_images, side = X.shape[0], X.shape[-1]
        factor = side // num_qubits
        leftout = side - num_qubits * factor
        # From 3D image to 1D image of [side] pixels
        X1d = X.sum((1, 2))
        # Merging multiple pixels into one to end up with [num_qubits] pixels

        # Instead of starting from the first pixel, the 1D image starts
        # from an offset to make it divisible by the number of qubits
        leftout_l, leftout_r = leftout // 2 + leftout % 2, leftout // 2
        X1d = X1d[:, leftout_l : side - leftout_r]

        # Applies a mean to the multiple pixels associated to the same qubit
        dX = X1d.reshape(num_images, num_qubits, factor).mean(axis=2)

        return dX

    @staticmethod
    def energy_cut(
        X: np.ndarray,
        Y: np.ndarray,
        max_energy: float,
        energy_avg: float,
        energy_shift: float,
    ) -> np.ndarray:
        X = X / 50
        X[X > max_energy] = max_energy
        X = X[np.abs(Y - energy_avg) < energy_shift]
        return X

    @staticmethod
    def make(
        data_path: str,
        num_qubits: int = 8,
    ):
        def process_filecontents(batch, num_qubits):
            x = batch["x"]
            y = batch["y"]

            batch["x"] = Preprocessor.energy_cut(
                Preprocessor.downsample_3d_to_1d(x, num_qubits=num_qubits),
                y,
                Preprocessor.MAX_ENERGY,
                Preprocessor.ENERGY_AVG,
                Preprocessor.ENERGY_SHIFT,
            )
            batch["y"] = y.reshape(-1, 1)
            return batch

        def process_file(args):
            fn, num_qubits = args
            file_content = read_path(fn)
            content_px = process_filecontents(file_content, num_qubits)
            return content_px

        path = Path(data_path)
        print(f"Using path: {str(path)}")

        files = list(path.glob("**/EleEscan_*.h5"))

        filecontents = [
            process_file(arg)
            for arg in tqdm.tqdm(
                zip(files, [num_qubits] * len(files)), total=len(files)
            )
        ]

        joined = join_batches(filecontents)
        os.makedirs(path, exist_ok=True)

        with h5py.File(path / f"clic_{num_qubits}px.h5", "w") as f:
            f.create_dataset("x", data=np.array(joined["x"]))
            f.create_dataset("y", data=np.array(joined["y"]))

        total_size = len(joined["x"])
        train_size = int(total_size * 0.6)
        val_size = int(total_size * 0.2)

        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        x_train, x_val, x_test = (
            np.array(joined["x"])[train_indices],
            np.array(joined["x"])[val_indices],
            np.array(joined["x"])[test_indices],
        )
        y_train, y_val, y_test = (
            np.array(joined["y"])[train_indices],
            np.array(joined["y"])[val_indices],
            np.array(joined["y"])[test_indices],
        )

        with h5py.File(path / f"clic_{num_qubits}px_train.h5", "w") as f:
            f.create_dataset("x", data=x_train)
            f.create_dataset("y", data=y_train)

        with h5py.File(path / f"clic_{num_qubits}px_val.h5", "w") as f:
            f.create_dataset("x", data=x_val)
            f.create_dataset("y", data=y_val)

        with h5py.File(path / f"clic_{num_qubits}px_test.h5", "w") as f:
            f.create_dataset("x", data=x_test)
            f.create_dataset("y", data=y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert clic data from file to destination path"
    )
    parser.add_argument(
        "-s", "--source", type=str, default="", help="path to clic file"
    )
    parser.add_argument("-t", "--to", type=str, help="path to destination folder")
    parser.add_argument("-p", "--pixels", type=int, default=8, help="number of pixels")

    args = parser.parse_args()
    Preprocessor.make(data_path=args.source, num_qubits=args.pixels)
