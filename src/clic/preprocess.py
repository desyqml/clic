import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import tqdm

from clic.paths import get_data_dir
from clic.utils import join_batches, read_path


class Preprocessor:
    MAX_ENERGY = 0.5
    ENERGY_AVG = 250  # Average energy for learning
    ENERGY_SHIFT = 25  # Energy range around the average to focus on

    @staticmethod
    def downsample_3d_to_1d(X: np.ndarray, num_pixels: int) -> np.ndarray:
        """Downsample data from:
        X  : (num_images, side, side, side)
        |              |----------------|
        V                    3D Image
        dX : (num_images, num_pixels)
        """
        num_images, side = X.shape[0], X.shape[-1]
        factor = side // num_pixels
        leftout = side - num_pixels * factor
        # From 3D image to 1D image of [side] pixels
        X1d = X.sum((1, 2))
        # Merging multiple pixels into one to end up with [num_pixels] pixels

        # Instead of starting from the first pixel, the 1D image starts
        # from an offset to make it divisible by the number of pixels
        leftout_l, leftout_r = leftout // 2 + leftout % 2, leftout // 2
        X1d = X1d[:, leftout_l : side - leftout_r]

        # Applies a mean to the multiple pixels associated to the same qubit
        dX = X1d.reshape(num_images, num_pixels, factor).mean(axis=2)

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
        path: str,
        num_pixels: int = 8,
    ):
        def process_filecontents(batch, num_pixels):
            x = batch["x"]
            y = batch["y"]

            batch["x"] = Preprocessor.energy_cut(
                Preprocessor.downsample_3d_to_1d(x, num_pixels=num_pixels),
                y,
                Preprocessor.MAX_ENERGY,
                Preprocessor.ENERGY_AVG,
                Preprocessor.ENERGY_SHIFT,
            )
            batch["y"] = y.reshape(-1, 1)
            return batch

        def process_file(args):
            fn, num_pixels = args
            file_content = read_path(fn)
            content_px = process_filecontents(file_content, num_pixels)
            return content_px

        _path = get_data_dir() if path is None else Path(path)
        print(f"Using path: {str(_path)}")

        files = list(_path.glob("**/EleEscan_*.h5"))

        filecontents = [
            process_file(arg)
            for arg in tqdm.tqdm(
                zip(files, [num_pixels] * len(files)), total=len(files)
            )
        ]

        joined = join_batches(filecontents)
        os.makedirs(_path, exist_ok=True)

        with h5py.File(_path / f"clic_{num_pixels}px.h5", "w") as f:
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

        with h5py.File(_path / f"clic_{num_pixels}px_train.h5", "w") as f:
            f.create_dataset("x", data=x_train)
            f.create_dataset("y", data=y_train)

        with h5py.File(_path / f"clic_{num_pixels}px_val.h5", "w") as f:
            f.create_dataset("x", data=x_val)
            f.create_dataset("y", data=y_val)

        with h5py.File(_path / f"clic_{num_pixels}px_test.h5", "w") as f:
            f.create_dataset("x", data=x_test)
            f.create_dataset("y", data=y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert clic data from file to destination path"
    )
    parser.add_argument(
        "-s", "--source", type=str, default=None, help="path to clic file"
    )
    parser.add_argument("-p", "--pixels", type=int, default=8, help="number of pixels")

    args = parser.parse_args()
    Preprocessor.make(path=args.source, num_pixels=args.pixels)
