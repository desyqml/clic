import argparse
import os
import sys
import tarfile
from pathlib import Path
from typing import Union

import wget

from clic.paths import get_data_dir


class Downloader:
    URL = "https://zenodo.org/records/3603122/files/Ele_FixedAngle.tar.gz?download=1"
    ARCHIVE_NAME = "Ele_FixedAngle.tar.gz"
    FOLDER_NAME = "Ele_FixedAngle"

    @staticmethod
    def download(path: Path, overwrite: bool = False) -> Path:
        """
        Downloads the dataset archive if not already present.

        Parameters
        ----------
        path : str
            Directory where the archive will be saved.
        overwrite : bool
            If True, forces re-download of the archive.

        Returns
        -------
        Path
            Path to the downloaded archive.
        """
        os.makedirs(path, exist_ok=True)
        archive_path = Path(path) / Downloader.ARCHIVE_NAME

        if archive_path.exists() and not overwrite:
            print(f"Archive already exists at {archive_path}. Skipping download.")
            return archive_path

        print(f"Downloading dataset from {Downloader.URL}...")
        wget.download(
            Downloader.URL,
            str(archive_path),
            bar=Downloader._progress_bar,
        )
        print(f"\nDownloaded to {archive_path}")
        return archive_path

    @staticmethod
    def extract(archive_path: Path, overwrite: bool = False) -> Path:
        """
        Extracts the dataset archive.

        Parameters
        ----------
        archive_path : Path
            Path to the .tar.gz archive.
        extract_to : Optional[str]
            Directory to extract to. Defaults to archive directory.
        overwrite : bool
            If True, extracts even if folder already exists.

        Returns
        -------
        Path
            Path to the extracted dataset folder.
        """
        dest_folder = archive_path.parent / Downloader.FOLDER_NAME

        if dest_folder.exists() and not overwrite:
            print(f"Folder '{dest_folder}' already exists. Skipping extraction.")
            return dest_folder

        print(f"Extracting {archive_path} to {archive_path.parent}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=archive_path.parent)
        print("Extraction completed.")

        return dest_folder

    @staticmethod
    def get(path: Union[str, None], overwrite: bool = False) -> Path:
        """
        Downloads and extracts the dataset if needed.

        Parameters
        ----------
        path : str
            Target directory for download and extraction.
        overwrite : bool
            Whether to overwrite existing files.

        Returns
        -------
        Path
            Path to the extracted dataset folder.
        """
        _path = get_data_dir() if path is None else Path(path)
        archive_path = Downloader.download(_path, overwrite)
        extracted_path = Downloader.extract(archive_path, overwrite)
        return extracted_path

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 80):
        percent = current / total * 100
        downloaded = current / (1024**3)
        total_gb = total / (1024**3)
        sys.stdout.write(
            f"\rDownloading {percent:.2f}% [{downloaded:.2f}/{total_gb:.2f} GB]"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="target directory for download and extraction",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing files",
    )
    args = parser.parse_args()

    Downloader.get(args.path, args.overwrite)
