# Download and Downsample `CLIC` dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16027525.svg)](https://doi.org/10.5281/zenodo.16027525)

* **Downloading the full dataset** (13.24 GB)
    To download the full dataset to a folder simply run (in `clic/src/clic/`):
    ```bash
    python downloader.py --path YOURPATH
    ```
    
    This will download the full `.tar.gz` file to the folder and it will extract it. The file is large and it will take a while. Run the script, relax, take a coffee, brush your teeth and come back later.

    (If path is not provided it will be saved in `.cache`)
    
    For further help run
    ```bash
    python downloader.py --help
    ```
    
* **Downsample the dataset** (1D only, for now)
    To downsample the dataset (that you have already downloaded through `downloader.py`) simply run (in `clic/src/clic/`)
    
    ```bash
    python preprocess.py -s SOURCE -t DESTINATION -p NUMBER OF PIXELS
    ```
    
    For example:
    ```bash
    python preprocess.py -s /home/samonaco/dust/test/ -t /home/samonaco/dust/test/downsampled/ -p 8
    ```
    
    The downsampling will take a while as well because the original file size is very large. Run the script, take another coffee, brush your teeth and come back later.


    Again, for further help run
    ```bash
    python preprocess.py --help
    ```

# Todo:
- [ ] Make 2D downsampling
- [ ] Parallel Preprocessing

# Citation
If you use this software in your research or publications, **please cite** the following: 

```
@software{monaco_2025_16027525,
  author       = {Monaco, Saverio and
                  Slim, Jamal and
                  Borras, Kerstin and
                  Krücker, Dirk},
  title        = {desyqml/clic: Initial public release},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.16027525},
  url          = {https://doi.org/10.5281/zenodo.16027525},
  swhid        = {swh:1:dir:fc3c8301f34f9eb5cc6d297dc3e51eb554bab5d6
                   ;origin=https://doi.org/10.5281/zenodo.16027205;vi
                   sit=swh:1:snp:422a799224336196205560b51072e0b9a646
                   0690;anchor=swh:1:rel:451fd5ed75243038cc8379223d23
                   1e34c0146c36;path=desyqml-clic-2ccf26b
                  },
}
```
