# Download and Downsample `CLIC` dataset
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