# Welcome to the repository for `MesenCoder`

This project contains scripts and models for the curated analysis of mesenchymal cell states across development and disease.

`MesenCoder` is under active development, so please stay tuned for new results!

The repository is organized into the following directories:

- `data`: 
    - `download.sh`: Bash script for downloading datasets from GEO and atlases
    - `summary.csv`: Summary and short descriptions for each dataset
    - `CCCA_summary.csv`: Additional descriptions for curated cancer cell atlas datasets
    - `logs`: Output logs from `download.sh` script
    - `features`: Curated gene lists from single-cell trajectories and gene-set databases
        - `biomart`: Lists of mouse/human homologs for each curated gene list
            - `union.csv`: Combined table of mouse/human homologs from all curated gene lists
- `envs`: Conda environment files for preprocessing, modeling, and analysis scripts/notebooks
- `modeling`: Scripts for modeling, training, and inference with `MesenCoder` conditional autoencoder
    - `dataset.py`: Contains custom class for `pytorch` mesenchyme datasets
    - `model.py`: Contains `pytorch` class for `MesenCoder` conditional autoencoder model
    - `lit_module.py`: Contains `lightning` module for training, validation, and prediction
    - `training`/`prediction.py`: CLI scripts for training, validation, and prediction
- `notebooks`: Jupyter notebooks for interactive analysis of preprocessing, modeling, and validation
- `preprocessing`: Utilities, scripts, and subdirectories for manual preprocessing of datasets:
    - Dataset subdirectories:
        - `preprocessing.ipynb`: QC, counts normalization, feature selection, dataset integration, dimensionality reduction, and cell-type assignment
        - `trajectory.ipynb`: trajectory analysis with `scfates` to select mesenchymization features (genes)
    - `scripts`: Contains scripts for gene homology and training dataset collation
    - `utils`: Contains utilities for loading and preprocessing datasets
