# Welcome to the repository for `mesenchymal-states`

This project contains the scripts and models for our curated analysis of mesenchymal cell states across development and disease. 

Our project is under active development, so please stay tuned for new results!

The repository is organized into the following directories:

- `data`: 
    - `download.sh`: Bash script for GEO and other data downloading
    - `summary.csv`: Summary and short descriptions of the curated datasets
    - `logs`: Output logs from `download.sh` script
    - `features`: Curated gene lists from single-cell trajectory analyses
        - `biomart`: Lists of homologs for curated gene lists
            - `union.csv`: Combined table of homologs for curated gene lists across all datasets 
- `envs`: Conda environment files for reproducible analyses
- `modeling`: This directory contains the model and training scripts for our `MesNet` structured embedding model
    - `dataset.py`: Contains custom class for `pytorch` datasets
    - `model.py`: Contains model class for `MesNet` implementation in `pytorch`
    - `lit_module.py`: Contains custom `lightning` module
    - `training.py`: CLI script for model training and validation
    - `hparam_sweep.yaml`: Configuration file for hyperparameter sweeps in `wandb`
- `notebooks`: Miscellaneous notebooks for interactive analyses
- `preprocessing`: Structured directory containing two preprocessing scripts for each dataset:
    - `preprocessing.ipynb`: QC, counts normalization, feature selection, dataset integration, dimensionality reduction, and cell-type assignment
    - `trajectory.ipynb`: trajectory analysis with `scfates` to select mesenchymization features (genes)
- `scripts`: Miscellaneous scripts for data preprocessing and collation
