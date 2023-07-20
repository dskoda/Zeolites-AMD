# Inorganic synthesis maps in zeolites

This repository contains all the data, plots, scripts, and notebooks to reproduce the manuscript:

D. Schwalbe-Koda et al. "Inorganic synthesis-structure maps in zeolites with machine learning and crystallographic distances". arXiv (2023)

All the raw data for the computational analysis is found in the [data](data/) folder.
The scripts used to reproduce the plots is available at the [zeo_amd](zeo_amd/) folder.
The Jupyter Notebooks in [nbs](nbs/) contain all the code required to reproduce the analysis and the plots.

## Installing and running

To reproduce the results from the manuscript, first create a new Python environment using your preferred virtual environment (e.g., `venv` or `conda`).
Then, clone this repository and install it with

```bash
git clone git@github.com:dskoda/Zeolites-AMD.git
cd Zeolites-AMD
pip install -e .
```

This should install all dependencies (see [pyproject.toml](pyproject.toml)) and two scripts (`zamd_compare` and `zamd_hyperopt`).

## Description of the data

The raw data folder contains all results shown in the paper, including:

 - Tabulated data for all losses, hyperparameters, and datasets analyzed in the manuscript.
 - Distance matrices calculated with AMD for all known and hypothetical zeolites.
 - Predictions of inorganic synthesis conditions for hypothetical zeolites.
 - Synthesis data used to perform regression of the classifiers.
 - Datasets of hypothetical and known zeolites.
 - Final XGBoost models used in the work.
 - All figures plotted in this work

For more details, see the [data](data/) and [dsets](dsets/) folders.

## Description of the code

The Jupyter notebooks at [nbs](nbs/) contain all information needed to reproduce the analysis of the manuscript.
Each notebook performs part of the analysis and replots the figures from the paper [figs](figs/).
The code in [zeo_amd](zeo_amd/) simplifies the analysis in the notebooks by bundling relevant functions and scripts in the `zeo_amd` package.
They contain scripts that perform, for this specific work:

- Clustering of the zeolite data
- Training and selection of classifiers
- Hyperparameter optimization
- Calculation of the distance matrix using the `amd` code
- Plotting of the minimum spanning tree

## Citing

This data has been produced for the following paper:

```bibtex
@article{SchwalbeKoda2023Inorganic,
    title = {Inorganic synthesis-structure maps in zeolites with machine learning and crystallographic distances},
    author = {Schwalbe-Koda, Daniel, Widdowson, Daniel E., Pham, Tuan Anh, Kurlin, Vitaliy E.},
    year = {2023},
    journal = {arXiv},
    doi = {},
    url = {},
    arxiv={},
}
```

## License

The code is not public - this repository is for peer-review only. Distribution of the data is prohibited until the repository is made available for all.
