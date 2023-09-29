# Inorganic synthesis maps in zeolites

This repository contains all the data, plots, scripts, and notebooks to reproduce the manuscript:

D. Schwalbe-Koda et al. "Inorganic synthesis-structure maps in zeolites with machine learning and crystallographic distances". arXiv:2307.10935 (2023)

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

To download the data that reproduces this paper, simply run

```bash
chmod +x download.sh
./download.sh
```

in the root of the repository.
The data is currently hosted on Google Drive during peer review.
It will be made public prior to the publication of the manuscript.

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
    journal = {arXiv:2307.10935},
    doi = {10.48550/arXiv.2307.10935},
    url = {https://doi.org/10.48550/arXiv.2307.10935},
    arxiv = {2307.10935},
}
```

## License

The data and all the content from this repository is distributed under the [Creative Commons Attribution 4.0 (CC-BY 4.0)](LICENSE.md).

This work was produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

Dataset released as: LLNL-MI-854709.
