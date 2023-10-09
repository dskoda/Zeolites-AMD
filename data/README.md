# Raw Data for: "Inorganic synthesis-structure maps in zeolites..."

This repository contains all the raw data to reproduce the manuscript:

D. Schwalbe-Koda et al. "Inorganic synthesis-structure maps in zeolites with machine learning and crystallographic distances". arXiv:2307.10935 (2023)

The raw data should be used in combination with the code hosted on GitHub: https://github.com/dskoda/Zeolites-AMD.

## Description of the data

The data in this link contains all necessary information to reproduce the manuscript.
In combination with the code hosted on GitHub, it can be visualized and analyzed accordingly.
The data files in this repository are:

- `hparams_rnd_*.json`: results of the hyperparameter optimization of all classifiers studied in this work. The data was produced by randomly sampling the train-validation-test sets. In some cases, the data was normalized (`_norm_`), and the train set was kept `balanced` or `unbalanced`.
- `hyp/*`: directory containing CIF files of hypothetical zeolites highlighted in the manuscript.
- `hyp_dm`: distance matrix of all hypothetical zeolites towards the known zeolites
- `hyp_predictions`: predictions of the synthesis conditions for all hypothetical zeolites
- `iza_dm.csv`: distance matrix between all IZA zeolites considered in this work computed with the AMD. The structures are exactly the idealized obtained from the IZA database.
- `iza_features.csv`: features of IZA zeolites, as extracted from the IZA database
- `iza_mst_nx.pkl`: minimum spanning tree of zeolites, serialized from `networkx` graphs using pickle.
- `iza_mst_positions.csv`: positions of nodes in the minimum spanning tree, for reproducibility.
- `iza_nnpscan_dm.csv`: distance matrix between IZA zeolites computed with the AMD. The zeolites have been optimized with the NNPSCAN method by Erlenbach et al (2022)
- `iza_nnpscan_features.csv`: Features of zeolites optimized with the NNPSCAN method by Erlenbach et al (2022)
- `iza_soap.csv`: distance matrix between all IZA zeolites considered in this work, as computed with SOAP. The structures are exactly the idealized obtained from the IZA database.
- `synthesis-complete.xlsx`: complete set of synthesis conditions for zeoiltes, extracted by Jensen et al. and augmented in this work.
- `synthesis_fraction.csv`: fraction of reported zeolite recipes containing certain elements in the literature. Used to create the training/test data in the work.
- `xgb_ensembles*`: pickle files containing the serialized ensemble models used in the evaluation of the data in this work. The models can be loaded with the `xgboost` Python package.

## License

The data and all the content from this repository is distributed under the [Creative Commons Attribution 4.0 (CC-BY 4.0)](LICENSE.md).

This work was produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

Dataset released as: LLNL-MI-854709.
