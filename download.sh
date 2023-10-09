#!/bin/bash
# Download the entire data for the project
# (hosted on Zenodo). This allows large data
# files to be downloaded without hosting too
# much on the GitHub repo

DATA_URL=https://zenodo.org/record/8422373/files

cd data

# Downloading main data
wget ${DATA_URL}/hparams_rnd_balanced.tar.gz
wget ${DATA_URL}/hparams_rnd_norm_balanced.tar.gz
wget ${DATA_URL}/hparams_rnd_norm_unbalanced.tar.gz
wget ${DATA_URL}/hparams_rnd_soap_balanced.tar.gz
wget ${DATA_URL}/hparams_rnd_soap_unbalanced.tar.gz
wget ${DATA_URL}/hparams_rnd_unbalanced.tar.gz
wget ${DATA_URL}/hyp_dm.tar.gz
wget ${DATA_URL}/hyp_features.tar.gz
wget ${DATA_URL}/hyp_predictions.tar.gz
wget ${DATA_URL}/xgb_ensembles.tar.gz
wget ${DATA_URL}/xgb_ensembles_hyp.tar.gz

for f in *.tar.gz
do
    tar -zxf $f
    rm $f
done

cd ../dsets

# Downloads the optimized structures from Erlenbach et al (2022)
# directly from their Zenodo repository
# (https://doi.org/10.5281/zenodo.5827897)

wget https://zenodo.org/record/5827897/files/DEEM_NNPscan.db
wget https://zenodo.org/record/5827897/files/IZA_NNPscan.db

cd ..
