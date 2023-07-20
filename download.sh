#!/bin/bash
# Download the entire data for the project
# (hosted elsewhere). This allows large data
# files to be downloaded without hosting too
# much on the GitHub repo
#
# NOTE: the Google Drive link is temporary and
# will be replaced by a permanent link upon
# publication


# Downloading main data
FILEID="1yvD-hFQdEVG4vtMmF0zVop4tz_SFPkw1"
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILEID}" -O data.tar.gz && rm -rf /tmp/cookies.txt

tar -zxvf data.tar.gz

# Downloads the optimized structures from Erlenbach et al (2022)
# directly from their Zenodo repository
# (https://doi.org/10.5281/zenodo.5827897)

wget https://zenodo.org/record/5827897/files/DEEM_NNPscan.db
wget https://zenodo.org/record/5827897/files/IZA_NNPscan.db

mv DEEM_NNPscan.db dsets/
mv IZA_NNPscan.db dsets/
