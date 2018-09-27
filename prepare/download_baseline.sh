#!/bin/bash

# Make the baseline directory for download
mkdir -p ../data/baseline
cd ../data/baseline

# Download all the files
wget -r ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

# Fix the directory sturcutre
pushd ftp.ncbi.nlm.nih.gov/pubmed/baseline/
cp * ../../../
popd
rm -rf ftp.ncbi.nlm.nih.gov

# Make sure all the files are OK
cat *.md5 > all.md5
md5sum -c all.md5
