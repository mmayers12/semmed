#!/bin/bash

mkdir -p ../data/basedir
cd ../data/basedir

wget -r ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
pushd ftp.ncbi.nlm.nih.gov/pubmed/baseline/
cp * ../../../
popd
rm -rf ftp.ncbi.nlm.nih.gov

cat *.md5 > all.md5

md5sum -c all.md5
