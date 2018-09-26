#!/bin/bash

USERNAME=$1
PASSWORD=$2

# Download the umls meathesaurus
./curl-uts-download.sh $USERNAME $PASSWORD https://download.nlm.nih.gov/umls/kss/2018AA/umls-2018AA-full.zip

# Unzip to the datadir
unzip umls-2018AA-full.zip -d ../data/

# Look at the MD5 Sums
cd ../data/2018AA-full/
md5sum -c 2018AA.MD5

# The actual metathesaurs is contained in a zipped files
unzip 2018aa-1-meta.nlm
unzip 2018aa-2-meta.nlm

# Some of the metathesaurus files are split, so need to be catted back together
cd 2018AA/META
for FILE in MRCONSO MRHIER MRREL MRSAT MRXNW_ENG MRXW
do
    # Cat the pieces together
    zcat $(ls | grep $FILE) > $FILE.RRF
done
