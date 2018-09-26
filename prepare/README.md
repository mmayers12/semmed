# Preparation for building the SemmedDB network

This folder contains the scripts and notebooks needed to download and do some inital processing on
all various data sources for this project.

## To run

Although a lot of scripts in this directory can be run in any order, a few have dependencies.  For Best results
run the shell scripts first, then run the ipython notebooks.

To ensure all scripts run properly, please enable the virtual enviornment before running

    1. `./download_semmeddb.sh`
    2. `./download_umls.sh {umls_usnername} {umls_password}`
    3. `./download_baseline.sh`


