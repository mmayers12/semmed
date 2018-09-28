# Preparation for building the SemmedDB network

This folder contains the scripts and notebooks needed to download and do some inital processing on
all various data sources for this project.

## To run

Although a lot of scripts in this directory can be run in any order, a few have dependencies.  For Best results
run the shell scripts first, then run the ipython notebooks.

To ensure all scripts run properly, please enable the virtual enviornment before running


### The following scripts must be run for the rest of the work

1. `./download_semmeddb.sh`
2. `./download_umls.sh {umls_usnername} {umls_password}`
3. `./download_baseline.sh`
4. `./get_semtype_files.sh`
5. `./download_drugcentral.sh`


## Requirments

`download_umls.sh` requires a UMLS account with your id and password

`download_drugcentral.sh` requires PostgreSQL to be installed on the system.
This dump was created with PostgreSQL 10, however it appears to work with 9.5 as well.

`download_semmeddb.sh` uses the program pv to view progress as it converts the SQL dump to a .csv file
