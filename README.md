# semmed

Code to create a hetnet from the data in Sem Med DB

## Importing Sem Med DB dump into mysql

Sem Med DB was downloaded from the [NIH website](https://skr3.nlm.nih.gov/SemMedDB/index.html). (15GB ~ 6 hours).

An empty database was created in mysql
    mysql> CREATE DATABASE semmed_db;

The database was then imported into mysql (~7.5 hrs)
    $ pv semmedDB semmedVER30_R_WHOLEDB_to12312016.sql.gz | zcat | mysql semmed_db


## Setting up the python environment.

Use anaconda with enviornment.yml to run this code.  After installing anaconda
use `conda env create envionment.yml` to install the enviornment. Then use
`source activate ml` to start the enviornmnet.


