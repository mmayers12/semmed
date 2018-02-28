# semmed

Code to create a hetnet from the data in Sem Med DB

## Downloading and extracting semmed db data

Sem Med DB predications were downloaded from the [NIH website](https://skr3.nlm.nih.gov/SemMedDB/index.html). (2.32.3GB ~ 1 hour). A UMLS licence/account is required for download.  The `curl-uts-download.sh` utility allows these files to be downloaded from the command line.  Syntax is as follows:

    ./curl-uts-download.sh {umls_user} {umls_password} https://skr3.nlm.nih.gov/SemMedDB/download/semmedVER30_R_PREDICATION_to12312016.sql.gz

The mysql data dump was then extracted and converted into a .csv file:

    $ cp col_names.txt semmedVER30_R.csv
    $ pv semmedVER30_R_WHOLEDB_to12312017.sql.gz | zcat | python mysqldump_to_csv.py >> semmed_ver30_R.csv

Column names were extracted from the mysql data dump, and can be prepended onto the data using the above statements.


## Setting up the python environment.

Use anaconda with enviornment.yml to run this code.  After installing anaconda
use `conda env create envionment.yml` to install the enviornment. Then use
`source activate ml` to start the enviornmnet.

## Getting Data from Drugcentral Dump

Install postgresql and load the dump into postgresql.
This command will allow for the extraction of tables from the dump:

    psql
    \connect drugcentral_04252017
    \COPY omop_relationship TO '/home/mmayers/projects/semmed/data/drugcentral_rel.csv' DELIMITER ',' CSV HEADER;
    \COPY identifier TO '/home/mmayers/projects/semmed/data/drugcentral_ids.csv' DELIMITER ',' CSV HEADER;
    \COPY approval TO '/home/mmayers/projects/semmed/data/drugcentral_approvals.csv' DELIMITER ',' CSV HEADER;


## Downloading the pubmed basline files

    $ ./download_baseline.sh

