# semmed

Code to create a hetnet from the data in Sem Med DB

## Downloading and extracting semmed db data

Sem Med DB predications were downloaded from the [NIH website](https://skr3.nlm.nih.gov/SemMedDB/index.html). (2.32.3GB ~ 1 hour). A UMLS licence/account is required for download.  The `curl-uts-download.sh` utility allows these files to be downloaded from the command line.  Syntax is as follows:

    ./curl-uts-download.sh {umls_user} {umls_password} https://skr3.nlm.nih.gov/SemMedDB/download/semmedVER30_R_PREDICATION_to12312016.sql.gz

The mysql data dump was then extracted and converted into a .csv file:

    $ pv semmedVER30_R_WHOLEDB_to12312017.sql.gz | zcat | python mysqldump_to_csv.py


## Setting up the python environment.

Use anaconda with enviornment.yml to run this code.  After installing anaconda
use `conda env create envionment.yml` to install the enviornment. Then use
`source activate ml` to start the enviornmnet.


