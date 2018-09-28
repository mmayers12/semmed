#!/bin/bash

# Download the large file (about 1hr)
wget https://skr3.nlm.nih.gov/SemMedDB/download/semmedVER31_R_PREDICATION_06302018.sql.gz -O ../data/semmedVER31_R_PREDICATION_06302018.sql.gz

# Column names already provided in directory
cp ../data/col_names.txt ../data/semmedVER31_R.csv

# pv allows progress to be monitored, if pv is not installed call the following line instead:
# zcat ../data/semmedVER31_R_PREDICATION_06302018.sql.gz | python mysqldump_to_csv.py >> ../data/semmedVER31_R.csv
pv ../data/semmedVER31_R_PREDICATION_06302018.sql.gz | zcat | python mysqldump_to_csv.py >> ../data/semmedVER31_R.csv

