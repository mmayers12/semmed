#!/bin/bash

DUMPDATE=$1
TABLENAME=$2
OUTNAME=$3
PSQLUSER=$(whoami)

psql -U $PSQLUSER -d drugcentral_$DUMPDATE -c "\COPY $TABLENAME TO '$(pwd | xargs dirname)/data/drugcentral_${OUTNAME}_${DUMPDATE}.csv' DELIMITER ',' CSV HEADER"
