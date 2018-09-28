#!/bin/bash

DUMPDATE=06212018

wget http://iridium.noip.me/drugcentral.dump.$DUMPDATE.sql.gz -O ../data/drugcentral.dump.$DUMPDATE.sql.gz

# The remainer of this file requires postgreSQL to be installed and running on the mchine
createdb drugcentral_$DUMPDATE
gunzip < ../data/drugcentral.dump.$DUMPDATE.sql.gz | psql drugcentral_$DUMPDATE

# Copy the required tables to disk for easy reading
./dc_copy_command.sh $DUMPDATE omop_relationship rel
./dc_copy_command.sh $DUMPDATE identifier ids
./dc_copy_command.sh $DUMPDATE approval approvals
./dc_copy_command.sh $DUMPDATE synonyms syn
./dc_copy_command.sh $DUMPDATE atc atc
./dc_copy_command.sh $DUMPDATE atc_ddd atc-ddd
