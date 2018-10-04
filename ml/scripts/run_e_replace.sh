#!/bin/bash

EDGE_NAME=$1

for i in {1950..2020..5}; do
    python edge_time_differences.py 1985 -e $EDGE_NAME -y $i
done
