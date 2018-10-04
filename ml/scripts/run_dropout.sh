#!/bin/bash

EDGE_NAME=$1

python scripts/edge_dropout_training.py 1985 -d $EDGE_NAME -r 1

for i in {0..4}; do
    for j in $(seq .25 .25 .75); do
        python scripts/edge_dropout_training.py 1985 -d $EDGE_NAME -r $j -n $i
    done
done
