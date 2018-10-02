#!/bin/bash

NFOLDS=5

# Run Standard CV pipline
# Run on the final network (2020), 2 PMIDs minimum, W=0.6 and use recall scoring
python scripts/time_network_training-nfold_CV.py 2020 -p 2 -w 0.6 -s recall -n $NFOLDS
# Do a by-compound split
python scripts/time_network_training-nfold_CV.py 2020 -p 2 -w 0.6 -s recall -n $NFOLDS -c


# Rerun Pipline with different seeds for different splits
for SEED in {1..10}
do
    python scripts/time_network_training-nfold_CV.py 2020 -p 2 -w 0.6 -s recall -n $NFOLDS -e $SEED
    # Split by compound
    python scripts/time_network_training-nfold_CV.py 2020 -p 2 -w 0.6 -s recall -n $NFOLDS -c -e $SEED
done

