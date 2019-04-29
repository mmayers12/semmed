#!/bin/bash

for YEAR in {1950..2020..1}
do
    # 2-pmid minimum, recall scoring, dwpc_w = 0.6, otherwise defaults
    python scripts/time_network_training-past_vs_future.py $YEAR -s recall -w 0.6 -p 2
done

