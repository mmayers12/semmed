# ML

This folder contains scripts for the machine learning pipleine used for repurposing predictions
in this project.

Approximate script runtime (32 cores, 384 GB system RAM) is:

- 72 hours for the 10x 5-fold xval
- 48 hours for the time-resolved networks
- 7 days for the edge dropout analysis
- 7 days for the edge replacement tests

## To run

Shell scripts are included to automatically pass the arguments used in the pipline to the python scripts
located in the `scripts` folder.

    ./run_cross_validation_pipline.sh
    ./run_time_ml_pipeline.sh
    python run_all_dropouts.py
    python run_all_replacments.py

### Analysis

The analysis folder contains the jupyter notebooks for analyzing the results of this pipeline.  Ideally
all 4 scripts will be run before running those notebooks

