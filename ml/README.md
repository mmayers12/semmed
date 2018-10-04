# ML

This folder contains scripts for the machine learning pipleine used for repurposing predictions
in this project.

Approximate script runtime (32 cores, 384 GB system RAM) is:

- 30 hrs for the 5-fold xval
- 24 hours for the time-resolved networks
- 72 hours for the edge dropout analyiss
- 72 hours for the edge replacement tests

## To run

Shell scripts are included to automatically pass the arguments used in the pipline to the python scripts
located in the `scripts` folder.

    ./run_cross_validation_pipline.sh
    ./run_time_ml_pipeline.sh

