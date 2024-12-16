#!/bin/bash

python submit_jobs_train.py --config config_tracking.gun --outdir  / --njobs 500 --nev 1 --queue workday

# python submit_jobs_train.py --config config_tracking.gun --outdir  /eos/experiment/fcc/ee/datasets/CLD_tracking/Pythia/Zcard_CLD_v5_v1/ --njobs 9000 --nev 1000 --queue workday


