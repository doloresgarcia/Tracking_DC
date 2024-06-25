#!/bin/bash

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

python submit_jobs_train.py  --njobs 100 \
                             --queue espresso \
                             --outdir /afs/cern.ch/user/a/adevita/public/workDir/dataset/ \
                             --configuration config.gun \
                             --script run_sequence_global.sh \
                             --sample gun \
                             --numEvent 100



