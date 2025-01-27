#!/bin/bash

TYPE=${1}
CONFIG=${2}
VERSION=${3}
NFILE=${4}

CURRPATH=$(pwd)
ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2025-01-12
set -- "${ORIG_PARAMS[@]}"

outdir=""
if [[ $VERSION -eq 2 ]]; then
    outdir="/eos/experiment/fcc/ee/idea_tracking/idea_v2_01_tracking/"
fi
if [[ $VERSION -eq 3 ]]; then
    outdir="/eos/experiment/fcc/ee/idea_tracking/idea_v3_01_tracking/"
fi

python src/submit_jobs_train.py  --queue microcentury --outdir $outdir --njobs $NFILE --type $TYPE --config $CONFIG --detectorVersion $VERSION

