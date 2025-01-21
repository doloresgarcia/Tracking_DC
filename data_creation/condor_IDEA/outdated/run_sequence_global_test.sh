#!/bin/bash
# the code comes from here: https://zenodo.org/records/8260741
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
# set -e
# set -x

# env
# df -h

OUTDIR=/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia_evaluation/
PFDIR=/afs/cern.ch/work/m/mgarciam/private/Tracking_wcoc/data_creation/
NEV=100

NUM=${1} #random seed
SAMPLE="Zcard" #main card
GUNCARD="config.gun"


WORKDIR=/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia_evaluation/scratch/${SAMPLE}_fakeCalo/${NUM}/
echo $WORKDIR
FULLOUTDIR=${OUTDIR}/${SAMPLE}_fakeCalo
PATH_TO_K4GEO="/afs/cern.ch/work/m/mgarciam/private/k4geo_versions/k4geo"
K4RECTRACKER_dir="/afs/cern.ch/work/m/mgarciam/private/k4RecTracker_dev_0"
mkdir -p $FULLOUTDIR

mkdir $WORKDIR
cd $WORKDIR

cp $K4RECTRACKER_dir/runIDEAtracker.py ./


#source  /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
# source /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/key4hep-stack/2024-03-10-gidfme/setup.sh
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2024-03-07
# # source /cvmfs/sw-nightlies.hsf.org/key4hep/releases/2024-02-26/x86_64-almalinux9-gcc11.3.1-opt/key4hep-stack/2024-02-26-uj7zqp/setup.sh


cd $K4RECTRACKER_dir
k4_local_repo
cd $WORKDIR
k4run runIDEAtracker.py

