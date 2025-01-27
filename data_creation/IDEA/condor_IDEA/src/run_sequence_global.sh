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

OUTDIR=${1} 
TYPE=${2} 
CONFIG=${3} 
DETECTOR=${4} 
SEED=${5} #random seed
NEV=1000


PFDIR=/afs/cern.ch/user/a/adevita/public/workDir/Tracking_DC/data_creation
WORKDIR=${OUTDIR}/${TYPE}/temp/
FULLOUTDIR=${OUTDIR}/${TYPE}/${CONFIG}

mkdir -p $WORKDIR

PATH_TO_K4GEO="/afs/cern.ch/user/a/adevita/public/workDir/k4geo"
K4RECTRACKER_dir="/afs/cern.ch/user/a/adevita/public/workDir/k4RecTracker"

sleep 5
echo ""
echo "Sourcing k4RecTracker setup.sh"

ORIG_PARAMS=("$@")
set --
cd $K4RECTRACKER_dir
source setup.sh
k4_local_repo
set -- "${ORIG_PARAMS[@]}"
echo ""

cd $WORKDIR
mkdir -p out_hepmc/
if [[ "${TYPE}" == "Pythia" ]]
      then 

      cp $PFDIR/Pythia_generation/${CONFIG}.cmd ${CONFIG}_${SEED}.cmd
      echo "Random:seed=${SEED}" >> ${CONFIG}_${SEED}.cmd

      k4run $PFDIR/Pythia_generation/pythia.py -n $NEV --Dumper.Filename out_hepmc/out_${SEED}.hepmc --Pythia8.PythiaInterface.pythiacard ${CONFIG}_${SEED}.cmd

fi
#rm ${CONFIG}_${SEED}.cmd

mkdir -p out_edm4hep/
ddsim --compactFile $PATH_TO_K4GEO/FCCee/IDEA/compact/IDEA_o1_v0${DETECTOR}/IDEA_o1_v0${DETECTOR}.xml \
      --outputFile out_edm4hep/out_sim_edm4hep_${SEED}.root \
      --inputFiles out_hepmc/out_${SEED}.hepmc \
      --numberOfEvents $NEV \
      --random.seed $SEED \
      --part.minimalKineticEnergy "0.001*MeV"
#rm out_hepmc/out_${SEED}.hepmc

mkdir -p out_digi/
k4run ${K4RECTRACKER_dir}/runIDEAtrackerDigitizer.py --detector "IDEA_v${DETECTOR}_o1" --inputFile out_edm4hep/out_sim_edm4hep_${SEED}.root --outputFile out_digi/output_IDEA_DIGI_${SEED}.root
#rm out_edm4hep/out_sim_edm4hep_${SEED}.root

python $PFDIR/data_processing/IDEA/process_tree_global.py out_digi/output_IDEA_DIGI_${SEED}.root ../${CONFIG}/${CONFIG}_${SEED}.root ${DETECTOR} False
#rm out_digi/output_IDEA_DIGI_${SEED}.root