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

OUTDIR=/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/
PFDIR=/afs/cern.ch/work/m/mgarciam/private/Tracking_wcoc/
NEV=10

NUM=${1} #random seed
SAMPLE="gun" #main card
GUNCARD="config.gun"


WORKDIR=/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/scratch/${SAMPLE}/${NUM}/
echo $WORKDIR
FULLOUTDIR=${OUTDIR}/${SAMPLE}
PATH_TO_K4GEO="/afs/cern.ch/work/m/mgarciam/private/k4geo"
K4RECTRACKER_dir="/afs/cern.ch/work/m/mgarciam/private/k4RecTracker"
mkdir -p $FULLOUTDIR
if [[ "${SAMPLE}" == "Zcard" ]]
then 
      cp $PFDIR/Pythia_generation/${SAMPLE}.cmd card.cmd
      echo "Random:seed=${NUM}" >> card.cmd
      cat card.cmd
      cp $PFDIR/Pythia_generation/pythia.py ./
fi
mkdir $WORKDIR
cd $WORKDIR

cp $PFDIR/data_processing/process_tree_global.py ./
cp $PFDIR/data_processing/tools_tree_global.py ./
cp $K4RECTRACKER_dir/runIDEAtrackerDigitizer.py ./

if [[ "${SAMPLE}" == "gun" ]] 
then 
      cp -r $PFDIR/gun/gun_random_angle.cpp .
      cp -r $PFDIR/gun/compile_gun_RA.x .
      cp -r $PFDIR/gun/${GUNCARD} .

      source compile_gun_RA.x
      echo 'nevents '${NEV} >> ${GUNCARD}
      echo "  "
      echo " ================================================================================ "
      echo "  "
      echo "running gun"
      echo "  "
      echo " ===============================================================================  "
      echo "  "

      ./gun ${GUNCARD}

fi 

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
## source /cvmfs/sw-nightlies.hsf.org/key4hep/releases/2024-02-26/x86_64-almalinux9-gcc11.3.1-opt/key4hep-stack/2024-02-26-uj7zqp/setup.sh
if [[ "${SAMPLE}" == "Zcard" ]]
then
      k4run $PFDIR/Pythia_generation/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
fi

ddsim --compactFile $PATH_TO_K4GEO/FCCee/IDEA/compact/IDEA_o1_v02/IDEA_o1_v02.xml \
      --outputFile out_sim_edm4hep.root \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV \
      --random.seed $NUM


cd $K4RECTRACKER_dir
k4_local_repo
# export K4RECTRACKER=$PWD/install/share/; PATH=$PWD/install/bin/:$PATH; CMAKE_PREFIX_PATH=$PWD/install/:$CMAKE_PREFIX_PATH; LD_LIBRARY_PATH=$PWD/install/lib:$PWD/install/lib64:$LD_LIBRARY_PATH; export PYTHONPATH=$PWD/install/python:$PYTHONPATH
cd $WORKDIR
k4run runIDEAtrackerDigitizer.py

python process_tree_global.py output_IDEA_DIGI.root reco_${SAMPLE}_${NUM}.root 

cp reco_${SAMPLE}_${NUM}.root $FULLOUTDIR/