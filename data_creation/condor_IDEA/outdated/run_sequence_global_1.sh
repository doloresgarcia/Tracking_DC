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
PFDIR=/afs/cern.ch/work/m/mgarciam/private/Tracking_wcoc/data_creation/
NEV=1000

NUM=${1} #random seed
SAMPLE="Zcard_CLD" #main card
GUNCARD="config.gun"


WORKDIR=/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/scratch/Zcard_jj_evaluation_v1/${NUM}/
echo $WORKDIR
FULLOUTDIR=${OUTDIR}/Zcard_jj_evaluation_v4/
PATH_TO_K4GEO="/afs/cern.ch/work/m/mgarciam/private/k4geo_versions/k4geo"
K4RECTRACKER_dir="/afs/cern.ch/work/m/mgarciam/private/k4RecTracker_dev_0"
mkdir -p $FULLOUTDIR

mkdir $WORKDIR
cd $WORKDIR
if [[ "${SAMPLE}" == "Zcard_CLD" ]]
      then 
      cp $PFDIR/Pythia_generation/${SAMPLE}.cmd card.cmd
      echo "Random:seed=${NUM}" >> card.cmd
      # cat card.cmd
      cp $PFDIR/Pythia_generation/pythia.py ./
fi
cp $PFDIR/data_processing/process_tree_global.py ./
cp $PFDIR/data_processing/tools_tree_global.py ./
cp $K4RECTRACKER_dir/runIDEAtrackerDigitizer.py ./

# source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2024-09-24
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-10-03
if [[ "${SAMPLE}" == "gun" ]] 
then 
      cp -r $PFDIR/gun/gun.cpp .
      cp -r $PFDIR/gun/CMakeLists.txt .
      cp -r $PFDIR/gun/${GUNCARD} .

      
      echo 'nevents '${NEV} >> ${GUNCARD}
      echo "  "
      echo " ================================================================================ "
      echo "  "
      echo "running gun"
      echo "  "
      echo " ===============================================================================  "
      echo "  "

      mkdir build install
      cd build
      cmake .. -DCMAKE_INSTALL_PREFIX=../install
      make install -j 8
      cd ..
      ./build/gun ${GUNCARD} 

fi 




if [[ "${SAMPLE}" == "Zcard_CLD" ]]
then
      k4run $PFDIR/Pythia_generation/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
fi

ddsim --compactFile $PATH_TO_K4GEO/FCCee/IDEA/compact/IDEA_o1_v02/IDEA_o1_v02.xml \
      --outputFile out_sim_edm4hep.root \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV \
      --random.seed $NUM 
      # --part.minimalKineticEnergy "0.001*MeV"
      # --action.tracker Geant4TrackerAction


cd $K4RECTRACKER_dir
k4_local_repo
# # # # # export K4RECTRACKER=$PWD/install/share/; PATH=$PWD/install/bin/:$PATH; CMAKE_PREFIX_PATH=$PWD/install/:$CMAKE_PREFIX_PATH; LD_LIBRARY_PATH=$PWD/install/lib:$PWD/install/lib64:$LD_LIBRARY_PATH; export PYTHONPATH=$PWD/install/python:$PYTHONPATH
cd $WORKDIR
k4run runIDEAtrackerDigitizer.py

python process_tree_global.py output_IDEA_DIGI.root reco_${SAMPLE}_${NUM}_mc.root False

cp reco_${SAMPLE}_${NUM}_mc.root $FULLOUTDIR/