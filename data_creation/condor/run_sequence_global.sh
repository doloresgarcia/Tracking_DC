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

OUTDIR=/afs/cern.ch/user/a/adevita/public/workDir/dataset
PFDIR=/afs/cern.ch/user/a/adevita/public/workDir/Tracking_DC/data_creation/
WORKDIR=/afs/cern.ch/user/a/adevita/public/workDir/temp/

NUM=${1}     # random seed
SAMPLE=${2}  # sample
CONFIG=${3}  # configuration file
NEV=${4}     # number of events

echo "Simulation has started"
echo "Random:seed = ${NUM}"
echo "Sample = ${SAMPLE}"
echo "Workdir = ${WORKDIR}"
echo "ConfigFile = ${CONFIG}"

# FULLOUTDIR="${OUTDIR}/${SAMPLE}_fakeCalo/${NUM}"
PATH_TO_K4GEO="/afs/cern.ch/user/a/adevita/public/workDir/k4geo/"
K4RECTRACKER_dir="/afs/cern.ch/user/a/adevita/public/workDir/k4RecTracker/"
SOURCEFILE_K4RECTRACKER="/afs/cern.ch/user/a/adevita/public/workDir/k4RecTracker/setup.sh"

# mkdir -p $FULLOUTDIR
mkdir -p $WORKDIR
cd $WORKDIR

sleep 5
echo ""
echo "Sourcing k4RecTracker setup.sh"

ORIG_PARAMS=("$@")
set --
source $SOURCEFILE_K4RECTRACKER
set -- "${ORIG_PARAMS[@]}"

echo ""

if [[ "${SAMPLE}" == "gun" ]] 
then 
      cp -r $PFDIR/gun/gun_random_angle.cpp .
      cp -r $PFDIR/gun/CMakeLists.txt .
      cp -r $PFDIR/gun/gun.cpp .
      cp -r $PFDIR/gun/${CONFIG} .

      echo "" >> $CONFIG
      echo "nevents ${NEV}" >> $CONFIG
      echo "  "
      echo " ================================================================================ "
      echo "  "
      echo "running gun"
      echo "  "
      echo " ===============================================================================  "
      echo "  "

      mkdir -p build install
      cd build
      cmake .. -DCMAKE_INSTALL_PREFIX=../install
      make install -j 8
      cd ..
      ./build/gun ${CONFIG} 

fi 

if [[ "${SAMPLE}" == "Zcard" ]]
      then 
      cp $PFDIR/Pythia_generation/${SAMPLE}.cmd card.cmd
      echo "Random:seed=${NUM}" >> card.cmd
      # cat card.cmd
      cp $PFDIR/Pythia_generation/pythia.py ./

      k4run $PFDIR/Pythia_generation/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
fi

ddsim --compactFile $PATH_TO_K4GEO/FCCee/IDEA/compact/IDEA_o1_v02/IDEA_o1_v02.xml \
      --outputFile out_sim_edm4hep.root \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV \
      --random.seed $NUM

cp $PFDIR/data_processing/process_tree_global.py ./
cp $PFDIR/data_processing/tools_tree_global.py ./
cp $K4RECTRACKER_dir/runIDEAtrackerDigitizer.py ./

cd $K4RECTRACKER_dir

cd $WORKDIR
k4run runIDEAtrackerDigitizer.py
mv output_IDEA_DIGI.root "${OUTDIR}/output_IDEA_DIGI_${SAMPLE}_${NUM}.root"

# python process_tree_global.py output_IDEA_DIGI.root reco_${SAMPLE}_${NUM}.root 
# cp reco_${SAMPLE}_${NUM}.root $FULLOUTDIR/