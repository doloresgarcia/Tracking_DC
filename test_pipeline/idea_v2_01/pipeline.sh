#!/usr/bin/env bash

k4run ../../data_creation/Pythia_generation/pythia.py -n 100 --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard Zcard.cmd

ddsim --compactFile ../../../k4geo/FCCee/IDEA/compact/IDEA_o1_v02/IDEA_o1_v02.xml \
      --outputFile out_sim_edm4hep.root \
      --inputFiles out.hepmc \
      --numberOfEvents 100 \
      --random.seed 42

k4run runIDEAdigitizer.py

k4run runIDEAtracker.py