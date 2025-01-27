import ROOT
from podio import root_io
import dd4hep as dd4hepModule
from ROOT import dd4hep

import sys
import math
import cppyy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import numpy as np
import matplotlib.colors as mcolors
from collections import Counter
import pandas as pd
import csv

import torch
import onnxruntime as ort


input_file = "out_tracker.root"
reader = root_io.Reader(input_file)
equal_val = 0
total_val = 0
for j,event in enumerate(reader.get("events")):

    if j < 1:
        input_data = []
        
        DC_hits = event.get("CDCHDigis")
        VTXD_hits = event.get("VTXDDigis")
        VTXIB_hits  = event.get("VTXIBDigis")
        VTXOB_hits = event.get("VTXOBDigis")
        tracks = event.get("CDCHTracks")

        print("Event:",j)
        for i,track in enumerate(tracks):
            
            if i < 2:
                hits_in_track = track.getTrackerHits()
                no_hits = 0
                for hit in hits_in_track:
                    no_hits += 1
                    
                    print(type(hit))