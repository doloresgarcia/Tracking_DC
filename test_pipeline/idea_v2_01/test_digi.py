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
        print("Event:",j)

        DC_hits = event.get("CDCHDigis")
        DC_association = event.get("CDCHDigisAssociation")
        DC_simhit = event.get("CDCHHits")
        
        VTXD_hits = event.get("VTXDDigis")
        VTXD_links = event.get("VTXD_links")
        VTXD_simhit = event.get("VTXDCollection")


        for DC_asso in DC_association:
            
            sim_hit = DC_asso.getSim()
            digi_hit = DC_asso.getDigi()
            
        # for VTXD_link in VTXD_links:
            
        #     print(dir(VTXD_link))
            