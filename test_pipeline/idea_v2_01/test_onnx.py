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


#input_file = "output_tracking.root"
input_file = "out_tracker.root"
reader = root_io.Reader(input_file)
equal_val = 0
total_val = 0
for j,event in enumerate(reader.get("events")):

    if j < 200:
        input_data = []
        
        DC_hits = event.get("CDCHDigis")
        VTXD_hits = event.get("VTXDDigis")
        VTXIB_hits  = event.get("VTXIBDigis")
        VTXOB_hits = event.get("VTXOBDigis")
        out_model = event.get("output_model").vec()

        ListGlobalInputs = []
        it = 0

        ListHitType_VTXD = [] 
        it_0 = 0 
        for input_hit in VTXD_hits:

            ListGlobalInputs.extend([
                input_hit.getPosition().x,
                input_hit.getPosition().y,
                input_hit.getPosition().z,
                1.0, 
                0.0, 0.0, 0.0
            ])
            
            ListHitType_VTXD.append(it)
            it += 1
            it_0 += 1

        ListHitType_VTXD_tensor = torch.tensor(ListHitType_VTXD, dtype=torch.float32)

        ListHitType_VTXIB = []
        it_1 = 0

        for input_hit in VTXIB_hits:
            ListGlobalInputs.extend([
                input_hit.getPosition().x,
                input_hit.getPosition().y,
                input_hit.getPosition().z,
                1.0,
                0.0, 0.0, 0.0
            ])
            ListHitType_VTXIB.append(it)
            it += 1
            it_1 += 1

        ListHitType_VTXIB_tensor = torch.tensor(ListHitType_VTXIB, dtype=torch.float32)

        ListHitType_VTXOB = []
        it_2 = 0
        for input_hit in VTXOB_hits:
            ListGlobalInputs.extend([
                input_hit.getPosition().x,
                input_hit.getPosition().y,
                input_hit.getPosition().z,
                1.0,
                0.0, 0.0, 0.0
            ])
            ListHitType_VTXOB.append(it)
            it += 1
            it_2 += 1

        ListHitType_VTXOB_tensor = torch.tensor(ListHitType_VTXOB, dtype=torch.float32)

        ListHitType_CDC = []
        it_3 = 0
        for input_hit in DC_hits:
            ListGlobalInputs.extend([
                input_hit.getLeftPosition().x,
                input_hit.getLeftPosition().y,
                input_hit.getLeftPosition().z,
                0.0,  
                input_hit.getRightPosition().x - input_hit.getLeftPosition().x,
                input_hit.getRightPosition().y - input_hit.getLeftPosition().y,
                input_hit.getRightPosition().z - input_hit.getLeftPosition().z
            ])
            ListHitType_CDC.append(it)
            it += 1
            it_3 += 1

        ListHitType_CDC_tensor = torch.tensor(ListHitType_CDC, dtype=torch.float32)

        total_val += it
        total_size = it * 7
        tensor_shape = (it, 7)

        list_global_inputs_array = np.array(ListGlobalInputs, dtype=np.float32).reshape(tensor_shape)
        input_data = [list_global_inputs_array]
        
        if it > 0 and it < 20000:
            ort.set_default_logger_severity(0)
            so = ort.SessionOptions()
            so.enable_profiling = True
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            ort_session = ort.InferenceSession(
            "/afs/cern.ch/user/a/adevita/public/workDir/k4RecTracker/model_multivector_input_011124_v2.onnx",
            # providers=["CPUExecutionProvider"],
            sess_options=so,
            )
            
            ort_inputs = {ort_session.get_inputs()[0].name: input_data[0]}
            ort_outs = ort_session.run(None, ort_inputs)
            
            print("Starting Comparison:")
            for i, ort_out in enumerate(ort_outs[0]):

                out_model_slice = out_model[i * 4:(i + 1) * 4]
                out_model_slice_array = np.array([out_model_slice[i] for i in range(len(out_model_slice))], dtype=np.float32)
                
                check = np.array_equal(ort_out, out_model_slice_array)
                
                if check:
                    equal_val += 1

print("Percentage of equal hits:",equal_val/total_val*100,"%")