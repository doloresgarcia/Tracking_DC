#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import wandb
import warnings

# warnings.filterwarnings("ignore")

from torch import nn
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os
import torch
import onnxruntime as ort

os.environ["TORCH_LOGS"] = "onnx_diagnostics"
os.environ["TORCHLIB_EXPERIMENTAL_PREFER_TRACING"] = "1"
import os

dic = torch.load(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/180324_Zcard_v_full/graphs_011123_onnx/0.pt",
    map_location="cpu",
)

input_data = dic["inputs"]
x = dic["model_output"]
ort.set_default_logger_severity(0)

so = ort.SessionOptions()
so.enable_profiling = True
so.inter_op_num_threads = 1
so.intra_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
# --> This option works and takes 3h to load
# GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
# GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
# GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all available optimizations including layout optimizations

print("starting to load")
ort_session = ort.InferenceSession(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/180324_Zcard_v_full/model_multivector_input_011124_v2.onnx",
    # providers=["CPUExecutionProvider"],
    sess_options=so,
)
print("finished loading to load")

input_data = input_data.cpu().numpy()


# input_data = [
#     pos_hits_xyz.cpu().numpy(),
#     hit_type.view(-1, 1).cpu().numpy(),
#     vector.cpu().numpy(),
# ]

# compute ONNX Runtime output prediction
ort_inputs = {
    ort_session.get_inputs()[0].name: input_data,
    # ort_session.get_inputs()[1].name: input_data[1],
    # ort_session.get_inputs()[2].name: input_data[2],
}  # to_numpy(input_data)
ort_outs = ort_session.run(None, ort_inputs)
print(x)
print(ort_outs[0])
torch.testing.assert_close(x, torch.Tensor(ort_outs[0]))
