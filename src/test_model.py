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
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from src.utils.parser_args import parser
from lightning.pytorch.loggers import WandbLogger

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.utils.train_utils import (
    train_load,
    test_load,
)
from src.gatr.interface import (
    embed_point,
    extract_scalar,
    extract_point,
    embed_scalar,
    embed_translation,
)
from src.utils.import_tools import import_module
import wandb

from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.profilers import AdvancedProfiler
from src.utils.train_utils import get_samples_steps_per_epoch, model_setup, get_gpu_dev

import os

import torch
import onnxruntime as ort

os.environ["TORCH_LOGS"] = "onnx_diagnostics"
os.environ["TORCHLIB_EXPERIMENTAL_PREFER_TRACING"] = "1"
from gatr.utils.tensors import expand_pairwise, to_nd
from src.models.Gatr_v import ExampleWrapper
import os

dic = torch.load(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/showers_df_evaluation/graphs_all_hdb/0_0_0.pt",
    map_location="cpu",
)

pos_hits_xyz = dic["graph"].ndata["pos_hits_xyz"]
hit_type = dic["graph"].ndata["hit_type"].view(-1, 1)
vector = dic["graph"].ndata["vector"]
# input_data = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)


model = ExampleWrapper.load_from_checkpoint(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/180324_Zcard_v_full/tets.ckpt",
    dev=0,
    args=None,
)
model.eval()
print("model loaded from checkpoint")
x = model.forward(torch.cat((pos_hits_xyz, hit_type, vector), dim=1))

########################################


dic = torch.load(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/showers_df_evaluation/graphs_all_hdb/0_0_0.pt",
    map_location="cpu",
)

pos_hits_xyz = dic["graph"].ndata["pos_hits_xyz"]
hit_type = dic["graph"].ndata["hit_type"].view(-1, 1)
vector = dic["graph"].ndata["vector"]
input_data = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)


ort.set_default_logger_severity(0)

so = ort.SessionOptions()
so.enable_profiling = True
print(so.inter_op_num_threads)
print(so.intra_op_num_threads)
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
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/model_multivector_1_input.onnx",
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
