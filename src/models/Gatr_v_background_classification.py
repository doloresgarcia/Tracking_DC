from os import path
import sys

from gatr import GATr, SelfAttentionConfig, MLPConfig

# from src.gatr.nets.gatr import GATr
# from src.gatr.layers.attention.config import SelfAttentionConfig
# from src.gatr.layers.mlp.config import MLPConfig
from gatr.interface import (
    embed_point,
    extract_scalar,
    extract_point,
    embed_scalar,
    embed_translation,
)
import torch
import torch.nn as nn
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.logger.logger_wandb import log_losses_wandb_tracking
from lightning.pytorch.serve import ServableModule, ServableModuleValidator
import lightning as L

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
)
from src.layers.losses import object_condensation_loss_tracking
from src.layers.batch_operations import obtain_batch_numbers

from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb
from src.gatr.primitives.linear import _compute_pin_equi_linear_basis
from src.gatr.primitives.attention import _build_dist_basis


class ExampleWrapper(L.LightningModule):  # nn.Module L.LightningModule
    def __init__(
        self,
        args,
    ):
        super().__init__()
        blocks = 10
        hidden_mv_channels = 16
        hidden_s_channels = 64
        self.input_dim = 3
        self.output_dim = 4
        self.args = args
        self.basis_gp = None
        self.basis_outer = None
        self.pin_basis = None
        self.basis_q = None
        self.basis_k = None
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)

        self.load_basis()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            # basis_gp=self.basis_gp,
            # basis_outer=self.basis_outer,
            # basis_pin=self.pin_basis,
            # basis_q=self.basis_q,
            # basis_k=self.basis_k,
        )

        self.clustering = nn.Linear(16, 1, bias=False)
        self.m = nn.Sigmoid()
        self.loss_bce = nn.BCELoss()
        self.vector_like_data = True

    def load_basis(self):

        filename = "/afs/cern.ch/user/m/mgarciam/.local/lib/python3.8/site-packages/gatr/primitives/data/geometric_product.pt"
        sparse_basis = torch.load(filename).to(torch.float32)
        basis = sparse_basis.to_dense()
        self.basis_gp = basis.to(device="cuda")
        filename = "/afs/cern.ch/user/m/mgarciam/.local/lib/python3.8/site-packages/gatr/primitives/data/outer_product.pt"
        sparse_basis_outer = torch.load(filename).to(torch.float32)
        sparse_basis_outer = sparse_basis_outer.to_dense()
        self.basis_outer = sparse_basis_outer.to(device="cuda")

        self.pin_basis = _compute_pin_equi_linear_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )
        self.basis_q, self.basis_k = _build_dist_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )

    def forward(self, g, input):  #
        # print("forward")
        pos_hits_xyz = input[:, 0:3]
        hit_type = input[:, 3].view(-1, 1)
        vector = input[:, 4:]
        inputs = self.ScaledGooeyBatchNorm2_1(pos_hits_xyz)
        velocities = embed_translation(vector)
        embedded_inputs = embed_point(inputs) + embed_scalar(hit_type) + velocities
        embedded_inputs = embedded_inputs.unsqueeze(-2)
        scalars = torch.zeros((inputs.shape[0], 1))
        mask = self.build_attention_mask(g)
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )
        output = embedded_outputs[:, 0, :]
        binary_score = self.clustering(output)
        return binary_score

    def build_attention_mask(self, g):
        """Construct attention mask from pytorch geometric batch.

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        batch_numbers = obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        model_output = self(batch_g, input_)

        input = model_output
        target = batch_g.ndata["is_overlay"]
        weight_not_b = len(batch_g.ndata["is_overlay"])/(2*torch.sum(batch_g.ndata["is_overlay"]==0))
        weight_b = len(batch_g.ndata["is_overlay"])/(2*torch.sum(batch_g.ndata["is_overlay"]==1))
        weight = batch_g.ndata["is_overlay"].clone()
        weight[batch_g.ndata["is_overlay"]==0]=weight_not_b
        weight[batch_g.ndata["is_overlay"]==1]=weight_b
        loss_value = self.loss_bce(weight, self.m(input), target, reduction=None)
        loss_value = torch.sum(loss_value)/torch.sum(weight)

        if self.trainer.is_global_zero:
            if   ((batch_idx - 1) % 10) == 0 :
                wandb.log({ "loss classification": loss_value.item()})

        # self.loss_final = loss.item()
        # dic = {}
        # batch_g.ndata["model_output"] = model_output
        # dic["graph"] = batch_g
        # dic["part_true"] = y

        # torch.save(
        #     dic,
        #     self.args.model_prefix + "/graphs/" + str(batch_idx) + ".pt",
        # )
        return loss_value

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_ = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        model_output = self(batch_g, input_)
        # dic = {}
        # batch_g.ndata["model_output"] = model_output
        # dic["graph"] = batch_g
        # dic["part_true"] = y
        # np.save(self.args.model_prefix + "/graphs/" + str(batch_idx) + "_onnx.npy", dic)
        # torch.save(
        #     dic,
        #     self.args.model_prefix + "/graphs/" + str(batch_idx) + ".pt",
        # )

        input = model_output
        target = batch_g.ndata["is_overlay"]
        weight_not_b = len(batch_g.ndata["is_overlay"])/(2*torch.sum(batch_g.ndata["is_overlay"]==0))
        weight_b = len(batch_g.ndata["is_overlay"])/(2*torch.sum(batch_g.ndata["is_overlay"]==1))
        weight = batch_g.ndata["is_overlay"].clone()
        weight[batch_g.ndata["is_overlay"]==0]=weight_not_b
        weight[batch_g.ndata["is_overlay"]==1]=weight_b
        loss_value = self.loss_bce(weight, self.m(input), target, reduction=None)
        loss_value = torch.sum(loss_value)/torch.sum(weight)

        if self.trainer.is_global_zero:
            if   ((batch_idx - 1) % 10) == 0 :
                wandb.log({ "loss val  classification": loss_value.item()})

   

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 2 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):
        if self.args.predict:
            store_at_batch_end(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers,
                0,
                0,
                0,
                predict=True,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.start_lr)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": StepLR(optimizer, step_size=4, gamma=0.1),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1,
            },
        }
