from os import path
import sys
from gatr import GATr, SelfAttentionConfig, MLPConfig
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


class ExampleWrapper(L.LightningModule):
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
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        self.clustering = nn.Linear(16, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(16, 1)
        self.vector_like_data = True

    def forward(self, input_data):
        pos_hits_xyz = input_data[:, 0:3]
        inputs_scalar = input_data[:, 3].view(-1, 1)
        inputs = self.ScaledGooeyBatchNorm2_1(pos_hits_xyz)
        if self.vector_like_data:
            velocities = input_data[:, 4:].view(-1, 1)
            velocities = embed_translation(velocities)
            embedded_inputs = (
                embed_point(inputs) + embed_scalar(inputs_scalar) + velocities
            )
        else:
            embedded_inputs = embed_point(inputs) + embed_scalar(inputs_scalar)
        embedded_inputs = embedded_inputs.unsqueeze(-2)

        scalars = torch.zeros((inputs.shape[0], 1))

        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=scalars)
        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=scalars)
        output = embedded_outputs[:, 0, :]
        x_point = output
        x_scalar = output
        x_cluster_coord = self.clustering(x_point)
        beta = self.beta(x_scalar)
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        return x

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
        input_data = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        model_output = self(input_data)

        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            tracking=True,
        )
        loss = loss
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss)

        self.loss_final = loss.item()
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        pos_hits_xyz = batch_g.ndata["pos_hits_xyz"]
        hit_type = batch_g.ndata["hit_type"].view(-1, 1)
        vector = batch_g.ndata["vector"]
        input_data = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)
        model_output = self(input_data)

        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
            y,
            q_min=self.args.qmin,
            frac_clustering_loss=0,
            clust_loss_only=self.args.clustering_loss_only,
            use_average_cc_pos=self.args.use_average_cc_pos,
            tracking=True,
        )
        loss = loss
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss, val=True)
        if self.trainer.is_global_zero:
            df_batch = evaluate_efficiency_tracks(
                batch_g,
                model_output,
                y,
                0,
                batch_idx,
                0,
                path_save=self.args.model_prefix + "showers_df_evaluation",
                store=True,
                predict=False,
            )
            if self.args.predict:
                if len(df_batch) > 0:
                    self.df_showers.append(df_batch)

    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.loss_final)

    def on_train_epoch_start(self):
        self.make_mom_zero()

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
