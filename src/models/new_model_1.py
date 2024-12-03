from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))

import torch
import torch.nn as nn
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.batch_operations import obtain_batch_numbers
from torch import Tensor
import lightning as L

from lightning.pytorch.callbacks import BaseFinetuning
from src.logger.logger_wandb import log_losses_wandb_tracking
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
)
from src.layers.losses import object_condensation_loss_tracking
from src.gnn_tracking.models.mlp import MLP
import math
from src.models.Edge_filtering import EFDeepSet

from src.layers.GravNetConv3 import GravNetConv, WeirdBatchNorm

class GravNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 96,
        space_dimensions: int = 3,
        propagate_dimensions: int = 22,
        k: int = 40,
        # batchnorm: bool = True
        weird_batchnom=False,
    ):
        super(GravNetBlock, self).__init__()
        self.d_shape = 64
        out_channels = self.d_shape
        propagate_dimensions = self.d_shape
        self.gravnet_layer = GravNetConv(
            self.d_shape,
            out_channels,
            space_dimensions,
            propagate_dimensions,
            k,
            weird_batchnom,
        ).jittable()

        self.post_gravnet = nn.Sequential(
            nn.Linear(
                out_channels + space_dimensions + self.d_shape, self.d_shape
            ),  #! Dense 3
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 4
            nn.ELU(),
        )
        self.pre_gravnet = nn.Sequential(
            nn.Linear(in_channels, self.d_shape),  #! Dense 1
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 2
            nn.ELU(),
        )

    def forward(
        self,
        g,
        x: Tensor,
        batch: Tensor,
        original_coords: Tensor,
        step_count,
        outdir,
        num_layer,
    ) -> Tensor:
        x = self.pre_gravnet(x)
        x_input = x
        xgn, graph, gncoords, loss_regularizing_neig, ll_r = self.gravnet_layer(
            g, x, original_coords, batch
        )
        g.ndata["gncoords"] = gncoords
        # if step_count % 50:
        #     PlotCoordinates(
        #         g, path="gravnet_coord", outdir=outdir, num_layer=str(num_layer)
        #     )
        # gncoords = gncoords.detach()
        x = torch.cat((xgn, gncoords, x_input), dim=1)
        x = self.post_gravnet(x)
        return x, graph, loss_regularizing_neig, ll_r


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

class DemoGravNet(L.LightningModule):
    def __init__(
        self,
        args,
        dev
    ):
        super().__init__()
        self.args = args
        acts = {
                    "relu": nn.ReLU(),
                    "tanh": nn.Tanh(),
                    "sigmoid": nn.Sigmoid(),
                    "elu": nn.ELU(),
                }
        input_dim= 4
        output_dim= 4
        n_postgn_dense_blocks= 3
        n_gravnet_blocks =4
        clust_space_norm = "twonorm"
        k_gravnet= 7
        activation= "elu"
        weird_batchnom=False
        self.act = acts[activation]

        N_NEIGHBOURS = [7, 64, 16, 64]
        TOTAL_ITERATIONS = len(N_NEIGHBOURS)
        self.return_graphs = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = TOTAL_ITERATIONS
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        self.Dense_1 = nn.Linear(input_dim, 64, bias=False)
        self.clust_space_norm = clust_space_norm

        self.d_shape = 64
        self.gravnet_blocks = nn.ModuleList(
            [
                GravNetBlock(
                    64 if i == 0 else (self.d_shape * i + 64),
                    k=N_NEIGHBOURS[i],
                    weird_batchnom=weird_batchnom,
                )
                for i in range(self.n_gravnet_blocks)
            ]
        )

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(4 * self.d_shape + 64 if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)

        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

        self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.01)
        
    def forward(self, g, y, step_count, eval=""):
        x = torch.cat((g.ndata["pos_hits_xyz"], g.ndata["hit_type"].view(-1, 1)), dim=1)
        original_coords = g.ndata["pos_hits_xyz"]
        g.ndata["original_coords"] = original_coords
        device = x.device
        batch = obtain_batch_numbers(x, g)
        x = self.ScaledGooeyBatchNorm2_1(x)
        x = self.Dense_1(x)
        assert x.device == device

        allfeat = []  # To store intermediate outputs
        allfeat.append(x)
        graphs = []
        loss_regularizing_neig = 0.0
        loss_ll = 0
        if self.trainer.is_global_zero and ((step_count % 100) == 0):
            PlotCoordinates(
                g,
                path="input_coords",
                outdir=self.args.model_prefix,
                features_type="ones",
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            x, graph, loss_regularizing_neig_block, loss_ll_ = gravnet_block(
                g,
                x,
                batch,
                original_coords,
                step_count,
                self.args.model_prefix,
                num_layer,
            )

            allfeat.append(x)
            loss_regularizing_neig = (
                loss_regularizing_neig_block + loss_regularizing_neig
            )
            loss_ll = loss_ll_ + loss_ll
            if len(allfeat) > 1:
                x = torch.concatenate(allfeat, dim=1)

        x = torch.cat(allfeat, dim=-1)
        assert x.device == device

        x = self.postgn_dense(x)
        x = self.ScaledGooeyBatchNorm2_2(x)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and ((step_count % 100) == 0):
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        return x

    def training_step(self, batch, batch_idx):
        y = batch[1]

        batch_g = batch[0]

        if self.trainer.is_global_zero:
            model_output = self(batch_g, y, batch_idx)
        else:
            model_output = self(batch_g, y, 1)

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
            # loss_type=self.args.losstype,
            tracking=True,
            CLD=True,
        )
        loss = loss
        # print("training step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss)

        self.loss_final = loss
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        model_output = self(batch_g, y, batch_idx, eval="_val")
        preds = model_output.squeeze()
        # dic = {}
        # batch_g.ndata["model_output"] = model_output
        # dic["graph"] = batch_g
        # dic["part_true"] = y

        # torch.save(
        #     dic,
        #     self.args.model_prefix + "/graphs/" + str(batch_idx) + ".pt",
        # )
        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
            y,
            q_min=self.args.qmin,
            frac_clustering_loss=0,
            clust_loss_only=self.args.clustering_loss_only,
            use_average_cc_pos=self.args.use_average_cc_pos,
            # loss_type=self.args.losstype,
            tracking=True,
            CLD=True,
        )

      
        # dic = {}
        # batch_g.ndata["model_output"] = model_output
        # dic["graph"] = batch_g
        # dic["part_true"] = y

        # torch.save(
        #     dic,
        #     self.args.model_prefix + "/graphs/" + str(batch_idx) + ".pt",
        # )


        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        # print("validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss, val=True)
        # self.validation_step_outputs.append([model_output, batch_g, y])
        if self.trainer.is_global_zero and self.args.predict:

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
                clustering_mode="clustering_normal",
                tau=self.args.tau,
            )
            print(df_batch)

            self.df_showers.append(df_batch)
            df_batch_ct = evaluate_efficiency_tracks(
                batch_g,
                batch_g.ndata["ct_track_label"],
                y,
                0,
                batch_idx,
                0,
                path_save=self.args.model_prefix + "showers_df_evaluation",
                store=True,
                predict=False,
                ct=True,
                clustering_mode="clustering_normal",
                tau=self.args.tau,
            )
            self.df_showers_ct.append(df_batch_ct)

    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.loss_final)


    def on_validation_epoch_start(self):
        self.df_showers = []
        self.df_showers_ct = []
        self.df_showes_db = []



    def on_validation_epoch_end(self):
        print("end of val predictiong")
        if self.args.predict:
            store_at_batch_end(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers,
                0,
                0,
                0,
                predict=True,
            )
            store_at_batch_end(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers_ct,
                0,
                0,
                1,
                predict=True,
            )
        # if self.trainer.is_global_zero:

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": StepLR(
                    optimizer, step_size=4, gamma=0.1
                ),  # ReduceLROnPlateau(optimizer),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


def obtain_batch_numbers(x, g):
    dev = x.device
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(dev))
        # num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch


