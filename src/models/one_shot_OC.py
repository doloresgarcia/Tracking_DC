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
from torch_geometric.nn.conv import GravNetConv
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



class DemoGravNet(L.LightningModule):
    def __init__(
        self,
        args,
        dev
    ):
        super().__init__()
        self.args = args
        n_layers=3
        in_dim=3
        depth = 1
        k = 2
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(GravNetConv(
                in_channels=in_dim,
                out_channels=in_dim,
                space_dimensions=3,
                propagate_dimensions=3,
                k=k,
            ))
        # layers = [
        #     GravNetConv(
        #         in_channels=in_dim,
        #         out_channels=in_dim,
        #         space_dimensions=3,
        #         propagate_dimensions=3,
        #         k=k,
        #     )
        #     for _ in range(depth)
        # ]
        # self._embedding = nn.Sequential(*layers)
        self._beta = nn.Linear(in_dim, 1)
        self.clustering = nn.Linear(in_dim, 3)
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(3, momentum=0.1)
        
    def forward(self, g, y, step_count, eval=""):
        batch_n = obtain_batch_numbers(g)
        inputs = g.ndata["pos_hits_xyz"]
        inputs = self.ScaledGooeyBatchNorm2_1(inputs)
        for i_layer, layer in enumerate(self.layers):
            latent =  layer(inputs, batch_n) 
        # latent = self._embedding(g.ndata["h_graph_constr"], batch_n)
        beta = self._beta(latent).squeeze()
        x = self.clustering(latent)
        eps = 1e-6
        beta = beta.clamp(eps, 1 - eps)
        g.ndata["original_coords"]=g.ndata["pos_hits_xyz"]
        if (step_count%100)==0:
            PlotCoordinates(
                g,
                path="input_coords",
                outdir=self.args.model_prefix,
                features_type="ones",
                predict=self.args.predict,
                epoch=str(self.current_epoch),
                step_count=step_count,
            )
        g.ndata["final_cluster"] = x
        g.ndata["beta"] = beta.view(-1)
        if (step_count%100)==0:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
                epoch=str(self.current_epoch),
                step_count=step_count,
            )
        return torch.cat((x, beta.view(-1,1)), dim=1)

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


def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    device = g.ndata["particle_number"].device
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(device))
        num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch


