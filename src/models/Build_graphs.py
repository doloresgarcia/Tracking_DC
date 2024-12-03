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

class SendScoresMessage(nn.Module):
    """Calculate message of an edge

        Args:
            x_i: Features of node 1 (node where the edge ends)
            x_j: Features of node 2 (node where the edge starts)
            edge_attr: Edge features

        Returns:
            Message
    """

    def __init__(self):
        super(SendScoresMessage, self).__init__()
        self.strict_loading = False
        node_indim = 3
        edge_indim =4 
        edge_outdim=4
        edge_hidden_dim = 40
        self.relational_model = MLP(
                2 * node_indim + edge_indim,
                edge_outdim,
                edge_hidden_dim,
            )
    def forward(self, edges):
        m = torch.cat((edges.src["x"],edges.dst["x"], edges.data["w"]), dim=1)
        w = self.relational_model(m)
        e_tilde = torch.cat((edges.src["x"], w), dim=1)
        return {"w": w, "w1":e_tilde}


class Aggre(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self):
        super(Aggre, self).__init__()
        node_indim = 3
        edge_outdim=4
        node_outdim =3
        node_hidden_dim = 40
        self.object_model = MLP(
                node_indim + edge_outdim,
                node_outdim,
                node_hidden_dim,
            )
    def forward(self, nodes):
        e_tilde_ = nodes.mailbox["w1"]
        x = self.object_model(e_tilde_)
        x = torch.mean(x, dim=1)
        # loss per neighbourhood of same object as src node
        return {"x": x}

class InteractionNetwork(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self):
        super(InteractionNetwork, self).__init__()
        self.send_scores = SendScoresMessage()
        self.Aggre = Aggre()
    def forward(self, g):
        g.update_all(self.send_scores, self.Aggre)
        x_tilde = g.ndata["x"]
        return x_tilde

class FreezeEFDeepSet(BaseFinetuning):
    def __init__(
        self,
    ):
        super().__init__()
        
    def freeze_before_training(self, pl_module):
        print("freezing the following module:", pl_module)

        self.freeze(pl_module.EFDeepSet)

        print("Edge Classification BEEN FROOOZEN")

    def finetune_function(self, pl_module, current_epoch, optimizer):
        print("Not finetunning")
     



class Skip1ResidualNetwork(L.LightningModule):
    def __init__(
        self,
        args,
        dev
    ):
        super().__init__()
        self.args = args
        n_layers=3
        hidden_dim=64
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(InteractionNetwork(
            ))
        # self.layers = [
        #     InteractionNetwork(
        #     )
        #     for _ in range(n_layers)
        # ]
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.alpha = 1
        self.embedding_input = nn.Linear(5,3)
        self.embedding_weight = nn.Linear(1,4)
        self.EFDeepSet = EFDeepSet(args,dev)
        self.p_beta = MLP(3, 1, hidden_dim, L=3)
        self.p_cluster = MLP(3, 3, hidden_dim, L=3)
    def forward(self, g, y, step_count, eval=""):
        w = self.EFDeepSet(g, y, step_count)
        print(w)
        print(torch.sum(torch.isnan(w)))
        # inputs: r, phi, z cylindrical coordinates, pseudorapidity and conformal tracking coordinates (6)
        x = self.embedding_input(g.ndata["h_graph_constr"])
        w = self.embedding_weight(w)
        g.ndata["x"] = x
        g.edata["w"] = w
        for i_layer, layer in enumerate(self.layers):
            if i_layer==0:
                g.ndata["x"] = self.relu(g.ndata["x"])
                g.edata["w"] = self.relu1(g.edata["w"])
            x_resi = g.ndata["x"]
            x_new = layer(g)
            g.ndata["x"] = math.sqrt(self.alpha) * x_resi + math.sqrt(1 - self.alpha) * x_new
            x_tilde = g.ndata["x"]
        beta = self.p_beta(x_tilde)
        x_cluster = self.p_cluster(x_tilde)
        g.ndata["original_coords"]=g.ndata["pos_hits_xyz"]
        # PlotCoordinates(
        #     g,
        #     path="input_coords",
        #     outdir=self.args.model_prefix,
        #     features_type="ones",
        #     predict=self.args.predict,
        #     epoch=str(self.current_epoch),
        #     step_count=step_count,
        # )
        g.ndata["final_cluster"] = x_cluster
        g.ndata["beta"] = beta.view(-1)
        # PlotCoordinates(
        #     g,
        #     path="final_clustering",
        #     outdir=self.args.model_prefix,
        #     predict=self.args.predict,
        #     epoch=str(self.current_epoch),
        #     step_count=step_count,
        # )
        return torch.cat((x_cluster, beta.view(-1,1)), dim=1)

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
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
        num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch


