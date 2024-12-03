from os import path
import sys
import torch
import torch.nn as nn
import dgl
import lightning as L
from torch.optim.lr_scheduler import StepLR
import wandb
import torch.nn.functional as F
import numpy as np

def edge_concat_data():
    def func(edges):
        x = torch.cat((edges.src["h_graph_constr"], edges.dst["h_graph_constr"], edges.data["z0"], edges.data["theta_slope"]), dim=1)
        return {"x": x}
    return func

def z0():
    def func(edges):
        distance = edges.src["z"]- edges.src["rho"]*(edges.src["z"] - edges.dst["z"])/(edges.src["rho"] - edges.dst["rho"]+1e-6)
        # print("nans z0", torch.min(distance), torch.max(distance))
        return {"z0": torch.log(torch.abs(distance))}
    return func


def theta_slope():
    def func(edges):
        distance = torch.abs((edges.src["theta"] - edges.dst["theta"])/(edges.src["rho"] - edges.dst["rho"]+1e-6))
        # print("nans theta_slope", torch.min(distance), torch.max(distance))
        return {"theta_slope": torch.log(distance+1e-6)}
    return func

def same_particle():
    def func(edges):
        distance = 1*(edges.src["particle_number"]==edges.dst["particle_number"])
        return {"same_particle": distance}
    return func


class EFDeepSet(L.LightningModule):
    def __init__(self,args,dev):
        super().__init__()
        node_indim= 10 #10 is input dim because there are two nodes with 5 input dim
        edge_indim=2
        hidden_dim=128
        self.fcnn = nn.Sequential(
            nn.Linear(node_indim+edge_indim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
        )

    def forward(self, g, y, step_count, eval=""):
        i = g.edges()[0]
        j = g.edges()[1]
        g.apply_edges(z0())
        g.apply_edges(theta_slope())
        g.apply_edges(edge_concat_data())
        g.apply_edges(self.fccn_())
        # x = self.fcnn(g.edata["x"])

        return g.edata["weight"]
    
    def fccn_(self):
        def func(edges):
            # print("nans edata", torch.sum(torch.isnan(edges.data["x"])))
            # print("min", torch.min(edges.data["x"]),  torch.max(edges.data["x"]))
            edges.data["x"][edges.data["x"]==-np.inf]=0
            x = self.fcnn(edges.data["x"])
            # print("nans output", torch.sum(torch.isnan(x)))
            return {"weight": x}
        return func
    
    def training_step(self, batch, batch_idx):
        # print("training step, ", self.trainer.is_global_zero)
        y = batch[1]

        batch_g = batch[0]

        if self.trainer.is_global_zero:
            model_output = self(batch_g, y, batch_idx)
        else:
            model_output = self(batch_g, y, 1)
        batch_g.apply_edges(same_particle())
        # print(model_output.shape, batch_g.edata["same_particle"].shape)
        loss = binary_focal_loss(model_output.view(-1), batch_g.edata["same_particle"])
        wandb.log(
            {
                "binary focal loss": loss.item(),
            })
        self.loss_final = loss
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        model_output = self(batch_g, y, batch_idx, eval="_val")
        batch_g.apply_edges(same_particle())
        # print(model_output.shape, batch_g.edata["same_particle"].shape)
        loss = binary_focal_loss(model_output.view(-1), batch_g.edata["same_particle"])
        wandb.log(
            {
                "binary focal loss val": loss.item(),
            })
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        # print("validation step", batch_idx, loss)
       

    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.loss_final)


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


def binary_focal_loss(
    inputs,
    targets,
    alpha=0.3,
    gamma=2.0):
    """Extracted function for JIT compilation."""
    # probs_pos = inpt
    # probs_neg = 1 - inpt
    # p = torch.sigmoid(inputs)
    # print(inputs)
    # print("nans inputs", torch.sum(torch.isnan(inputs)))
    # print(targets)
    # print("nans targets", torch.sum(torch.isnan(targets)))
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    # p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)

    # if alpha >= 0:
    #     alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    #     loss = alpha_t * loss



    # print(probs_pos)
    # pos_term = -alpha * probs_neg.pow(gamma) * target * probs_pos.log()
    # # print("nansum", torch.sum(torch.isnan(probs_pos.log())))
    # neg_term = -(1.0 - alpha) * probs_pos.pow(gamma) * (1.0 - target) * probs_neg.log()
    # loss_tmp = pos_term + neg_term



    # p_t = inpt * target + (1 - inpt) * (1 - target)
    # print(torch.sum(torch.isnan(ce_loss)))
    # print(ce_loss)
    return torch.mean(ce_loss)