import dgl
import torch
import os
from sklearn.cluster import DBSCAN
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import wandb
from sklearn.cluster import DBSCAN, HDBSCAN


def hfdb_obtain_labels(X, device, eps=0.1):
    # hdbscan gives -1 if noise.. 1 if +
    hdb = HDBSCAN(min_cluster_size=8, min_samples=8, cluster_selection_epsilon=eps).fit(
        X.detach().cpu()
    )
    # hdb = DBSCAN(min_samples=10, eps=0.5).fit(X.detach().cpu())
    labels_hdb = hdb.labels_ + 1  # noise class goes to zero
    labels_hdb = np.reshape(labels_hdb, (-1))
    labels_hdb = torch.Tensor(labels_hdb).long().to(device)
    return labels_hdb


def evaluate_efficiency_tracks(
    batch_g,
    model_output,
    y,
    local_rank,
    step,
    epoch,
    path_save,
    store=False,
    predict=False,
    ct=False,
    clustering_mode="dbscan",
    tau=False
):
    number_of_showers_total = 0
    if not ct:
        batch_g.ndata["coords"] = model_output[:, 0:3]
        batch_g.ndata["beta"] = model_output[:, 3]
    else:
        batch_g.ndata["model_output"] = model_output
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)
    df_list = []
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        dic["part_true"] = y[mask]

        betas = torch.sigmoid(dic["graph"].ndata["beta"])
        X = dic["graph"].ndata["coords"]
        if ct:

            # labels can start at -1 in which case the 0 is the 'noise'
            labels_ = graphs[i].ndata["model_output"].long() + 1
            map_from = list(np.unique(labels_.detach().cpu()))
            labels = map(lambda x: map_from.index(x), labels_)
            labels = (
                torch.Tensor(list(labels))
                .long()
                .to(dic["graph"].ndata["coords"].device)
            )
        else:
            if clustering_mode == "clustering_normal":
                clustering1 = get_clustering(betas, X, tbeta=0.6, td=0.2)
                map_from = list(np.unique(clustering1.detach().cpu()))
                cluster_id = map(lambda x: map_from.index(x), clustering1)
                clustering_ordered = (
                    torch.Tensor(list(cluster_id)).long().to(model_output.device)
                )
                if torch.unique(clustering1)[0] != -1:
                    clustering = clustering_ordered + 1
                else:
                    clustering = clustering_ordered

                labels = clustering.view(-1).long()
            elif clustering_mode == "dbscan":
                labels = hfdb_obtain_labels(X, betas.device)

        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
        # print("particle_ids", particle_ids)
        shower_p_unique = torch.unique(labels)
        # print("shower_p_unique", shower_p_unique)
        (
            shower_p_unique,
            row_ind,
            col_ind,
            i_m_w,
            i_m_w_unique,
            iou_matrix,
        ) = match_showers(
            labels,
            dic,
            particle_ids,
            model_output,
            local_rank,
            i,
            path_save,
        )

        if len(row_ind) > 1:
            df_event, number_of_showers_total = generate_showers_data_frame(
                labels,
                dic,
                shower_p_unique,
                particle_ids,
                row_ind,
                col_ind,
                i_m_w,
                i_m_w_unique,
                number_of_showers_total=number_of_showers_total,
                step=step,
                number_in_batch=i,
                tau=tau
            )
            # if len(shower_p_unique) < len(particle_ids):
            # print("storing  event", local_rank, step, i)
            # torch.save(
            #     dic,
            #     path_save
            #     + "/graphs_2810/"
            #     + str(local_rank)
            #     + "_"
            #     + str(step)
            #     + "_"
            #     + str(i)
            #     + ".pt",
            # )
            df_list.append(df_event)
    if len(df_list) > 0:
        df_batch = pd.concat(df_list)
    else:
        df_batch = []
    if store:
        store_at_batch_end(
            path_save, df_batch, local_rank, step, epoch, predict=predict
        )
    return df_batch


def store_at_batch_end(
    path_save,
    df_batch,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
):
    path_save_ = (
        path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(epoch) + "0_0_0_Zjj_4000_5000_v1.pt"
    )
    if predict:
        df_batch = pd.concat(df_batch)
        df_batch.to_pickle(path_save_)

    # log_efficiency(df_batch)


def log_efficiency(df):
    # take the true showers non nan
    if len(df) > 0:
        # drop fake showers with less than 4 hits
        mask_nan_showers = np.isnan(df["reco_showers_E"].values) * (
            df["pred_showers_E"].values < 4
        )
        df = df.drop(mask_nan_showers.nonzero()[0])
        mask = ~np.isnan(df["reco_showers_E"])
        showers_are_not_nan = ~np.isnan(df["pred_showers_E"][mask].values)
        showers_have_higher_than_75 = (
            df["e_pred_and_truth"][mask].values / df["reco_showers_E"][mask].values
        ) >= 0.75
        eff = np.sum(showers_are_not_nan * showers_have_higher_than_75) / len(
            df["pred_showers_E"][mask].values
        )

        # mask = ~np.isnan(df["reco_showers_E"])
        # eff = np.sum(~np.isnan(df["pred_showers_E"][mask].values)) / len(
        #     df["pred_showers_E"][mask].values
        # )
        wandb.log({"efficiency validation": eff})


def calculate_number_of_unique_hits_per_particle(labels, dic):

    unique_labels = torch.unique(labels)
    if torch.sum(labels == 0) == 0:
        # if there is no zero MC adds one
        unique_labels = torch.cat(
            (
                torch.Tensor([0]).to(unique_labels.device).view(-1),
                unique_labels.view(-1),
            ),
            dim=0,
        )
    number_of_unique_hits_all = []
    for i in range(0, len(unique_labels)):
        mask_label = labels == unique_labels[i]
        number_of_unique_hits = len(
            torch.unique(dic["graph"].ndata["unique_id"][mask_label])
        )

        number_of_unique_hits_all.append(number_of_unique_hits)

    return torch.Tensor(np.array(number_of_unique_hits_all)).to(labels.device).view(-1)


def generate_showers_data_frame(
    labels,
    dic,
    shower_p_unique,
    particle_ids,
    row_ind,
    col_ind,
    i_m_w,
    i_m_w_unique,
    number_of_showers_total=None,
    step=0,
    number_in_batch=0,
    tau=False
):
    # calculate number of unique_hits_per_shower
    unique_hits_per_MC = calculate_number_of_unique_hits_per_particle(
        dic["graph"].ndata["particle_number"].long(), dic
    )

    unique_hits_per_MC = unique_hits_per_MC[1:].view(-1)

    unique_hits_per_reconstructed_track = calculate_number_of_unique_hits_per_particle(
        labels, dic
    )
    # independent of labels having a 0 or not this creates a 0
    e_pred_showers = 1.0 * scatter_add(
        torch.ones_like(labels).view(-1),
        labels.long(),
    )

    # MC particles without noise 'particle'
    e_reco_showers = scatter_add(
        torch.ones_like(labels).view(-1),
        dic["graph"].ndata["particle_number"].long(),
    )
    e_reco_showers = e_reco_showers[1:]
    
    if tau:
        tau_mom_per_particle = scatter_mean(
            dic["graph"].ndata["tau_mom"].view(-1),
            dic["graph"].ndata["particle_number"].long(),
        )
        tau_mom_per_particle = tau_mom_per_particle[1:]

    dic["graph"].ndata["hit_type_0"] = 1 * (dic["graph"].ndata["hit_type"] == 0)
    number_cdc_hits = scatter_add(
        dic["graph"].ndata["hit_type_0"].view(-1),
        dic["graph"].ndata["particle_number"].long(),
    )

    number_cdc_hits = number_cdc_hits[1:]

    dic["graph"].ndata["hit_type_1"] = 1 * (dic["graph"].ndata["hit_type"] == 1)
    number_vtx_hits = scatter_add(
        dic["graph"].ndata["hit_type_1"].view(-1),
        dic["graph"].ndata["particle_number"].long(),
    )
    number_vtx_hits = number_vtx_hits[1:]

    delta_MC = calculate_delta_MC(dic)
    e_true_showers = dic["part_true"][:, 5]  #
    pt_true_showers = dic["part_true"][:, 6]  #
    genstatus_true_showers = dic["part_true"][:, 7]  #
    R_showers = dic["part_true"][:, 4]  #
    theta_showers = dic["part_true"][:, 0]  #
    row_ind = torch.Tensor(row_ind).to(e_pred_showers.device).long()
    col_ind = torch.Tensor(col_ind).to(e_pred_showers.device).long()
    if torch.sum(particle_ids == 0) > 0:
        # particle id can be 0 because there is noise
        # then row ind 0 in any case corresponds to particle 1.
        # if there is particle_id 0 then row_ind should be +1?
        row_ind_ = row_ind - 1
    else:
        # if there is no zero then index 0 corresponds to particle 1.
        row_ind_ = row_ind

    pred_showers = shower_p_unique
    index_matches = col_ind + 1
    index_matches = index_matches.to(e_pred_showers.device).long()
    matched_es = torch.zeros_like(e_reco_showers) * (torch.nan)
    matched_es = matched_es.to(e_pred_showers.device)

    matched_es[row_ind_] = e_pred_showers[index_matches]

    # unique_hits_matched_reconstructed = torch.zeros_like(e_reco_showers) * (torch.nan)
    # unique_hits_matched_reconstructed = unique_hits_matched_reconstructed.to(
    #     e_pred_showers.device
    # )

    # unique_hits_matched_reconstructed[row_ind_] = unique_hits_per_reconstructed_track[
    #     index_matches.long()
    # ]
    unique_hits_matched_reconstructed = torch.zeros_like(e_reco_showers) * (torch.nan)
    ie_unique = obtain_intersection_values(i_m_w_unique, row_ind, col_ind, particle_ids)
    unique_hits_matched_reconstructed[row_ind_] = ie_unique.to(e_pred_showers.device)

    intersection_E = torch.zeros_like(e_reco_showers) * (torch.nan)

    ie = obtain_intersection_values(i_m_w, row_ind, col_ind, particle_ids)

    intersection_E[row_ind_] = ie.to(e_pred_showers.device)

    pred_showers[index_matches] = -1
    pred_showers[
        0
    ] = (
        -1
    )  # this takes into account that the class 0 for pandora and for dbscan is noise
    mask = pred_showers != -1
    fake_showers_e = e_pred_showers[mask]

    fake_showers_nuhr = unique_hits_per_reconstructed_track[mask]

    fake_showers_showers_e_truw = torch.zeros((fake_showers_e.shape[0])) * (torch.nan)
    fake_showers_showers_e_truw = fake_showers_showers_e_truw.to(e_pred_showers.device)
    e_reco = torch.cat((e_reco_showers, fake_showers_showers_e_truw), dim=0)
    vtx_hits = torch.cat((number_vtx_hits, fake_showers_showers_e_truw), dim=0)
    cdc_hits = torch.cat((number_cdc_hits, fake_showers_showers_e_truw), dim=0)

    e_true = torch.cat((e_true_showers, fake_showers_showers_e_truw), dim=0)
    pt_true = torch.cat((pt_true_showers, fake_showers_showers_e_truw), dim=0)
    gen_status_true = torch.cat(
        (genstatus_true_showers, fake_showers_showers_e_truw), dim=0
    )
    R_true = torch.cat((R_showers, fake_showers_showers_e_truw), dim=0)
    theta_true = torch.cat((theta_showers, fake_showers_showers_e_truw), dim=0)
    e_pred = torch.cat((matched_es, fake_showers_e), dim=0)
    # number_uhr = torch.cat(
    #     (unique_hits_matched_reconstructed, fake_showers_nuhr), dim=0
    # )
    number_uh = torch.cat((unique_hits_per_MC, fake_showers_showers_e_truw), dim=0)
    delta_MC_ = torch.cat((delta_MC, fake_showers_showers_e_truw), dim=0)
    e_pred_t = torch.cat(
        (
            intersection_E,
            torch.zeros_like(fake_showers_e) * (torch.nan),
        ),
        dim=0,
    )

    number_uhr = torch.cat(
        (
            unique_hits_matched_reconstructed,
            torch.zeros_like(fake_showers_e) * (torch.nan),
        ),
        dim=0,
    )
    if tau:
        tau_mom = torch.cat((tau_mom_per_particle, fake_showers_showers_e_truw), dim=0)
    d = {
        "reco_showers_E": e_reco.detach().cpu(),
        "true_showers_E": e_true.detach().cpu(),
        "true_showers_pt": pt_true.detach().cpu(),
        "pred_showers_E": e_pred.detach().cpu(),
        "e_pred_and_truth": e_pred_t.detach().cpu(),
        "vtx_hits": vtx_hits.detach().cpu(),
        "cdc_hits": cdc_hits.detach().cpu(),
        "delta_MC": delta_MC_.detach().cpu(),
        "R": R_true.detach().cpu(),
        "theta": theta_true.detach().cpu(),
        "gen_status": gen_status_true.detach().cpu(),
        "number_unique_hits": number_uh.detach().cpu(),
        "number_unique_hits_reconstructed": number_uhr.detach().cpu(),
    }
    
    if tau:
        d["tau_mom"]= tau_mom.detach().cpu()
    df = pd.DataFrame(data=d)
    if number_of_showers_total is None:
        return df
    else:
        return df, number_of_showers_total


def obtain_intersection_matrix(shower_p_unique, particle_ids, labels, dic, unique_ids):
    len_pred_showers = len(shower_p_unique)
    intersection_matrix = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )
    intersection_matrix_w = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )
    intersection_matrix_w_unique = torch.zeros(
        (len_pred_showers, len(particle_ids))
    ).to(shower_p_unique.device)
    e_hits = torch.ones_like(dic["graph"].ndata["unique_id"])
    unique_labels = torch.unique(labels)
    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        unique_ids = unique_ids.clone()
        h_hits = e_hits.clone()
        counts[mask_p] = 1
        h_hits[~mask_p] = 0
        intersection_matrix_w[:, index] = scatter_add(h_hits, labels)
        intersection_matrix[:, index] = scatter_add(counts, labels)
        for j in range(0, len(unique_labels)):
            mask_label_j = labels == unique_labels[j]
            number_of_unique_labels_in_label_j_and_MC = len(
                torch.unique(dic["graph"].ndata["unique_id"][mask_label_j * mask_p])
            )
            intersection_matrix_w_unique[
                j, index
            ] = number_of_unique_labels_in_label_j_and_MC
    return intersection_matrix, intersection_matrix_w, intersection_matrix_w_unique


def calculate_delta_MC(dic):
    pseudorapidity = -torch.log(torch.tan(dic["part_true"][:, 0] / 2))
    phi = dic["part_true"][:, 1]
    x1 = torch.cat((pseudorapidity.view(-1, 1), phi.view(-1, 1)), dim=1)
    distance_matrix = torch.cdist(x1, x1, p=2)
    values, _ = torch.sort(distance_matrix, dim=1)
    delta_MC = values[:, 1]
    return delta_MC


def obtain_union_matrix(shower_p_unique, particle_ids, labels, dic):
    len_pred_showers = len(shower_p_unique)
    union_matrix = torch.zeros((len_pred_showers, len(particle_ids)))

    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        for index_pred, id_pred in enumerate(shower_p_unique):
            mask_pred_p = labels == id_pred
            mask_union = mask_pred_p + mask_p
            union_matrix[index_pred, index] = torch.sum(mask_union)

    return union_matrix


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.7, td=0.05):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points).to(betas.device)
    clustering = -1 * torch.ones(n_points, dtype=torch.long).to(betas.device)
    while len(indices_condpoints) > 0 and len(unassigned) > 0:
        index_condpoint = indices_condpoints[0]
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
        # calculate indices_codpoints again
        indices_condpoints = find_condpoints(betas, unassigned, tbeta)
    return clustering


def find_condpoints(betas, unassigned, tbeta):
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    device = betas.device
    mask_unassigned = torch.zeros(n_points).to(device)
    mask_unassigned[unassigned] = True
    select_condpoints = mask_unassigned.to(bool) * select_condpoints
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    return indices_condpoints


def obtain_intersection_values(intersection_matrix_w, row_ind, col_ind, particle_ids):
    list_intersection_E = []
    # print(intersection_matrix_w.shape)
    # print(row_ind)
    # print(col_ind)
    # intersection_matrix_w = intersection_matrix_w
    if torch.sum(particle_ids == 0) > 0:
        # removing also the MC particle corresponding to noise
        intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, 1:], 1, 0)
        row_ind = row_ind - 1
    else:
        intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, :], 1, 0)
    for i in range(0, len(col_ind)):
        list_intersection_E.append(
            intersection_matrix_wt[row_ind[i], col_ind[i]].view(-1)
        )
    return torch.cat(list_intersection_E, dim=0)


def plot_iou_matrix(iou_matrix, image_path):
    iou_matrix = torch.transpose(iou_matrix[1:, :], 1, 0)
    fig, ax = plt.subplots()
    iou_matrix = iou_matrix.detach().cpu().numpy()
    ax.matshow(iou_matrix, cmap=plt.cm.Blues)
    for i in range(0, iou_matrix.shape[1]):
        for j in range(0, iou_matrix.shape[0]):
            c = np.round(iou_matrix[j, i], 1)
            ax.text(i, j, str(c), va="center", ha="center")
    fig.savefig(image_path, bbox_inches="tight")
    wandb.log({"iou_matrix": wandb.Image(image_path)})


def match_showers(
    labels,
    dic,
    particle_ids,
    model_output,
    local_rank,
    i,
    path_save,
):
    iou_threshold = 0.02
    shower_p_unique = torch.unique(labels)
    if torch.sum(labels == 0) == 0:
        # if there is no zero it adds one
        shower_p_unique = torch.cat(
            (
                torch.Tensor([0]).to(shower_p_unique.device).view(-1),
                shower_p_unique.view(-1),
            ),
            dim=0,
        )
    # all hits weight the same
    e_hits = dic["graph"].ndata["unique_id"]
    i_m, i_m_w, i_m_w_unique = obtain_intersection_matrix(
        shower_p_unique, particle_ids, labels, dic, e_hits
    )
    i_m = i_m.to(model_output.device)
    i_m_w = i_m_w.to(model_output.device)
    i_m_w_unique = i_m_w_unique.to(model_output.device)
    u_m = obtain_union_matrix(shower_p_unique, particle_ids, labels, dic)
    u_m = u_m.to(model_output.device)
    iou_matrix = i_m / u_m

    # taking from index 1 here excludes 0 the noise track from the predicted tracks
    if torch.sum(particle_ids == 0) > 0:
        # removing also the MC particle corresponding to noise
        iou_matrix_num = (
            torch.transpose(iou_matrix[1:, 1:], 1, 0).clone().detach().cpu().numpy()
        )
    else:
        iou_matrix_num = (
            torch.transpose(iou_matrix[1:, :], 1, 0).clone().detach().cpu().numpy()
        )

    iou_matrix_num[iou_matrix_num < iou_threshold] = 0
    row_ind, col_ind = linear_sum_assignment(
        -iou_matrix_num
    )  # row_ind are particles that are matched and col_ind the ind of preds they are matched to
    # next three lines remove solutions where there is a shower that is not associated and iou it's zero (or less than threshold)
    mask_matching_matrix = iou_matrix_num[row_ind, col_ind] > 0
    row_ind = row_ind[mask_matching_matrix]
    col_ind = col_ind[mask_matching_matrix]
    if torch.sum(particle_ids == 0) > 0:
        row_ind = row_ind + 1
    # if i == 0 and local_rank == 0:
    #     if path_save is not None:
    #         image_path = path_save + "/example_1_clustering.png"
    #         plot_iou_matrix(iou_matrix, image_path)

    return shower_p_unique, row_ind, col_ind, i_m_w, i_m_w_unique, iou_matrix
