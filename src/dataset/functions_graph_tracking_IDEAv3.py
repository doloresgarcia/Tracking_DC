import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum, scatter_min, scatter_max
from sklearn.preprocessing import StandardScaler
import time

from src.dataset.functions_graph_tracking import check_unique_particles,fix_splitted_tracks,find_cluster_id,remove_loopers

def create_inputs_from_table(output, get_vtx, cld=False):
    
    graph_empty = False
    number_hits = np.int32(np.sum(output["pf_mask"][0]))

    number_part = np.int32(np.sum(output["pf_mask"][1]))
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    hit_particle_link_tau = None
    
    points_hits_vtx = torch.permute(
        torch.tensor(output["pf_points_vtx"][:, 0:number_hits]), (1, 0)
    )
    
    points_hits_dc = torch.permute(
        torch.tensor(output["pf_points_dc"][:, 0:number_hits]), (1, 0)
    )
    
    features_hits = torch.permute(
        torch.tensor(output["pf_features"][:, 0:number_hits]), (1, 0)
    )
    
    hit_type = features_hits[:, 0].clone()
    
    unique_list_particles = list(np.unique(hit_particle_link))
    unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)
    features_particles = torch.permute(
        torch.tensor(output["pf_vectors"][:, 0:number_part]),
        (1, 0),
    )

    y_data_graph = features_particles
    y_id = features_particles[:, 4]
    
    # check if they need
    mask_particles = check_unique_particles(unique_list_particles, y_id)
    y_data_graph = features_particles[mask_particles]
    
    if features_particles.shape[0] >= torch.sum(mask_particles).item():
        hit_particle_link = fix_splitted_tracks(hit_particle_link, y_data_graph)

        cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
        unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)

        features_particles = torch.permute(
            torch.tensor(output["pf_vectors"][:, 0:number_part]),
            (1, 0),
        )

        y_id = features_particles[:, 4]
        mask_particles = check_unique_particles(unique_list_particles, y_id)
        
        y_data_graph = features_particles[mask_particles]
        # print(y_data_graph.shape, unique_list_particles.shape)
        assert len(y_data_graph) == len(unique_list_particles)
        
    else:
        graph_empty = True
 
    if graph_empty:
        return [None]
    
    else:
        result = [
            y_data_graph,
            cluster_id,
            hit_particle_link,
            points_hits_vtx,
            points_hits_dc,
            features_hits,
            hit_type
            ]
        
        return result

def create_graph_tracking_global_v3(output, get_vtx=False, vector=False, overlay=False):
    
    graph_empty = False
    result = create_inputs_from_table(output, get_vtx)
    
    if len(result) == 1:
        graph_empty = True
    else:
        (
            y_data_graph,
            cluster_id,
            hit_particle_link,
            points_hits_vtx,
            points_hits_dc,
            features_hits,
            hit_type
        ) = result
        
        if not overlay:
            mask_not_loopers, mask_particles = remove_loopers(
                hit_particle_link, y_data_graph, points_hits_dc, cluster_id
            )

            cluster_id = cluster_id[mask_not_loopers]
            hit_particle_link = hit_particle_link[mask_not_loopers]
  
            points_hits_dc = points_hits_dc[mask_not_loopers]
            points_hits_vtx = points_hits_vtx[mask_not_loopers]
            features_hits = features_hits[mask_not_loopers]
            hit_type = hit_type[mask_not_loopers]
            
            y_data_graph = y_data_graph[mask_particles]
            cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)

        if cluster_id.shape[0] > 0:
            mask_dc = hit_type == 0
            mask_vtx = hit_type == 1
            number_of_vtx = torch.sum(mask_vtx)
            number_of_dc = torch.sum(mask_dc)
            g = dgl.DGLGraph()
            g.add_nodes(number_of_vtx + number_of_dc)
            
            dc_points = points_hits_dc[mask_dc]
            vtx_points = points_hits_vtx[mask_vtx]
            
            vector_like_data = vector
            particle_number = torch.cat(
                        (cluster_id[mask_vtx], cluster_id[mask_dc]), dim=0
                    )
            
            coordinate_hits = torch.cat(
                (vtx_points, points_hits_vtx[mask_dc]*0), dim=0
            ) 
            
            coordinate_planes = torch.cat(
                (points_hits_dc[mask_vtx]*0, dc_points), dim=0
            ) 
            
            coordinate_hits_withDC = torch.cat(
                (vtx_points, points_hits_vtx[mask_dc]), dim=0
            ) 
                    
            particle_number_nomap = torch.cat(
                (
                    hit_particle_link[mask_vtx],
                    hit_particle_link[mask_dc],
                ),
                dim=0,
            )
            
            hit_type_all = torch.cat(
                (hit_type[mask_vtx], hit_type[mask_dc]), dim=0
            )
                       
            g.ndata["hit_type"] = hit_type_all
            g.ndata["particle_number"] = particle_number
            g.ndata["particle_number_nomap"] = particle_number_nomap
            g.ndata["coordinate_hits"] = coordinate_hits
            g.ndata["coordinate_planes"] = coordinate_planes
            g.ndata["true_coordinates"] = coordinate_hits_withDC
            
            if len(y_data_graph) < 1:
                graph_empty = True
                
            if features_hits.shape[0] < 10:
                graph_empty = True
        else:
            graph_empty = True
            
    if graph_empty:
        g = 0
        y_data_graph = 0
    return [g, y_data_graph], graph_empty

