import torch
import copy
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from rich.console import Console
from tqdm import tqdm
CONSOLE = Console(width=120)
from collections import defaultdict
from utils.calc_utils import *
from warp import S2NRaySamples
from utils.loss_utils import error_map_L1

class dynamicGS_term:
    regular_xyz = None
    regular_xyz_joint = None
    
    
    @staticmethod
    def _detach_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        elif isinstance(obj, (list, tuple)):

            return type(obj)(dynamicGS_term._detach_tensors(x) for x in obj)
        elif isinstance(obj, dict):

            return {k: dynamicGS_term._detach_tensors(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):

            new_obj = copy.copy(obj)
            for k, v in obj.__dict__.items():
                setattr(new_obj, k, dynamicGS_term._detach_tensors(v))
            return new_obj
        else:

            return obj
        
    @classmethod
    def load_gaussian(cls, GaussianModel):
        GaussianModel_copy = cls._detach_tensors(GaussianModel)
        cls.canonical_gaussians = GaussianModel_copy
        cls.canonical_gaussians._xyz.requires_grad = False
        cls.canonical_gaussians._features_dc.requires_grad = False
        cls.canonical_gaussians._features_rest.requires_grad = False
        cls.canonical_gaussians._scaling.requires_grad = False
        cls.canonical_gaussians._rotation.requires_grad = False
        cls.canonical_gaussians._opacity.requires_grad = False

    @classmethod
    def load_gaussian_joint(cls, GaussianModel):
        cls.canonical_gaussians_joint = copy.deepcopy(GaussianModel)
        cls.canonical_gaussians._xyz.requires_grad = False
        cls.canonical_gaussians._features_dc.requires_grad = False
        cls.canonical_gaussians._features_rest.requires_grad = False
        cls.canonical_gaussians._scaling.requires_grad = False
        cls.canonical_gaussians._rotation.requires_grad = False
        cls.canonical_gaussians._opacity.requires_grad = False


    @classmethod
    def regular_term_setup(cls, GaussianModel, velocity_option = True):
        if cls.regular_xyz == None:
            cls.regular_xyz = GaussianModel.get_xyz.clone().detach()
        cls.xyz_velocity = torch.zeros_like(GaussianModel.get_xyz)    

        if velocity_option:
            cls.xyz_velocity = GaussianModel.get_xyz.clone().detach() - cls.regular_xyz


        cls.regular_xyz = GaussianModel.get_xyz.clone().detach()
        cls.regular_features_dc = GaussianModel.get_features_dc.clone().detach()
        cls.regular_features_rest = GaussianModel.get_features_rest.clone().detach()
        cls.regular_scaling = GaussianModel.get_scaling_ori.clone().detach()
        cls.regular_rotation = GaussianModel.get_rotation.clone().detach()
        cls.regular_opacity = GaussianModel.get_opacity_ori.clone().detach()
        print(cls.regular_rotation.shape)
        cls.pre_rotations_inv = quaternion_inverse(cls.regular_rotation)
        cls.prev_neighbor_points = cls.regular_xyz[cls.indices_]
        cls.prev_diff = cls.regular_xyz.unsqueeze(1) - cls.prev_neighbor_points


    @classmethod
    def regular_term_prune(cls, mask):

        valid_points_mask = ~mask
        cls.regular_xyz = cls.regular_xyz[valid_points_mask]
        cls.regular_features_dc = cls.regular_features_dc[valid_points_mask]
        cls.regular_features_rest = cls.regular_features_rest[valid_points_mask]
        cls.regular_scaling = cls.regular_scaling[valid_points_mask]
        cls.regular_rotation = cls.regular_rotation[valid_points_mask]
        cls.regular_opacity = cls.regular_opacity[valid_points_mask]

        cls.pre_rotations_inv = quaternion_inverse(cls.regular_rotation)
        cls.prev_neighbor_points = cls.regular_xyz[cls.indices_]
        cls.prev_diff = cls.regular_xyz.unsqueeze(1) - cls.prev_neighbor_points

    @classmethod
    def regular_term_extend(cls, gaussians):
        """Extend reference attributes and neighbor offsets for new Gaussians."""
        current_count = cls.regular_xyz.shape[0]
        new_count = gaussians.get_xyz.shape[0]
        added_count = new_count - current_count
        
        if added_count <= 0:
            return


        device = cls.regular_xyz.device


        new_xyz = gaussians.get_xyz[current_count:].clone().detach()
        cls.regular_xyz = torch.cat([cls.regular_xyz, new_xyz], dim=0)


        new_features_dc = gaussians.get_features_dc[current_count:].clone().detach()
        cls.regular_features_dc = torch.cat([cls.regular_features_dc, new_features_dc], dim=0)

        new_features_rest = gaussians.get_features_rest[current_count:].clone().detach()
        cls.regular_features_rest = torch.cat([cls.regular_features_rest, new_features_rest], dim=0)


        new_scaling = gaussians.get_scaling_ori[current_count:].clone().detach()
        cls.regular_scaling = torch.cat([cls.regular_scaling, new_scaling], dim=0)


        new_rotation = gaussians.get_rotation[current_count:].clone().detach()
        cls.regular_rotation = torch.cat([cls.regular_rotation, new_rotation], dim=0)


        new_opacity = gaussians.get_opacity_ori[current_count:].clone().detach()
        cls.regular_opacity = torch.cat([cls.regular_opacity, new_opacity], dim=0)


        cls.pre_rotations_inv = quaternion_inverse(cls.regular_rotation)
        

        cls.prev_neighbor_points = cls.regular_xyz[cls.indices_]
        cls.prev_diff = cls.regular_xyz.unsqueeze(1) - cls.prev_neighbor_points


    @classmethod
    def graph_init(cls, GaussianModel, filename = None, k=8, load_graph_path=None):
        CONSOLE.log('Initializing graph')


        points_np = GaussianModel.get_xyz.detach().cpu().numpy()

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points_np)
        cls.dist_, cls.indices_ = nbrs.kneighbors(points_np)

        cls.dist_ = torch.tensor(cls.dist_[:,1:], dtype=torch.float32, requires_grad=False).cuda()
        cls.indices_ = torch.tensor(cls.indices_[:,1:], device=cls.dist_.device, dtype=torch.long)
        CONSOLE.log(f"rigid loss init shape: {cls.indices_.shape}")
        epsilon = 1e-6
        cls.graph_weights_ = 1.0 / (cls.dist_  + epsilon).detach()
        cls.graph_weights_ = cls.graph_weights_ / cls.graph_weights_.sum(dim=1, keepdim=True)

        cls.graph_weights_ = cls.graph_weights_.unsqueeze(-1)


        cls.indices_ = cls.indices_.to(dtype=torch.long).detach()


        if load_graph_path:
            indices_path = os.path.join(load_graph_path, 'indices.npy')
            if os.path.exists(indices_path):
                CONSOLE.log(f"Loading indices from {indices_path}")
                cls.indices_ = torch.tensor(np.load(indices_path), device=cls.dist_.device, dtype=torch.long)
            else:
                CONSOLE.log(f"Saving indices to {indices_path}")
                np.save(indices_path, cls.indices_.cpu().numpy())
        CONSOLE.log('Graph initialized')
    
 
    @classmethod
    def extend_knn_graph(cls, selected_pts_mask, gaussians, N = 1):
        """Let new Gaussians inherit their source points' neighbors."""
        num_existing_points = selected_pts_mask.shape[0]

        selected_indices = torch.where(selected_pts_mask)[0]
        selected_indices = selected_indices.repeat(N)

        num_new_points = selected_indices.shape[0]

        if num_new_points == 0:
            print("No new points selected, skipping k-NN graph extension.")
            return


        new_indices = cls.indices_[selected_indices].clone()
        new_weights = cls.graph_weights_[selected_indices].clone() / 5


        cls.indices_ = torch.cat([cls.indices_, new_indices], dim=0)
        cls.graph_weights_ = torch.cat([cls.graph_weights_, new_weights], dim=0)
        cls.graph_weights_.requires_grad = False
        cls.regular_term_extend(gaussians)
        print(f"Extended k-NN graph: Added {num_new_points} new points.")


    @classmethod
    def graph_update(cls, gaussians, prune_mask, current_ring, rings=1, k=8, local=True):
        """Update affected neighbors after pruning and return the compact update mask."""
        points = gaussians.get_xyz
        device = points.device
        N = points.shape[0]


        for _ in range(rings):
            current_ring_indices = torch.where(current_ring)[0]

            current_ring[cls.indices_[current_ring_indices].flatten().unique()] = True
        
        valid_mask = ~prune_mask

        if local:
            neighbor_indices = cls.indices_[valid_mask]
            neighbor_valid = valid_mask[neighbor_indices]
            has_invalid_neighbor = ~neighbor_valid.all(dim=1)
            dirty_idx = torch.where(valid_mask)[0][has_invalid_neighbor]
    
            update_idx = torch.where(current_ring)[0]
            update_idx = torch.unique(torch.cat([update_idx, dirty_idx]))
        else:
            update_idx = torch.where(valid_mask)[0]

        points_update = points[update_idx].clone().detach().cpu().numpy()
        points_valid = points[valid_mask].clone().detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points_valid)
        dist_, indices_ = nbrs.kneighbors(points_update)
        print(points_update.shape, points_valid.shape)
        dist_ = torch.tensor(dist_, dtype=torch.float32, requires_grad=False, device=device)
        indices_ = torch.tensor(indices_, device=device, dtype=torch.long)
        indices_.requires_grad = False

        indices_ = indices_[..., 1:]
        dist_ = dist_[..., 1:]


        valid_idx = torch.where(valid_mask)[0]
        nn_idx_global = valid_idx[indices_]  

        inv_dist = 1.0 / (dist_ + 1e-6)
        weights = (inv_dist / inv_dist.sum(dim=1, keepdim=True)).unsqueeze(-1)

        cls.indices_[update_idx] = nn_idx_global
        cls.graph_weights_[update_idx] = weights
        cls.graph_weights_.requires_grad_(False)


        cls.indices_ = cls.indices_[valid_mask]
        cls.graph_weights_ = cls.graph_weights_[valid_mask]

        cls.graph_weights_[gaussians.last_frame_points:] = cls.graph_weights_[gaussians.last_frame_points:] / 5

        valid_idx = torch.where(valid_mask)[0]
        global_to_compact = -torch.ones(N, dtype=torch.long, device=device)

        global_to_compact[valid_idx] = torch.arange(valid_idx.shape[0], device=device)

        cls.indices_ = global_to_compact[cls.indices_]  # Map the global indices to compact space


        return current_ring[valid_mask]


    @classmethod
    def new_points_graph_update(cls, gaussians, k=8):

        points = gaussians.get_xyz
        device = points.device
        N = points.shape[0]

        new_points_mask = torch.zeros(N, dtype=torch.bool, device=device)
        new_points_mask[gaussians.last_frame_points:] = True

        full_dist = torch.cdist(points[new_points_mask], points[:gaussians.last_frame_points])
        print(full_dist.shape)

        dist, nn_idx = torch.topk(full_dist, k=k+1, dim=1, largest=False)


        nn_idx = nn_idx[:, 1:]
        precise_dist = dist[:, 1:]

        inv_dist = 1.0 / (precise_dist + 1e-6)
        weights = (inv_dist / inv_dist.sum(dim=1, keepdim=True)).unsqueeze(-1)


        new_points_indices_ = nn_idx
        new_points_weights_ = weights

        cls.indices_ = torch.cat([cls.indices_, new_points_indices_], dim=0)
        cls.graph_weights_ = torch.cat([cls.graph_weights_, new_points_weights_], dim=0)

    
    @classmethod
    def save_graph_to_file(cls, gaussians, filename, sign_mask=None, valid_points_mask=None):
        """Save an OBJ graph, marking sign_mask points in red."""
        mkdir_p(os.path.dirname(filename))

        points = gaussians.get_xyz.clone()
        
        if valid_points_mask is not None:
            points = points[valid_points_mask]

        
        points_np = points.detach().cpu().numpy()
        indices_np = cls.indices_.cpu().numpy()

        with open(filename, "w") as f:

            for i in range(points_np.shape[0]):
                x, y, z = points_np[i]
                if sign_mask is not None and i < len(sign_mask) and sign_mask[i]:
                    r, g, b = 1.0, 0.0, 0.0
                else:
                    r, g, b = 0.0, 1.0, 0.0
                f.write(f"v {x} {y} {z} {r} {g} {b}\n")


            for i in tqdm(range(indices_np.shape[0]), desc="Saving edges"):
                for j in range(indices_np.shape[1]):
                    f.write(f"l {i+1} {indices_np[i,j]+1}\n")

    @classmethod
    def load_graph_from_file(cls, filename, device="cuda"):
        """Load OBJ connectivity and inverse-distance weights; return points, indices and weights."""
        CONSOLE.log(f"Loading graph from {filename}")	
        vertices = []
        edges = []
        

        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "l":
                    edges.append((int(parts[1])-1, int(parts[2])-1))
        
        if not vertices or not edges:
            return None, None, None
        
        points = torch.tensor(vertices, dtype=torch.float32, device=device)
        N = len(vertices)
        

        neighbor_dict = defaultdict(list)
        for src, dst in edges:
            neighbor_dict[src].append(dst)
        k = max(len(v) for v in neighbor_dict.values()) if neighbor_dict else 0
        

        indices = torch.full((N, k), -1, dtype=torch.long, device=device)
        weights = torch.zeros((N, k, 1), dtype=torch.float32, device=device)
        

        for src in neighbor_dict:
            neighbors = neighbor_dict[src]
            dist = torch.norm(points[src] - points[neighbors], dim=1)
            

            if len(neighbors) > k:
                _, idx = torch.topk(dist, k, largest=False)
                neighbors = [neighbors[i] for i in idx]
                dist = dist[idx]
            

            indices[src, :len(neighbors)] = torch.tensor(neighbors, device=device)
            

            epsilon = 1e-6
            inv_dist = 1.0 / (dist + epsilon)
            normalized_weights = (inv_dist / inv_dist.sum()).unsqueeze(-1)
            weights[src, :len(neighbors)] = normalized_weights
        

        cls.indices_ = indices
        cls.graph_weights_ = weights
        cls.graph_weights_.requires_grad_(False)
        
        return points, indices, weights


    @classmethod
    def graph_init_joint(cls, GaussianModel, frame_idx=None, k = 8, filename=None):
        points_np = GaussianModel.get_xyz.clone().detach().cpu().numpy()
        frame_idx = frame_idx if frame_idx is not None else S2NRaySamples.stFrame_
        frame_motion = (frame_idx - S2NRaySamples.stFrame_) // S2NRaySamples.step_
        joint_xyz = S2NRaySamples.currNodes_[frame_motion].clone().detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(joint_xyz)
        cls.dist_joint, cls.indices_joint = nbrs.kneighbors(points_np)
        cls.dist_joint = torch.tensor(cls.dist_joint, dtype=torch.float32, requires_grad=False).cuda()
        cls.graph_weights_joint = torch.exp(-50 * cls.dist_joint)
        cls.graph_weights_joint = cls.graph_weights_joint / cls.graph_weights_joint.sum(dim=1, keepdim=True)
        cls.graph_weights_joint.requires_grad = False

        cls.indices_joint = torch.tensor(cls.indices_joint, device=cls.dist_joint.device, dtype=torch.long)
        cls.indices_joint.requires_grad = False
        points = np.concatenate([points_np, joint_xyz], axis = 0)
        cls.edges = cls.indices_joint + points_np.shape[0]
        if filename:
            with open(filename, 'w') as f:

                for i in range(points.shape[0]):
                    if(i<points_np.shape[0]):
                        f.write(f"v {points[i, 0]} {points[i, 1]} {points[i, 2]} 0 255 0\n")
                    else:
                        f.write(f"v {points[i, 0]} {points[i, 1]} {points[i, 2]} 255 0 0\n")
                

                for i in range(cls.edges.shape[0]):
                    for j in range(cls.edges.shape[1]):
                        # OBJ format uses 1-indexed vertices
                        f.write(f"l {i+1} {cls.edges[i, j]+1}\n")
            print(filename)

    @classmethod
    def save_graph(cls, GaussianModel, frame_idx=0, filename=None):
        points_np = GaussianModel.get_xyz.clone().detach().cpu().numpy()
        frame_motion = (frame_idx - S2NRaySamples.stFrame_) // S2NRaySamples.step_
        joint_xyz = S2NRaySamples.currNodes_[frame_motion].clone().detach().cpu().numpy()
        points = np.concatenate([points_np, joint_xyz], axis = 0)

        with open(filename, 'w') as f:

            for i in range(points.shape[0]):
                if(i<points_np.shape[0]):
                    f.write(f"v {points[i, 0]} {points[i, 1]} {points[i, 2]} 0 255 0\n")
                else:
                    f.write(f"v {points[i, 0]} {points[i, 1]} {points[i, 2]} 255 0 0\n")
            

            for i in range(cls.edges.shape[0]):
                for j in range(cls.edges.shape[1]):
                    # OBJ format uses 1-indexed vertices
                    f.write(f"l {i+1} {cls.edges[i, j]+1}\n")
    @classmethod
    def get_photometric_mask(cls, gaussians_canonical, cams, pipe, background, render_func):
        error_masks = []
        for cam_idx, render_view in enumerate(cams):
            rendering = render_func(render_view, gaussians_canonical, pipe, background)["render"].cpu()
            gt_image = render_view.original_image.cpu()
            error_map = error_map_L1(rendering, gt_image)
            error_map_data = torch.sum(error_map, axis=0) / error_map.shape[0]
            error_masks.append(error_map_data)
        error_masks = torch.stack(error_masks)
        return error_masks
    @classmethod
    def compute_loss(cls, gaussians_canonical, lossp,  is_start_frame, stage=2):
        loss = 0
        loss_info = {}
        

        if lossp.scaling_term:
            threshold_coefficient = lossp.scaling_threshold_coefficient
            scaling_value = dynamicGS_term.scaling_control_loss(
                gaussians_canonical,
                threshold_coefficient=threshold_coefficient,
            )
            scaling_loss = lossp.alpha_scaling * scaling_value
            loss_info["scaling"] = scaling_loss.item()
            loss += scaling_loss
            

        if is_start_frame and lossp.isotropic_term:
            isotropic_loss = lossp.alpha_isotropic * dynamicGS_term.compute_isotropic_loss(gaussians_canonical)
            loss_info["isotropic"] = isotropic_loss.item()
            loss += isotropic_loss


        if is_start_frame == False and lossp.graph_term:
            weight_scale = 1
            rigid_loss = lossp.alpha_rigid * dynamicGS_term.compute_rigid_loss(gaussians_canonical, None, weight_scale=weight_scale)
            loss_info["rigid"] = rigid_loss.item()
            loss += rigid_loss


        if is_start_frame == False and lossp.regular_term:
            regular_loss_pos, regular_loss_color = dynamicGS_term.regular_term_compute(gaussians_canonical)
            regular_loss_pos = lossp.alpha_regular * lossp.alpha_regular_position * regular_loss_pos
            regular_loss_color = lossp.alpha_regular * regular_loss_color
            loss_info["reg_pos"] = regular_loss_pos.item()
            loss_info["reg_col"] = regular_loss_color.item()
            loss += regular_loss_pos + regular_loss_color
        
        if lossp.laplacian_term and is_start_frame:
            laplacian_loss = lossp.alpha_laplacian * dynamicGS_term.compute_laplacian_loss(gaussians_canonical)
            loss_info["laplacian"] = laplacian_loss.item()
            loss += laplacian_loss


        if is_start_frame and lossp.repulsion_term:
            repulsion_loss = lossp.alpha_repulsion * dynamicGS_term.compute_repulsion_loss(gaussians_canonical, h=0.01)
            loss_info["repulsion"] = repulsion_loss.item()
            loss += repulsion_loss
            
        return loss, loss_info
    

    @classmethod
    def compute_laplacian_loss(cls, GaussianModel):
        """Penalize displacement from the mean neighbor position."""
        xyz = GaussianModel.get_xyz[:]
        neighbor_xyz = xyz[cls.indices_]

        mean_neighbor_xyz = neighbor_xyz.mean(dim=1).clone().detach()
        laplacian = xyz - mean_neighbor_xyz

        loss = (laplacian ** 2).sum(dim=1).mean()
        return loss
    
    @classmethod
    def compute_repulsion_loss(cls, GaussianModel, h=0.01):
        """Penalize neighbor distances below h."""
        xyz = GaussianModel.get_xyz[:]
        neighbor_xyz = xyz[cls.indices_]

        diff = xyz.unsqueeze(1) - neighbor_xyz
        dist2 = (diff ** 2).sum(dim=-1)


        margin = h ** 2
        repulsion = torch.clamp(margin - dist2, min=0.0)

        loss = repulsion.mean()
        return loss
    
    @classmethod
    def compute_graph_loss(cls, GaussianModel):
        neighbor_points = GaussianModel.get_xyz[cls.indices_]

        diff = GaussianModel.get_xyz.unsqueeze(1) - neighbor_points

        squared_distances = (diff ** 2).sum(dim=-1)

        dist_2 =  torch.sqrt(squared_distances + 1e-8)

        loss = torch.mean(cls.weight  *  ((cls.dist_ - dist_2) ** 2).sum(1))
        return loss
    

    @classmethod
    def compute_rigid_loss(cls, GaussianModel, rigid_indices=None, weight_scale = 6):
        if rigid_indices == None:

            rotations = GaussianModel.get_rotation[:]

            rel_rotations = quaternion_multiply(norm_quaternion(rotations), cls.pre_rotations_inv)
            rel_rotations = norm_quaternion(rel_rotations)

            rel_rots = build_rotation(rel_rotations)

            neighbor_points = GaussianModel.get_xyz[:][cls.indices_]
            curr_diff = GaussianModel.get_xyz[:].unsqueeze(1) - neighbor_points

            offset = torch.einsum('bij,bnj->bni', rel_rots, cls.prev_diff) - curr_diff

            loss = weight_scale * torch.sum(  (cls.graph_weights_ * ( offset) ** 2).sum(2).sum(1)).mean()


        return loss


    @classmethod
    def compute_rigid_loss_topo(cls, GaussianModel, iteration = 0, weight_down_freq=200, down_percentage=0.025, min_edge_num=5, k=8):
        if not hasattr(cls, 'weight') or iteration == 0:
            cls.weight = torch.ones((GaussianModel.get_xyz.shape[0], k, 1), device=GaussianModel.get_xyz.device).detach()
            cls.weight.requires_grad = False
        if not hasattr(cls, 'offset_accum') or iteration == 0:
            cls.offset_accum = torch.zeros((GaussianModel.get_xyz.shape[0], k), device=GaussianModel.get_xyz.device).detach()
            cls.offset_accum.requires_grad = False

        rotations = GaussianModel.get_rotation

        rel_rotations = quaternion_multiply(norm_quaternion(rotations), cls.pre_rotations_inv)
        rel_rotations = norm_quaternion(rel_rotations)

        rel_rots = build_rotation(rel_rotations)

        neighbor_points = GaussianModel.get_xyz[cls.indices_]
        curr_diff = GaussianModel.get_xyz.unsqueeze(1) - neighbor_points

        offset = torch.einsum('bij,bnj->bni', rel_rots, cls.prev_diff) - curr_diff
        if iteration > 2000 and iteration <= 4000:

            offset_energy = ((offset) ** 2).sum(2).detach()


            cls.offset_accum = 0.9 * cls.offset_accum.detach() + 0.1 * offset_energy
            if iteration % weight_down_freq == 0:
                flattened_offset = cls.offset_accum.flatten()

                num_elements = flattened_offset.shape[0]
                top_k_percentage = down_percentage 
                top_k_count = int(num_elements * top_k_percentage)
                top_k_values, top_k_indices = torch.topk(flattened_offset, top_k_count)

                rows = top_k_indices // cls.offset_accum.shape[1] 
                cols = top_k_indices % cls.offset_accum.shape[1] 

                down_indices = torch.stack([rows, cols], dim=1) 
                print(down_indices.shape)
                down_indices = down_indices[cls.offset_accum.squeeze()[down_indices[:, 0], down_indices[:, 1]] > 5e-5]
                print(down_indices.shape)
                
                non_zero_edges_count = (cls.weight != 0).sum(dim=1)
                few_edge_points = non_zero_edges_count.squeeze() <= min_edge_num
                down_indices_0 = down_indices[:, 0].to(torch.int64)
                few_edge_points_indices = torch.nonzero(few_edge_points).squeeze().to(torch.int64)
                mask = (down_indices_0.unsqueeze(-1) == few_edge_points_indices.unsqueeze(0))
                down_indices_mask = mask.any(dim=1)
                down_indices = down_indices[~down_indices_mask]
                
                print(down_indices.shape)

                if down_indices.shape[0] > 0:

                    rows, cols = down_indices[:, 0], down_indices[:, 1]
                    

                    edge_energies = cls.offset_accum[rows, cols]


                    sorted_rows, sort_idx = torch.sort(rows)
                    sorted_cols = cols[sort_idx]
                    sorted_energies = edge_energies[sort_idx]
                    

                    unique_rows, counts = torch.unique_consecutive(sorted_rows, return_counts=True)
                    starts = torch.cat([torch.zeros(1, device=rows.device, dtype=torch.long), counts.cumsum(0)[:-1]])
                    

                    keep_mask = torch.zeros_like(sorted_rows, dtype=torch.bool)
                    for start, count in zip(starts, counts):

                        point_energies = sorted_energies[start:start+count]
                        local_sort = torch.argsort(point_energies, descending=True)
                        

                        n_keep = min(count, min_edge_num)
                        keep_indices = local_sort[:n_keep] + start
                        keep_mask[keep_indices] = True
                    

                    filtered_rows = sorted_rows[keep_mask]
                    filtered_cols = sorted_cols[keep_mask]
                    down_indices = torch.stack([filtered_rows, filtered_cols], dim=1)

                print(down_indices.shape)
                cls.weight = cls.weight.clone()
                cls.weight[down_indices[:, 0], down_indices[:, 1]] *= 0.5

                cls.weight[cls.weight < 0.125] = 0
                zero_indices = torch.nonzero(cls.weight == 0)
                print(zero_indices.shape)


                cls.offset_accum = torch.zeros_like(cls.offset_accum).detach()

        weighted_offset = offset * cls.weight
        loss = torch.sum( ((weighted_offset) ** 2).sum(2).sum(1)).mean()
        return loss


    @classmethod
    def compute_rigid_loss_12n(cls, GaussianModel):

        rotations = GaussianModel.get_rotation

        rel_rotations = quaternion_multiply(norm_quaternion(rotations), cls.ori_rotations_inv)
        rel_rotations = norm_quaternion(rel_rotations)

        rel_rots = build_rotation(rel_rotations)

        neighbor_points = GaussianModel.get_xyz[cls.indices_]
        curr_diff = GaussianModel.get_xyz.unsqueeze(1) - neighbor_points

        offset = torch.einsum('bij,bnj->bni', rel_rots, cls.ori_diff) - curr_diff
        loss = torch.sum( ((offset) ** 2).sum(2).sum(1)).mean()
        return loss


    @classmethod
    def regular_term_compute(cls, GaussianModel):
        loss_features_dc = torch.norm(GaussianModel.get_features_dc[:GaussianModel.last_frame_points] - cls.regular_features_dc[:GaussianModel.last_frame_points], p = 2)
        loss_features_rest = torch.norm(GaussianModel.get_features_rest[:GaussianModel.last_frame_points] - cls.regular_features_rest[:GaussianModel.last_frame_points], p = 2)
        loss_scaling = torch.norm(GaussianModel.get_scaling_ori[:GaussianModel.last_frame_points] - cls.regular_scaling[:GaussianModel.last_frame_points], p = 2)
        loss_opacity = torch.norm(GaussianModel.get_opacity_ori[:GaussianModel.last_frame_points] - cls.regular_opacity[:GaussianModel.last_frame_points], p = 2)
        regular_loss_pos =  ( loss_opacity + loss_scaling) 
        regular_loss_color = (loss_features_dc + loss_features_rest ) 

        return regular_loss_pos, regular_loss_color
    
    @classmethod
    def regular_term_compute_12n(cls, GaussianModel):
        loss_features_dc = torch.norm(GaussianModel.get_features_dc - cls.ori_features_dc, p = 2)
        loss_features_rest = torch.norm(GaussianModel.get_features_rest - cls.ori_features_rest, p = 2)
        loss_scaling = torch.norm(GaussianModel.get_scaling_ori - cls.ori_scaling, p = 2)
        loss_opacity = torch.norm(GaussianModel.get_opacity_ori - cls.ori_opacity, p = 2)
        ori_loss_pos =  ( loss_opacity + loss_scaling) 
        ori_loss_color = (loss_features_dc + loss_features_rest ) 

        return ori_loss_pos, ori_loss_color


    @classmethod
    def deform_term_compute(cls, GaussianModel, dst_xyz, dst_mask):
        distance = torch.norm(GaussianModel.get_xyz - dst_xyz, p = 2, dim = 1)
        masked_distance = distance * dst_mask[:,0]
        loss = masked_distance.sum()

        return loss


    @classmethod
    def regular_term_setup_acc(cls, GaussianModel):

        if not hasattr(cls, 'position_history'):
            cls.position_history = []
        

        current_xyz = GaussianModel.get_xyz.clone().detach()
        cls.position_history.append(current_xyz)
        

        if len(cls.position_history) > 3:
            cls.position_history.pop(0)
        

        cls.regular_xyz = current_xyz
        cls.regular_features_dc = GaussianModel.get_features_dc.clone().detach()
        cls.regular_features_rest = GaussianModel.get_features_rest.clone().detach()
        cls.regular_scaling = GaussianModel.get_scaling_ori.clone().detach()
        cls.regular_rotation = GaussianModel.get_rotation.clone().detach()
        cls.regular_opacity = GaussianModel.get_opacity_ori.clone().detach()

        if len(cls.position_history) >= 2:
            cls.xyz_velocity = current_xyz - cls.position_history[-2]
        
        if len(cls.position_history) >= 3:
            prev_velocity = cls.position_history[-2] - cls.position_history[-3]
            cls.xyz_acceleration = cls.xyz_velocity - prev_velocity
        else:
            cls.xyz_acceleration = None

        cls.pre_rotations_inv = quaternion_inverse(cls.regular_rotation)
        cls.prev_neighbor_points = cls.regular_xyz[cls.indices_]
        cls.prev_diff = cls.regular_xyz.unsqueeze(1) - cls.prev_neighbor_points

    @classmethod
    def add_velocity_next(cls, GaussianModel):
        base_prediction = GaussianModel.get_xyz + cls.xyz_velocity
        
        if hasattr(cls, 'xyz_acceleration') and cls.xyz_acceleration is not None and len(cls.position_history) >= 3:
            GaussianModel._xyz = base_prediction +  0.5 * cls.xyz_acceleration
        else:
            GaussianModel._xyz = base_prediction
        
    
    @classmethod
    def compute_isotropic_loss(cls, GaussianModel):

        r = 7
        scaling_exp = torch.exp(GaussianModel.get_scaling_ori)
        epsilon = 1e-8 
        max_val, _ = torch.max(scaling_exp, dim=1)
        min_val, _ = torch.min(scaling_exp, dim=1)

        ratio = torch.max(max_val / (min_val + epsilon), torch.tensor([r]).cuda())
        ratio = torch.nan_to_num(ratio, nan=0.0)

        loss = torch.mean(ratio) - r

        return loss
        
    @classmethod
    def scaling_control_loss(cls, GaussianModel, threshold_coefficient=6.0):
        """Penalize scales exceeding threshold_coefficient times the mean axis length."""

        avg_scaling = GaussianModel.get_scaling.mean().detach()

        threshold = avg_scaling * threshold_coefficient

        excess = GaussianModel.get_scaling - threshold


        positive_excess = F.relu(excess)


        loss = positive_excess.sum()
        return loss


class node_graph:
    regular_xyz = None
    regular_xyz_joint = None


    def regular_term_setup(self, GaussianModel, velocity_option = True):
        if self.regular_xyz == None:
            self.regular_xyz = GaussianModel.get_xyz.clone().detach()
        self.xyz_velocity = torch.zeros_like(GaussianModel.get_xyz)    

        if velocity_option:
            self.xyz_velocity = GaussianModel.get_xyz.clone().detach() - self.regular_xyz


        self.regular_xyz = GaussianModel.get_xyz.clone().detach()
        self.regular_features_dc = GaussianModel.get_features_dc.clone().detach()
        self.regular_features_rest = GaussianModel.get_features_rest.clone().detach()
        self.regular_scaling = GaussianModel.get_scaling_ori.clone().detach()
        self.regular_rotation = GaussianModel.get_rotation.clone().detach()
        self.regular_opacity = GaussianModel.get_opacity_ori.clone().detach()

        self.pre_rotations_inv = quaternion_inverse(self.regular_rotation)
        self.prev_neighbor_points = self.regular_xyz[self.indices_]
        self.prev_diff = self.regular_xyz.unsqueeze(1) - self.prev_neighbor_points

    def regular_term_prune(self, mask):

        valid_points_mask = ~mask
        self.regular_xyz = self.regular_xyz[valid_points_mask]
        self.regular_features_dc = self.regular_features_dc[valid_points_mask]
        self.regular_features_rest = self.regular_features_rest[valid_points_mask]
        self.regular_scaling = self.regular_scaling[valid_points_mask]
        self.regular_rotation = self.regular_rotation[valid_points_mask]
        self.regular_opacity = self.regular_opacity[valid_points_mask]

        self.pre_rotations_inv = quaternion_inverse(self.regular_rotation)
        self.prev_neighbor_points = self.regular_xyz[self.indices_]
        self.prev_diff = self.regular_xyz.unsqueeze(1) - self.prev_neighbor_points

    def regular_term_extend(self, gaussians):
        """Extend reference attributes and neighbor offsets for new Gaussians."""
        current_count = self.regular_xyz.shape[0]
        new_count = gaussians.get_xyz.shape[0]
        added_count = new_count - current_count
        
        if added_count <= 0:
            return


        device = self.regular_xyz.device


        new_xyz = gaussians.get_xyz[current_count:].clone().detach()
        self.regular_xyz = torch.cat([self.regular_xyz, new_xyz], dim=0)


        new_features_dc = gaussians.get_features_dc[current_count:].clone().detach()
        self.regular_features_dc = torch.cat([self.regular_features_dc, new_features_dc], dim=0)

        new_features_rest = gaussians.get_features_rest[current_count:].clone().detach()
        self.regular_features_rest = torch.cat([self.regular_features_rest, new_features_rest], dim=0)


        new_scaling = gaussians.get_scaling_ori[current_count:].clone().detach()
        self.regular_scaling = torch.cat([self.regular_scaling, new_scaling], dim=0)


        new_rotation = gaussians.get_rotation[current_count:].clone().detach()
        self.regular_rotation = torch.cat([self.regular_rotation, new_rotation], dim=0)


        new_opacity = gaussians.get_opacity_ori[current_count:].clone().detach()
        self.regular_opacity = torch.cat([self.regular_opacity, new_opacity], dim=0)


        self.pre_rotations_inv = quaternion_inverse(self.regular_rotation)
        

        self.prev_neighbor_points = self.regular_xyz[self.indices_]
        self.prev_diff = self.regular_xyz.unsqueeze(1) - self.prev_neighbor_points


    def graph_init(self, xyz, filename = None, k=8, load_graph_path=None):


        points_np = xyz.detach().cpu().numpy()

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points_np)
        self.dist_, self.indices_ = nbrs.kneighbors(points_np)

        self.dist_ = torch.tensor(self.dist_[:,1:], dtype=torch.float32, requires_grad=False).cuda()
        self.indices_ = torch.tensor(self.indices_[:,1:], device=self.dist_.device, dtype=torch.long)
        epsilon = 1e-6
        self.graph_weights_ = 1.0 / (self.dist_  + epsilon).detach()
        self.graph_weights_ = self.graph_weights_ / self.graph_weights_.sum(dim=1, keepdim=True)

        self.graph_weights_ = self.graph_weights_.unsqueeze(-1)


        self.indices_ = self.indices_.to(dtype=torch.long).detach()


        if load_graph_path:
            indices_path = os.path.join(load_graph_path, 'indices.npy')
            if os.path.exists(indices_path):
                CONSOLE.log(f"Loading indices from {indices_path}")
                self.indices_ = torch.tensor(np.load(indices_path), device=self.dist_.device, dtype=torch.long)
            else:
                CONSOLE.log(f"Saving indices to {indices_path}")
                np.save(indices_path, self.indices_.cpu().numpy())
    
    def extend_knn_graph(self, selected_pts_mask, gaussians, N = 1):
        """Let new Gaussians inherit their source points' neighbors."""
        num_existing_points = selected_pts_mask.shape[0]

        selected_indices = torch.where(selected_pts_mask)[0]
        selected_indices = selected_indices.repeat(N)

        num_new_points = selected_indices.shape[0]

        if num_new_points == 0:
            print("No new points selected, skipping k-NN graph extension.")
            return


        new_indices = self.indices_[selected_indices].clone()
        new_weights = self.graph_weights_[selected_indices].clone() / 4


        self.indices_ = torch.cat([self.indices_, new_indices], dim=0)
        self.graph_weights_ = torch.cat([self.graph_weights_, new_weights], dim=0)
        self.graph_weights_.requires_grad = False
        self.regular_term_extend(gaussians)
        print(f"Extended k-NN graph: Added {num_new_points} new points.")


    def graph_update(self, gaussians, prune_mask, current_ring, rings=2, k=8):
        """Update affected neighbors after pruning and return the compact update mask."""
        points = gaussians.get_xyz
        device = points.device
        N = points.shape[0]


        for _ in range(rings):
            current_ring_indices = torch.where(current_ring)[0]

            current_ring[self.indices_[current_ring_indices].flatten().unique()] = True
        
        valid_mask = ~prune_mask


        update_idx = torch.where(valid_mask)[0]


        full_dist = torch.cdist(points[update_idx], points[valid_mask])

        dist, nn_idx = torch.topk(full_dist, k=k+1, dim=1, largest=False)


        nn_idx = nn_idx[:, 1:]

        precise_dist = dist[:, 1:]

        valid_idx = torch.where(valid_mask)[0]
        nn_idx_global = valid_idx[nn_idx]  

        inv_dist = 1.0 / (precise_dist + 1e-6)
        weights = (inv_dist / inv_dist.sum(dim=1, keepdim=True)).unsqueeze(-1)

        self.indices_[update_idx] = nn_idx_global
        self.graph_weights_[update_idx] = weights
        self.graph_weights_.requires_grad_(False)


        self.indices_ = self.indices_[valid_mask]
        self.graph_weights_ = self.graph_weights_[valid_mask]

        valid_idx = torch.where(valid_mask)[0]
        global_to_compact = -torch.ones(N, dtype=torch.long, device=device)

        global_to_compact[valid_idx] = torch.arange(valid_idx.shape[0], device=device)

        self.indices_ = global_to_compact[self.indices_]  # Map the global indices to compact space


        return current_ring[valid_mask]


    def save_graph_to_file(self, gaussians, filename, sign_mask=None, valid_points_mask=None):
        """Save an OBJ graph, marking sign_mask points in red."""
        mkdir_p(os.path.dirname(filename))

        points = gaussians.get_xyz.clone()
        
        if valid_points_mask is not None:
            points = points[valid_points_mask]

        
        points_np = points.detach().cpu().numpy()
        indices_np = self.indices_.cpu().numpy()

        with open(filename, "w") as f:

            for i in range(points_np.shape[0]):
                x, y, z = points_np[i]
                if sign_mask is not None and i < len(sign_mask) and sign_mask[i]:
                    r, g, b = 1.0, 0.0, 0.0
                else:
                    r, g, b = 0.0, 1.0, 0.0
                f.write(f"v {x} {y} {z} {r} {g} {b}\n")


            for i in tqdm(range(indices_np.shape[0]), desc="Saving edges"):
                for j in range(indices_np.shape[1]):
                    f.write(f"l {i+1} {indices_np[i,j]+1}\n")

    def load_graph_from_file(self, filename, device="cuda"):
        """Load OBJ connectivity and inverse-distance weights; return points, indices and weights."""
        CONSOLE.log(f"Loading graph from {filename}")	
        vertices = []
        edges = []
        

        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "l":
                    edges.append((int(parts[1])-1, int(parts[2])-1))
        
        if not vertices or not edges:
            return None, None, None
        
        points = torch.tensor(vertices, dtype=torch.float32, device=device)
        N = len(vertices)
        

        neighbor_dict = defaultdict(list)
        for src, dst in edges:
            neighbor_dict[src].append(dst)
        k = max(len(v) for v in neighbor_dict.values()) if neighbor_dict else 0
        

        indices = torch.full((N, k), -1, dtype=torch.long, device=device)
        weights = torch.zeros((N, k, 1), dtype=torch.float32, device=device)
        

        for src in neighbor_dict:
            neighbors = neighbor_dict[src]
            dist = torch.norm(points[src] - points[neighbors], dim=1)
            

            if len(neighbors) > k:
                _, idx = torch.topk(dist, k, largest=False)
                neighbors = [neighbors[i] for i in idx]
                dist = dist[idx]
            

            indices[src, :len(neighbors)] = torch.tensor(neighbors, device=device)
            

            epsilon = 1e-6
            inv_dist = 1.0 / (dist + epsilon)
            normalized_weights = (inv_dist / inv_dist.sum()).unsqueeze(-1)
            weights[src, :len(neighbors)] = normalized_weights
        

        self.indices_ = indices
        self.graph_weights_ = weights
        self.graph_weights_.requires_grad_(False)
        
        return points, indices, weights


    def update_father_graph(self, indices):
        if not hasattr(self, 'father_indices'):
            self.father_indices = indices
        else:
            self.father_indices = torch.cat([self.father_indices, indices], dim=0)
    
    def prune_father_graph(self, GaussianModel, mask):
        
        invalid_father_indices = self.father_indices[mask].flatten().unique()
        prune_mask = torch.zeros(GaussianModel._xyz.shape[0], dtype=torch.bool)
        prune_mask[invalid_father_indices] = True
        GaussianModel.prune_points_no_opt(prune_mask)
        
    
    def regular_father_term_setup(self, appearance_gs, motion_gs, k=8, warpDQB=None, adaptive=False, reference_appearance_rotation=False):
        appearance_xyz = appearance_gs.get_xyz.clone().detach()
        motion_xyz = motion_gs.get_xyz.clone().detach()
        motion_rotation = motion_gs.get_rotation.clone().detach()
        self.motion_xyz = motion_xyz

        motion_rotation_neighbor = motion_rotation.repeat_interleave(repeats=k, dim=0)
        self.father_pre_rotations_inv = quaternion_inverse(motion_rotation_neighbor)
        if reference_appearance_rotation:
            # Relative rotation belongs to the child whose offset is constrained.
            # At the propagated reference state the anchor energy must be zero.
            self.father_pre_rotations_inv = quaternion_inverse(
                appearance_gs.get_rotation.clone().detach())
        print(11111123132132131)
        print(self.father_indices.max())
        print(self.father_indices.shape)
        print(appearance_xyz.shape)
        father_neighbor_xyz = appearance_xyz[self.father_indices]
        self.father_prev_diff = motion_xyz.unsqueeze(1) - father_neighbor_xyz
        self.father_prev_diff = self.father_prev_diff.reshape(-1,1,3)
        if warpDQB is not None and adaptive:
            xyz_diff = warpDQB.dx.clone().detach()
            xyz_dist = torch.norm(xyz_diff, dim=1)
            new_points_xyz_dist = torch.zeros((motion_xyz.shape[0]-warpDQB.dx.shape[0], 1), device=xyz_diff.device)
            xyz_dist = torch.cat([xyz_dist, new_points_xyz_dist], axis=0)
            self.father_weights = torch.exp(-600 * xyz_dist ** 2) * 10
            print("adaptive weights: ", self.father_weights.min(), self.father_weights.max(), self.father_weights.mean())
        else:
            self.father_weights = torch.ones(self.motion_xyz.shape[0], device=self.motion_xyz.device)

        self.father_weights = self.father_weights.repeat_interleave(repeats=k, dim=0).clone().detach().squeeze(-1)


    def compute_loss(self, gaussians_canonical, lossp,  is_start_frame, stage=2, frame_idx=-1):
        loss = 0
        loss_info = {}
        

        if lossp.scaling_term:
            scaling_loss = lossp.alpha_scaling * scaling_control_loss(gaussians_canonical, threshold_coefficient=lossp.scaling_threshold)
            loss_info["scaling"] = scaling_loss.item()
            loss += scaling_loss


        if  lossp.isotropic_term:
            isotropic_loss = lossp.alpha_isotropic * compute_isotropic_loss(gaussians_canonical)
            loss_info["isotropic"] = isotropic_loss.item()
            loss += isotropic_loss


        if is_start_frame == False and lossp.graph_term:
            rigid_loss = lossp.alpha_rigid * self.compute_rigid_loss(gaussians_canonical, None, weight_scale=6)
            loss_info["rigid"] = rigid_loss.item()
            loss += rigid_loss


        if is_start_frame == False and lossp.father_graph_term:
            rigid_loss = lossp.alpha_father_rigid * self.father_rigid_loss(gaussians_canonical,)
            loss_info["f_rigid"] = rigid_loss.item()
            loss += rigid_loss


        if is_start_frame == False and lossp.regular_term:
            regular_loss_pos, regular_loss_color = self.compute_regular_loss(gaussians_canonical)
            regular_loss_pos = lossp.alpha_regular * lossp.alpha_regular_position * regular_loss_pos
            regular_loss_color = lossp.alpha_regular * regular_loss_color
            loss_info["reg_pos"] = regular_loss_pos.item()
            loss_info["reg_col"] = regular_loss_color.item()
            loss += regular_loss_pos + regular_loss_color


        if is_start_frame and lossp.laplacian_term:
            laplacian_loss = lossp.alpha_laplacian * self.compute_laplacian_loss(gaussians_canonical)
            loss_info["laplacian"] = laplacian_loss.item()
            loss += laplacian_loss


        if  lossp.repulsion_term:
            repulsion_loss = lossp.alpha_repulsion * self.compute_repulsion_loss(gaussians_canonical)
            loss_info["repulsion"] = repulsion_loss.item()
            loss += repulsion_loss
            
        return loss, loss_info
    

    @torch.compile
    def compute_laplacian_loss(self, GaussianModel):
        """Penalize displacement from the mean neighbor position."""
        xyz = GaussianModel.get_xyz[:]
        neighbor_xyz = xyz[self.indices_]

        mean_neighbor_xyz = neighbor_xyz.mean(dim=1)
        laplacian = xyz - mean_neighbor_xyz

        loss = (laplacian ** 2).sum(dim=1).mean()
        return loss
    
    @torch.compile
    def compute_repulsion_loss(self, GaussianModel, h=0.012):
        """Penalize neighbor distances below h."""
        xyz = GaussianModel.get_xyz[:]
        neighbor_xyz = xyz[self.indices_]

        diff = xyz.unsqueeze(1) - neighbor_xyz
        dist2 = (diff ** 2).sum(dim=-1)


        margin = h ** 2
        repulsion = torch.clamp(margin - dist2, min=0.0)

        loss = repulsion.mean()
        return loss
    
    
    def compute_graph_loss(self, GaussianModel):
        neighbor_points = GaussianModel.get_xyz[self.indices_]

        diff = GaussianModel.get_xyz.unsqueeze(1) - neighbor_points

        squared_distances = (diff ** 2).sum(dim=-1)

        dist_2 =  torch.sqrt(squared_distances + 1e-8)

        loss = torch.mean(self.weight  *  ((self.dist_ - dist_2) ** 2).sum(1))
        return loss


    def compute_rigid_loss(self, GaussianModel, rigid_indices=None, weight_scale = 6):
        if rigid_indices == None:

            rotations = GaussianModel.get_rotation[:]

            rel_rotations = quaternion_multiply(norm_quaternion(rotations), self.pre_rotations_inv)
            rel_rotations = norm_quaternion(rel_rotations)

            rel_rots = build_rotation(rel_rotations)

            neighbor_points = GaussianModel.get_xyz[:][self.indices_]
            curr_diff = GaussianModel.get_xyz[:].unsqueeze(1) - neighbor_points

            offset = torch.einsum('bij,bnj->bni', rel_rots, self.prev_diff) - curr_diff

            loss = torch.sum(  (self.graph_weights_ * ( offset) ** 2).sum(2).sum(1)).mean()

        return loss
    
    def father_rigid_loss(self, appearance_gs,):

        rotations = appearance_gs.get_rotation[:]

        rel_rotations = quaternion_multiply(norm_quaternion(rotations), self.father_pre_rotations_inv)
        rel_rotations = norm_quaternion(rel_rotations)

        rel_rots = build_rotation(rel_rotations)

        motion_xyz = self.motion_xyz
        curr_diff = motion_xyz.unsqueeze(1) - appearance_gs.get_xyz[self.father_indices]
        curr_diff = curr_diff.reshape(-1,1,3)
        
        offset = torch.einsum('bij,bnj->bni', rel_rots, self.father_prev_diff) - curr_diff

        loss = torch.sum( self.father_weights * (( offset) ** 2).sum(2).sum(1)).mean()
        return loss


    def compute_regular_loss(self, GaussianModel):
        loss_features_dc = torch.norm(GaussianModel.get_features_dc[:GaussianModel.last_frame_points] - self.regular_features_dc, p = 2)
        loss_features_rest = torch.norm(GaussianModel.get_features_rest[:GaussianModel.last_frame_points] - self.regular_features_rest, p = 2)
        loss_scaling = torch.norm(GaussianModel.get_scaling_ori[:GaussianModel.last_frame_points] - self.regular_scaling, p = 2)
        loss_opacity = torch.norm(GaussianModel.get_opacity_ori[:GaussianModel.last_frame_points] - self.regular_opacity, p = 2)
        regular_loss_pos =  ( loss_opacity + loss_scaling) 
        regular_loss_color = (loss_features_dc + loss_features_rest ) 

        return regular_loss_pos, regular_loss_color
    
    
    def regular_term_compute_12n(self, GaussianModel):
        loss_features_dc = torch.norm(GaussianModel.get_features_dc - self.ori_features_dc, p = 2)
        loss_features_rest = torch.norm(GaussianModel.get_features_rest - self.ori_features_rest, p = 2)
        loss_scaling = torch.norm(GaussianModel.get_scaling_ori - self.ori_scaling, p = 2)
        loss_opacity = torch.norm(GaussianModel.get_opacity_ori - self.ori_opacity, p = 2)
        ori_loss_pos =  ( loss_opacity + loss_scaling) 
        ori_loss_color = (loss_features_dc + loss_features_rest ) 

        return ori_loss_pos, ori_loss_color


    def add_velocity_next(self, GaussianModel):
        base_prediction = GaussianModel.get_xyz + self.xyz_velocity
        
        if hasattr(self, 'xyz_acceleration') and self.xyz_acceleration is not None and len(self.position_history) >= 3:
            GaussianModel._xyz = base_prediction +  0.5 * self.xyz_acceleration
        else:
            GaussianModel._xyz = base_prediction


@torch.compile
def compute_isotropic_loss( GaussianModel):

    r = 7
    scaling_exp = torch.exp(GaussianModel.get_scaling_ori)
    epsilon = 1e-8 
    max_val, _ = torch.max(scaling_exp, dim=1)
    min_val, _ = torch.min(scaling_exp, dim=1)

    ratio = torch.max(max_val / (min_val + epsilon), torch.tensor([r]).cuda())
    ratio = torch.nan_to_num(ratio, nan=0.0)

    loss = torch.mean(ratio) - r

    return loss


def scaling_control_loss(GaussianModel, threshold_coefficient=2, lower_threshold_coefficient=0.2):
    """Penalize scales outside the configured size bounds."""

    avg_scaling = GaussianModel.get_scaling.mean().detach()
    upper_threshold = max(avg_scaling * threshold_coefficient, 0.022)
    lower_threshold = min(avg_scaling * lower_threshold_coefficient, 0.0009)
    

    scaling = GaussianModel.get_scaling.max(axis = 1)[0]
    

    excess = scaling - upper_threshold
    deficit = lower_threshold - scaling
    

    positive_excess = F.relu(excess)

    positive_deficit = F.relu(deficit)


    loss = positive_excess.sum() + positive_deficit.sum()
    
    return loss
