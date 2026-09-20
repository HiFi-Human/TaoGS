import torch
import numpy as np
import os
from utils.graph_utils import dynamicGS_term
from rich.console import Console
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.join(repo_root, "third_party", "RoMa"))
import cv2

from natsort import natsorted
from utils.corr_init import init_gaussians_with_corr
from functools import partial
from utils.sequence_utils import read_sequence_transforms
import open3d as o3d
from romatch import roma_outdoor, roma_indoor
CONSOLE = Console(width=120)


def paired_candidate_mask(errors, visibility, source, neighbor,
                          rows_a, cols_a, rows_b, cols_b, threshold=0.46):
    """Require newly visible pixels in both views and summed mean RGB L1 >= threshold.

    Pixel coordinates use the error-map resolution. Visibility maps follow the
    same view order, may have a different resolution, and use zero for newly visible.
    """
    def sample(view, rows, cols):
        error = errors[view]
        flow = visibility[view]
        r = np.clip(np.floor(rows * (flow.shape[0] / error.shape[0])).astype(int),
                    0, flow.shape[0] - 1)
        c = np.clip(np.floor(cols * (flow.shape[1] / error.shape[1])).astype(int),
                    0, flow.shape[1] - 1)
        return error[rows, cols], flow[r, c] == 0

    error_a, new_a = sample(source, rows_a, cols_a)
    error_b, new_b = sample(neighbor, rows_b, cols_b)
    return new_a & new_b & ((error_a + error_b) >= threshold)


class TopoHandler:
    def __init__(self, npz_file_path, json_path, cfg):
        self.flow_mask =  np.load(npz_file_path)
        self.flow_mask = self.flow_mask['arr_0']
        self.frame_nums, self.flow_n_views, self.H, self.W = self.flow_mask.shape

        self.frames = read_sequence_transforms(json_path)['frames']
        self.frames = natsorted(self.frames, key=lambda x: int(os.path.splitext(os.path.basename(x['file_path']))[0]))

        # Flow columns use numeric camera IDs, including gaps in calibration.
        self.flow_view_indices = [
            int(os.path.splitext(os.path.basename(frame['file_path']))[0])
            for frame in self.frames
        ]

        self.n_views = len(self.frames)

        self.cfg = cfg
        if cfg.roma_model == "indoors":
            self.roma_model = roma_indoor(device='cuda')
        else:
            self.roma_model = roma_outdoor(device='cuda')
        self.roma_model.upsample_preds = False
        self.roma_model.symmetric = False

    def save_point_cloud_with_colors(self, xyz, mask, save_path):
        """Save selected points in red and remaining points in green."""
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        colors = np.zeros_like(xyz)
        colors[mask] = [1, 0, 0]
        colors[~mask] = [0, 1, 0]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(save_path, point_cloud)
        print(f"Point cloud saved at: {save_path}")


    def check_out_of_mask(self, gaussian, view_num_threshold=8, kernel_size=40):
        """Select points outside dilated foreground masks in enough views."""
        xyz = gaussian.get_xyz
        N_points = xyz.shape[0]
        counters = torch.zeros(N_points, dtype=torch.int32, device=xyz.device)
        
        # Convert OpenGL camera axes to the projection convention.
        flip_mat = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=xyz.device)

        for view_idx in range(self.n_views):
            frame = self.frames[view_idx]
            mask_view = gaussian.masks[view_idx]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask_view = cv2.dilate(mask_view, kernel, iterations=1)

            W, H = mask_view.shape[1], mask_view.shape[0]
            
            if not isinstance(mask_view, torch.Tensor):
                mask_view = torch.tensor(mask_view, device=xyz.device)
            else:
                mask_view = mask_view.to(xyz.device)
            

            transform_matrix = torch.tensor(frame['transform_matrix'], 
                                        dtype=torch.float32, 
                                        device=xyz.device)
            RT = torch.linalg.inv(transform_matrix @ flip_mat)[:3, :]

            if 'fl_x' in frame:
                K = torch.tensor([
                    [frame['fl_x'], 0, frame['cx']],
                    [0, frame['fl_y'], frame['cy']],
                    [0, 0, 1]
                ], dtype=torch.float32, device=xyz.device)
            else:
                K = torch.tensor(frame['K'], dtype=torch.float32, device=xyz.device)
                
            K_scaled = K.clone()
            K_W = K[0, 2] * 2
            scale = K_W / W
            K_scaled[:2, :] /= scale

            xyz_homo = torch.hstack([xyz, torch.ones((N_points, 1), device=xyz.device)])
            camera_coords = (RT @ xyz_homo.T).T

            valid_depth = camera_coords[:, 2] > 0
            camera_coords = camera_coords[valid_depth]
            valid_depth_indices = torch.where(valid_depth)[0]

            pixel_coords_homo = K_scaled @ camera_coords.T
            pixel_coords = pixel_coords_homo[:2] / pixel_coords_homo[2]
            u = torch.round(pixel_coords[0]).long()
            v = torch.round(pixel_coords[1]).long()

            valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            valid_indices = valid_depth_indices[valid_uv]
            invalid_indices = valid_depth_indices[~valid_uv]

            if valid_indices.shape[0] > 0:
                valid_v = v[valid_uv]
                valid_u = u[valid_uv]
                
                mask_values = mask_view[valid_v, valid_u]

                hit_indices = valid_indices[mask_values == 0]

                counters[hit_indices] += 1
                    
        selected_indices = torch.where(counters >= view_num_threshold)[0]
        selected_indices = selected_indices.long()

        selected_mask = torch.zeros(N_points, dtype=torch.bool, device=xyz.device)
        selected_mask[selected_indices] = True

        return selected_mask


    def topo_add(self, gaussians, viewpoint_stack, frame_idx, verbose=False, error_masks=None):
        """Add triangulated candidates passing paired visibility and RGB filtering."""
        matting_masks = np.asarray(gaussians.masks) != 0
        mask = np.logical_not(matting_masks).astype(np.uint8)
        matching_views = viewpoint_stack
        camera_ids = [int(os.path.splitext(os.path.basename(cam.image_name))[0])
                      for cam in matching_views]
        visibility = self.flow_mask[frame_idx - 1, camera_ids]
        pair_filter = partial(paired_candidate_mask,
                              error_masks.detach().cpu().numpy(), visibility)

        new_data_dict = init_gaussians_with_corr(
            gaussians, 
            matching_views,
            self.cfg, 
            gaussians._xyz.device,                                                                                    
            verbose=verbose, 
            roma_model=self.roma_model, 
            mask=mask, 
            return_new_data=True, pair_filter=pair_filter)
        
        if new_data_dict is None:
            return 0
        
        new_xyz = new_data_dict["new_xyz"]
        new_features_dc = new_data_dict["new_features_dc"]
        new_features_rest = new_data_dict["new_features_rest"]
        new_opacities = new_data_dict["new_opacities"]
        new_scaling = new_data_dict["new_scaling"]
        new_rotation = new_data_dict["new_rotation"]

        gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

        prune_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device)
        init_ring_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
        init_ring_mask[gaussians.last_frame_points:] = True  # All new points
        current_ring = dynamicGS_term.new_points_graph_update(gaussians, k = 8)
        dynamicGS_term.regular_term_extend(gaussians)

        new_points_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device)
        new_points_mask[gaussians.last_frame_points:] = True
        os.makedirs(os.path.dirname(os.path.dirname(gaussians.cpc_path) + "/../topo/%d.ply"% gaussians.frame_idx), exist_ok=True)
        self.save_point_cloud_with_colors(gaussians.get_xyz, new_points_mask, os.path.dirname(gaussians.cpc_path) + "/../topo/%d_cotracker.ply"% gaussians.frame_idx)
        
        return new_xyz.shape[0]
