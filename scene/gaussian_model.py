#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from warp import batch_rotmat2qvec_torch, norm_quaternion
from utils.calc_utils import quaternion_multiply
from rich.console import Console
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
CONSOLE = Console(width=120)


def batch_slerp_K3(quats, weights):
    batch_size, K, _ = quats.shape
    if K == 1:
        return quats[:, 0]
    # Normalize quaternions (ensure all quaternions are unit quaternions)
    quats = quats / torch.norm(quats, dim=2, keepdim=True)

    # The base quaternion (first quaternion in the batch)
    base_quat = quats[:, 0]  # shape: [batch, 4]

    # The remaining quaternions to interpolate
    next_quats = quats[:, 1:]  # shape: [batch, K-1, 4]
    alphas = weights[:, 1:]  # shape: [batch, K-1]

    # Compute dot products between base_quat and each next_quat
    dots = torch.sum(base_quat[:, None, :] * next_quats, dim=-1)  # shape: [batch, K-1]

    # Handle negative dots by flipping quaternions if necessary
    flip_mask = dots < 0
    next_quats[flip_mask] = -next_quats[flip_mask]
    dots[flip_mask] = -dots[flip_mask]

    # Compute theta_0 and sin(theta_0)
    theta_0 = torch.acos(torch.clamp(dots, -1.0, 1.0))  # shape: [batch, K-1]
    sin_theta_0 = torch.sin(theta_0)

    # Handle small angles (close to zero) for linear interpolation
    small_angle = sin_theta_0 < 1e-6
    sin_theta_0[small_angle] = 1.0  # Avoid division by zero

    # Compute theta based on the alpha values
    theta = theta_0 * alphas  # shape: [batch, K-1]
    sin_theta = torch.sin(theta)
    sin_theta_m = torch.sin(theta_0 - theta)

    # Compute SLERP coefficients
    s0 = sin_theta_m / sin_theta_0  # shape: [batch, K-1]
    s1 = sin_theta / sin_theta_0  # shape: [batch, K-1]

    # For small angles, apply linear interpolation instead of SLERP
    s0[small_angle] = 1.0 - alphas[small_angle]
    s1[small_angle] = alphas[small_angle]

    # Interpolate quaternions
    weighted_quats = s0[:, :, None] * base_quat[:, None, :] + s1[:, :, None] * next_quats  # shape: [batch, K-1, 4]

    # Sum the weighted quaternions and normalize the result
    result = weighted_quats.sum(dim=1)  # shape: [batch, 4]
    result = result / torch.norm(result, dim=1, keepdim=True)

    return result


def quaternion_rotate_vector(q, v):
    """
    使用四元数批量旋转三维向量，q 和 v 都是批量输入
    :param q: 形状为 [gaussian, knn, 4] 的四元数张量
    :param v: 形状为 [gaussian, knn, 3] 的三维向量张量
    :return: 形状为 [gaussian, knn, 3] 的旋转后的三维向量
    """
    # 将向量 v 表示为四元数 (0, vx, vy, vz)
    zeros = torch.zeros(v.shape[:-1] + (1,), device=v.device)
    v_q = torch.cat([zeros, v], dim=-1)  # 形状变为 [gaussian, knn, 4]

    # 四元数的共轭 (逆)
    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)  # 形状为 [gaussian, knn, 4]

    # 计算 q * v_q * q_conj
    intermediate = quaternion_multiply(q, v_q)
    v_rotated = quaternion_multiply(intermediate, q_conj)

    # 返回旋转后的向量（取虚部部分）
    return v_rotated[..., 1:]  # 忽略第一个分量（实部）

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self._aval_indices = torch.empty(0, dtype=torch.long, device="cuda")
        self.track_aval_indices = False
        self._xyz_static = None
        self._features_dc_static = None
        self._features_rest_static = None
        self._scaling_static = None
        self._rotation_static = None
        self._opacity_static = None
        self.max_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.masks = []
        self.setup_functions()


    @property
    def get_scaling(self):
        if self._scaling_static is None:
            return self.scaling_activation(self._scaling)
        else:
            return self.scaling_activation(torch.cat((self._scaling, self._scaling_static), dim=0))
    
    @property
    def get_scaling_ori(self):
        if self._scaling_static is None:
            return self._scaling
        else:
            return torch.cat((self._scaling, self._scaling_static), dim=0)
    
    @property
    def get_rotation(self):
        if self._rotation_static is None:
            return self.rotation_activation(self._rotation)
        else:
            return self.rotation_activation(torch.cat((self._rotation, self._rotation_static), dim=0))

    @property
    def get_rotation_ori(self):
        if self._rotation_static is None:
            return self._rotation
        else:
            return torch.cat((self._rotation, self._rotation_static), dim=0)

    @property
    def get_xyz(self):
        if self._xyz_static is None:
            return self._xyz
        else:
            return torch.cat((self._xyz, self._xyz_static), dim=0)
    
    @property
    def get_features(self):
        if self._features_dc_static is None and self._features_rest_static is None:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            features_dc = torch.cat((self._features_dc, self._features_dc_static), dim=0)
            features_rest = torch.cat((self._features_rest, self._features_rest_static), dim=0)
            return torch.cat((features_dc, features_rest), dim=1)

    
    @property
    def get_features_dc(self):
        if self._features_dc_static is None:
            return self._features_dc
        else:
            return torch.cat((self._features_dc, self._features_dc_static), dim=0)
    
    @property
    def get_features_rest(self):
        if self._features_rest_static is None:
            return self._features_rest
        else:
            return torch.cat((self._features_rest, self._features_rest_static), dim=0)
    
    @property
    def get_opacity(self):
        if self._opacity_static is None:
            return self.opacity_activation(self._opacity)
        else:
            return self.opacity_activation(torch.cat((self._opacity, self._opacity_static), dim=0))
    
    @property
    def get_opacity_ori(self):
        if self._opacity_static is None:
            return self._opacity
        else:
            return torch.cat((self._opacity, self._opacity_static), dim=0)
    
    def update_static(self, frame_id):
        num_points = self.get_xyz.shape[0]
        self._features_dc_static = self._features_dc[:num_points//10*9].clone().detach().requires_grad_(False)
        self._features_dc = self._features_dc[num_points//10*9:].clone().detach().requires_grad_(True)
        self._features_rest_static = self._features_rest[:num_points//10*9].clone().detach().requires_grad_(False)
        self._features_rest = self._features_rest[num_points//10*9:].clone().detach().requires_grad_(True)
        self._scaling_static = self._scaling[:num_points//10*9].clone().detach().requires_grad_(False)
        self._scaling = self._scaling[num_points//10*9:].clone().detach().requires_grad_(True)
        self._opacity_static = self._opacity[:num_points//10*9].clone().detach().requires_grad_(False)
        self._opacity = self._opacity[num_points//10*9:].clone().detach().requires_grad_(True)
        self._rotation_static = self._rotation[:num_points//10*9].clone().detach().requires_grad_(False)
        self._rotation = self._rotation[num_points//10*9:].clone().detach().requires_grad_(True)
        self._xyz_static = self._xyz[:num_points//10*9].clone().detach().requires_grad_(False)
        self._xyz = self._xyz[num_points//10*9:].clone().detach().requires_grad_(True)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_scaling_ori)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        CONSOLE.log("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.99 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.kernel_sum = self._xyz.shape[0]
        
        self.last_frame_points = self._xyz.shape[0]
        self.last_frame_points_origin = self._xyz.shape[0]
        self._aval_indices = torch.arange(self._xyz.shape[0], device="cuda")

    def initialize_from_data(self, data, spatial_lr_scale: float):
        """Initialize an empty model from Gaussian tensors returned by EDGS.

        ``densification_postfix`` is intentionally an optimizer-aware append
        operation.  It is therefore not suitable for the first model state,
        where no optimizer or dummy Gaussian exists yet.  This method creates
        all six trainable tensors and their bookkeeping in one place.
        """
        required = (
            "new_xyz",
            "new_features_dc",
            "new_features_rest",
            "new_opacities",
            "new_scaling",
            "new_rotation",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError("EDGS initialization is missing Gaussian fields: %s" % ", ".join(missing))

        device = self._xyz.device
        tensors = {
            name: data[name].detach().to(device=device, dtype=torch.float32).contiguous()
            for name in required
        }
        xyz = tensors["new_xyz"]
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("EDGS new_xyz must have shape [N, 3], got %s" % (tuple(xyz.shape),))
        n_points = xyz.shape[0]
        if n_points == 0:
            raise ValueError("EDGS produced an empty Gaussian point cloud")

        expected_shapes = {
            "new_features_dc": (n_points, 1, 3),
            "new_scaling": (n_points, 3),
            "new_rotation": (n_points, 4),
            "new_opacities": (n_points, 1),
        }
        for name, expected in expected_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise ValueError(
                    "EDGS %s must have shape %s, got %s"
                    % (name, expected, tuple(tensors[name].shape))
                )
        rest = tensors["new_features_rest"]
        if rest.ndim != 3 or rest.shape[0] != n_points or rest.shape[2] != 3:
            raise ValueError(
                "EDGS new_features_rest must have shape [N, K, 3], got %s"
                % (tuple(rest.shape),)
            )

        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError("EDGS %s contains non-finite values" % name)

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(tensors["new_features_dc"].requires_grad_(True))
        self._features_rest = nn.Parameter(rest.requires_grad_(True))
        self._opacity = nn.Parameter(tensors["new_opacities"].requires_grad_(True))
        self._scaling = nn.Parameter(tensors["new_scaling"].requires_grad_(True))
        self._rotation = nn.Parameter(tensors["new_rotation"].requires_grad_(True))

        self._xyz_static = None
        self._features_dc_static = None
        self._features_rest_static = None
        self._scaling_static = None
        self._rotation_static = None
        self._opacity_static = None
        self.spatial_lr_scale = float(spatial_lr_scale)
        self.kernel_sum = n_points
        self.last_frame_points = n_points
        self.last_frame_points_origin = n_points
        self._aval_indices = torch.arange(n_points, dtype=torch.long, device=device)
        self.max_radii2D = torch.zeros((n_points,), device=device)
        self.xyz_gradient_accum = torch.zeros((n_points, 1), device=device)
        self.RT_gradient_accum = torch.zeros((n_points, 1), device=device)
        self.denom = torch.zeros((n_points, 1), device=device)
        self.active_sh_degree = 0
        CONSOLE.log("Initialized Gaussian model from EDGS:", n_points)

    def training_setup_motion(self, training_args, frame_idx=0):
        self.frame_idx = frame_idx
        self.track_aval_indices = True
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.RT_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        params = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)


    def training_setup_t2(
        self,
        training_args,
        frame_idx=0,
        is_start_frame=False,
        stop1000_xyz=False,
        appearance_dc_lr_scale=1.0,
    ):
        appearance_dc_lr_scale = float(appearance_dc_lr_scale)
        if appearance_dc_lr_scale <= 0.0:
            raise ValueError(
                "appearance_dc_lr_scale must be positive, got %s"
                % appearance_dc_lr_scale
            )
        self.frame_idx = frame_idx
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.RT_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        attribute_scale = 1.0 if is_start_frame else 10.0
        opacity_scaling_scale = attribute_scale
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init_t2 * self.spatial_lr_scale / attribute_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * appearance_dc_lr_scale / attribute_scale, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr / opacity_scaling_scale, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr / opacity_scaling_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr / attribute_scale, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        if is_start_frame:
            self.xyz_scheduler_args = get_expon_lr_func(
                                                        lr_init=training_args.position_lr_init_t2*self.spatial_lr_scale ,
                                                        lr_final=training_args.position_lr_final_t2*self.spatial_lr_scale ,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        else:
            if not stop1000_xyz:
                self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init_t2*self.spatial_lr_scale / attribute_scale,
                                                    lr_final=training_args.position_lr_final_t2*self.spatial_lr_scale / attribute_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
            else:
                self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init_t2*self.spatial_lr_scale / attribute_scale,
                                                        lr_final=training_args.position_lr_final_t2*self.spatial_lr_scale / attribute_scale,
                                                        lr_delay_steps=1000,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
                self.xyz_scheduler_args2 = get_expon_lr_func(lr_init=0,
                                            lr_final=0,
                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                            max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration, fix_xyz=False,stop1000_xyz=False):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz" :
                if iteration<=1000 and stop1000_xyz:
                    lr=self.xyz_scheduler_args2(iteration)
                else:
                    lr = self.xyz_scheduler_args(iteration) if not fix_xyz else 0
                # The scheduler returns a value; Adam only sees it when the
                # optimizer parameter group is updated explicitly.
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.get_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacity_ori.detach().cpu().numpy()
        scale = self.get_scaling_ori.detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_with_idx(self, path, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()[idx]
        normals = np.zeros_like(xyz)
        f_dc = self.get_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[idx]
        f_rest = self.get_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[idx]
        opacities = self.get_opacity_ori.detach().cpu().numpy()[idx]
        scale = self.get_scaling_ori.detach().cpu().numpy()[idx]
        rotation = self.get_rotation.detach().cpu().numpy()[idx]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, spatial_lr_scale : float):
        CONSOLE.log("Loading ply file : ", path)
        self.cpc_path = path
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self.spatial_lr_scale = spatial_lr_scale
        self.kernel_sum = xyz.shape[0]
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.last_frame_points = self._xyz.shape[0]
        self.last_frame_points_origin = self._xyz.shape[0]

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._aval_indices = torch.arange(self._xyz.shape[0], device="cuda")

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree


    def warping(self, warpDQB):
        RT = warpDQB.rel_trans   # Assume [node, 3, 4]
        R = RT[..., :3]  # Shape: [node, 3, 3]
        quaternion = batch_rotmat2qvec_torch(R) # Shape: [node, 4]
        quaternion_gathered = quaternion[warpDQB.indices_] # Shape: [node, 4]
        RT_gathered = RT[warpDQB.indices_]  # Shape: [gaussian, knn, 3, 4]
        # Separate the rotation and translation components
        RT_gathered = RT_gathered

        self.rotations = RT_gathered[..., :3]  # Shape: [gaussian, knn, 3, 3]
        translations = RT_gathered[..., 3]  # Shape: [gaussian, knn, 3]

        self.pos = self.get_xyz.detach() # [gaussian, 3]
        self.pos_expanded = self.pos.unsqueeze(1).expand(-1, warpDQB.indices_.size(1), -1)

        self.pos_transformed = torch.einsum('gkij,gkj->gki', self.rotations, self.pos_expanded) + translations

        self.pos_out = torch.sum(self.pos_transformed * warpDQB.graph_weights_, dim=1)  # Weighted sum over knn dimension
        self.q_out = batch_slerp_K3(quaternion_gathered, warpDQB.graph_weights_.squeeze(-1))
        self.q_out = self.q_out / torch.norm(self.q_out, dim=1, keepdim=True)
        rots = norm_quaternion(self.get_rotation_ori.detach())
        self.new_rots = quaternion_multiply(self.q_out, rots)
        
        xyz_diff = torch.norm(self.pos_out - self.pos, dim=1)
        threshold = torch.quantile(xyz_diff, 0.7)
        stable_mask = ~(xyz_diff >= threshold).to(device="cuda")

        self._xyz = self.pos_out
        self._rotation = self.new_rots


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self.track_aval_indices:
            surviving_previous = torch.nonzero(
                valid_points_mask[:self._aval_indices.shape[0]]).squeeze(-1)
            self._aval_indices = self._aval_indices[surviving_previous]
            self.last_frame_points = self._aval_indices.shape[0]
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune_points_no_opt(self, mask):
        valid_points_mask = ~mask

        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def initialize_postfix(self, new_xyz, new_features_dc, new_features_rest,
                        new_opacities, new_scaling, new_rotation):
        self._xyz = nn.Parameter(torch.cat((self._xyz, new_xyz.detach()), dim=0).requires_grad_())
        self._features_dc = nn.Parameter(torch.cat((self._features_dc, new_features_dc.detach()), dim=0).requires_grad_())
        self._features_rest = nn.Parameter(torch.cat((self._features_rest, new_features_rest.detach()), dim=0).requires_grad_())
        self._opacity = nn.Parameter(torch.cat((self._opacity, new_opacities.detach()), dim=0).requires_grad_())
        self._scaling = nn.Parameter(torch.cat((self._scaling, new_scaling.detach()), dim=0).requires_grad_())
        self._rotation = nn.Parameter(torch.cat((self._rotation, new_rotation.detach()), dim=0).requires_grad_())

        # 初始化辅助变量
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")


    def prune_least_opacity(self, n=30):
        num_to_prune = int(self._opacity.shape[0] * n / 100)

        _, indices = torch.topk(self._opacity.squeeze(), num_to_prune, largest=False)

        prune_mask = torch.zeros(self._opacity.shape[0], dtype=torch.bool, device=self._opacity.device)
        prune_mask[indices] = True

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        num_pruned = int(prune_mask.sum().item())
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        return num_pruned

    def densify_and_prune_to_target(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        target_points,
        target_tolerance=0.05,
        promote_to_aval=True,
        force_target_cap=True,
    ):
        """Densify once, then apply target-aware pruning for the first frame.

        Clone/split is deliberately completed before any opacity or screen-size
        candidate is considered.  While the cloud is below the target lower
        bound, all candidates are protected so gradient-led densification cannot
        be immediately undone by the strong first-frame prune.  Once the lower
        bound is reached, candidate pruning may remove points, but never below
        that bound.  A separate lowest-opacity cap handles clouds above the
        upper bound.

        ``promote_to_aval`` is specific to first-frame use: all points retained
        by this operation are canonical motion points for the next frame.  The
        existing topology densification path does not call this method and is
        therefore unchanged.
        """
        target_points = int(target_points)
        target_tolerance = float(target_tolerance)
        if target_points < 1:
            raise ValueError("target_points must be positive")
        if not 0.0 <= target_tolerance < 1.0:
            raise ValueError("target_tolerance must be in [0, 1)")

        before = int(self.get_xyz.shape[0])
        lower = max(1, int(np.floor(target_points * (1.0 - target_tolerance))))
        upper = max(lower, int(np.ceil(target_points * (1.0 + target_tolerance))))

        # Keep this computation aligned with the released densification path,
        # but do not call densify_and_prune: its immediate prune would erase the
        # growth that the target-aware policy must protect.
        grads = self.xyz_gradient_accum / self.denom
        grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        after_densify = int(self.get_xyz.shape[0])
        densified = max(0, after_densify - before)

        opacity_candidate_mask = (self.get_opacity < min_opacity).squeeze(-1)
        size_candidate_mask = torch.zeros_like(opacity_candidate_mask)
        if max_screen_size is not None and max_screen_size > 0:
            size_candidate_mask = torch.logical_or(
                self.max_radii2D > max_screen_size,
                self.get_scaling.max(dim=1).values > 0.1 * extent,
            )

        opacity_candidates = int(opacity_candidate_mask.sum().item())
        size_candidates = int(size_candidate_mask.sum().item())
        candidate_mask = torch.logical_or(
            opacity_candidate_mask, size_candidate_mask
        )
        candidate_count = int(candidate_mask.sum().item())

        # Under target, preserve the complete densification result.  At or above
        # target, retain at least the lower bound even when every point is a
        # prune candidate.  For an overfull cloud, use the upper bound as the
        # normal-prune floor so target capping remains an explicit final step.
        if after_densify > upper:
            normal_keep = upper
        elif after_densify >= lower:
            normal_keep = lower
        else:
            normal_keep = after_densify
        prune_budget = max(0, after_densify - normal_keep)
        protected_below_lower = after_densify < lower
        if protected_below_lower:
            prune_budget = 0

        opacity_pruned = 0
        size_pruned = 0
        normal_prune_mask = torch.zeros(
            (after_densify,), dtype=torch.bool, device=self._xyz.device
        )
        if candidate_count > 0 and prune_budget > 0:
            opacity_indices = torch.nonzero(
                opacity_candidate_mask, as_tuple=False
            ).squeeze(-1)
            size_only_indices = torch.nonzero(
                torch.logical_and(size_candidate_mask, ~opacity_candidate_mask),
                as_tuple=False,
            ).squeeze(-1)
            opacity_values = self.get_opacity.squeeze(-1)
            # Opacity candidates have priority, and each group is ordered by
            # activated opacity for deterministic strongest-prune behavior.
            if opacity_indices.numel() > 0:
                opacity_indices = opacity_indices[
                    torch.argsort(opacity_values[opacity_indices])
                ]
            if size_only_indices.numel() > 0:
                size_only_indices = size_only_indices[
                    torch.argsort(opacity_values[size_only_indices])
                ]
            ordered_candidates = torch.cat(
                (opacity_indices, size_only_indices), dim=0
            )
            selected_indices = ordered_candidates[:prune_budget]
            normal_prune_mask[selected_indices] = True
            opacity_pruned = int(
                torch.logical_and(
                    normal_prune_mask, opacity_candidate_mask
                ).sum().item()
            )
            size_pruned = int(
                torch.logical_and(
                    normal_prune_mask,
                    torch.logical_and(size_candidate_mask, ~opacity_candidate_mask),
                ).sum().item()
            )
            self.prune_points(normal_prune_mask)

        after_normal_prune = int(self.get_xyz.shape[0])

        target_pruned = 0
        if force_target_cap and after_normal_prune > upper:
            target_pruned = after_normal_prune - upper
            _, indices = torch.topk(
                self.get_opacity.squeeze(-1), target_pruned, largest=False
            )
            prune_mask = torch.zeros(
                (after_normal_prune,), dtype=torch.bool, device=self._xyz.device
            )
            prune_mask[indices] = True
            self.prune_points(prune_mask)

        after = int(self.get_xyz.shape[0])
        total_pruned = opacity_pruned + size_pruned + target_pruned

        # Densification/post-pruning changes the tensors through the optimizer,
        # so refresh the count metadata explicitly.  At the end of first-frame
        # processing every retained point is canonical for the next frame;
        # previous/new-point split.
        self.kernel_sum = after
        if hasattr(self, "RT_gradient_accum"):
            self.RT_gradient_accum = torch.zeros(
                (after, 1), dtype=self._xyz.dtype, device=self._xyz.device
            )
        if promote_to_aval and self.track_aval_indices:
            self._aval_indices = torch.arange(
                after, dtype=torch.long, device=self._xyz.device
            )
            self.last_frame_points = after

        protection = "below-lower" if protected_below_lower else "active"
        CONSOLE.log(
            "[first-frame densify] before=%d after_densify=%d after=%d "
            "densified=%d pruned=%d opacity_candidates=%d opacity_pruned=%d "
            "size_candidates=%d size_pruned=%d target_pruned=%d "
            "target=%d tolerance=%.3f range=[%d,%d] low_watermark=%s "
            "candidate_count=%d prune_budget=%d"
            % (
                before,
                after_densify,
                after,
                densified,
                total_pruned,
                opacity_candidates,
                opacity_pruned,
                size_candidates,
                size_pruned,
                target_pruned,
                target_points,
                target_tolerance,
                lower,
                upper,
                protection,
                candidate_count,
                prune_budget,
            ),
            markup=False,
        )
        return {
            "before": before,
            "after_densify": after_densify,
            "after": after,
            "densified": densified,
            "pruned": total_pruned,
            "opacity_candidates": opacity_candidates,
            "opacity_pruned": opacity_pruned,
            "size_candidates": size_candidates,
            "size_pruned": size_pruned,
            "target_pruned": target_pruned,
            "target": target_points,
            "lower": lower,
            "upper": upper,
            "low_watermark_protected": protected_below_lower,
        }

    def prune_to_target_by_opacity(self, target_points, promote_to_aval=True):
        """One-shot lowest-opacity pruning with an exact hard floor."""
        target_points = int(target_points)
        if target_points < 1:
            raise ValueError("target_points must be positive")
        before = int(self.get_xyz.shape[0])
        num_to_prune = max(0, before - target_points)
        if num_to_prune > 0:
            _, indices = torch.topk(
                self.get_opacity.squeeze(-1), num_to_prune, largest=False
            )
            prune_mask = torch.zeros(
                (before,), dtype=torch.bool, device=self._xyz.device
            )
            prune_mask[indices] = True
            self.prune_points(prune_mask)

        after = int(self.get_xyz.shape[0])
        if after < target_points:
            raise RuntimeError(
                "final opacity prune crossed hard floor: %d < %d"
                % (after, target_points)
            )
        self.kernel_sum = after
        if hasattr(self, "RT_gradient_accum"):
            self.RT_gradient_accum = torch.zeros(
                (after, 1), dtype=self._xyz.dtype, device=self._xyz.device
            )
        if promote_to_aval and self.track_aval_indices:
            self._aval_indices = torch.arange(
                after, dtype=torch.long, device=self._xyz.device
            )
            self.last_frame_points = after
        CONSOLE.log(
            "[first-frame final opacity cap] before=%d pruned=%d after=%d target=%d"
            % (before, num_to_prune, after, target_points),
            markup=False,
        )
        return {"before": before, "pruned": num_to_prune, "after": after, "target": target_points}

    def only_prune(self, min_opacity, ):


        prune_mask = (torch.max(self.get_scaling, dim=1).values < min_opacity).squeeze()
        

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def topo_densify_and_split(self, grads, grad_threshold, gaussian_graph, scene_extent, N=1):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        selected_pts_mask[:self.last_frame_points] = False

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        if selected_pts_mask.sum() > 0:
            gaussian_graph.extend_knn_graph(selected_pts_mask, self, N)


    def topo_densify_and_clone(self, grads, grad_threshold, gaussian_graph, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask[:self.last_frame_points] = False
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        if selected_pts_mask.sum() > 0:
            gaussian_graph.extend_knn_graph(selected_pts_mask, self)


    def topo_densify_and_prune(
        self,
        max_grad,
        gaussian_graph,
        min_opacity,
        extent,
        max_screen_size,
        tp=None,
        local_graph=True,
        prune_out_of_mask=True,
        preserve_previous_points=False,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if self._xyz.shape[0] == self.last_frame_points:
            return

        self.topo_densify_and_clone(grads, max_grad, gaussian_graph, extent)
        self.topo_densify_and_split(grads, max_grad, gaussian_graph, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if tp is not None and prune_out_of_mask:
            out_of_mask = tp.check_out_of_mask(self, kernel_size=15, view_num_threshold=10)
            prune_mask = torch.logical_or(prune_mask, out_of_mask)

        # Apply the previous-frame safety limit after all pruning reasons have
        # been combined.  Otherwise out-of-mask points bypass the limit and a
        # single noisy frame can delete most of the motion representation.
        previous_mask = prune_mask[:self.last_frame_points]
        proposed_previous_prune = previous_mask.sum().item()
        if preserve_previous_points:
            prune_mask = prune_mask.clone()
            prune_mask[:self.last_frame_points] = False
            CONSOLE.log(
                "Preserved previous points: rejected %d prune candidates"
                % proposed_previous_prune)
        else:
            max_previous_prune = int(self.last_frame_points * 0.01)
            if proposed_previous_prune > max_previous_prune:
                candidate_indices = torch.nonzero(previous_mask, as_tuple=False).squeeze(-1)
                candidate_opacity = self.get_opacity[:self.last_frame_points].squeeze(-1)[candidate_indices]
                _, selected_candidates = torch.topk(
                    -candidate_opacity, k=max_previous_prune)
                previous_mask = torch.zeros_like(previous_mask)
                previous_mask[candidate_indices[selected_candidates]] = True
                prune_mask = prune_mask.clone()
                prune_mask[:self.last_frame_points] = previous_mask
                CONSOLE.log(
                    "Capped previous-point pruning: proposed %d, kept %d"
                    % (proposed_previous_prune, max_previous_prune))

        if prune_mask.sum() > 0:
            CONSOLE.log("Prune Mask new: ", prune_mask.sum())


        init_ring_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        init_ring_mask[self.last_frame_points:] = True  # All new points

        gaussian_graph.graph_update(
            self, prune_mask, init_ring_mask, rings=2, k=8, local=local_graph)

        self.prune_points(prune_mask)
        gaussian_graph.regular_term_prune(prune_mask)

        torch.cuda.empty_cache()


    def appearance_init(self,  N=2):
        n_init_points = self.get_xyz.shape[0]
        selected_pts_mask = torch.ones(n_init_points, dtype=torch.bool, device="cuda")
        stds = self.get_scaling[selected_pts_mask].repeat(N,1) * 0.5
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (1*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def edge_based_densify(self, motion_gaussian, gaussian_graph, new_mask,  spatial_lr_scale : float, offset_min=1/3, offset_max=1/2, appearace_graph=None, random=False, sanitize_new_points=False):
        """
        For each Gaussian, create new Gaussians along edges to neighbors.
        New Gaussians are placed between [offset_min, offset_max] along each edge.
        After that, prune the original Gaussians.
        """
        new_mask = new_mask.bool()
        print("new mask: ", new_mask.sum())
        if new_mask.sum() == 0:
            CONSOLE.log("[edge_based_init] No points selected for splitting (mask is empty).")
            return
        xyz = motion_gaussian.get_xyz[new_mask]
        scaling = motion_gaussian.get_scaling[new_mask]
        rotation = motion_gaussian._rotation[new_mask]
        features_dc = motion_gaussian._features_dc[new_mask]
        features_rest = motion_gaussian._features_rest[new_mask]
        opacity = motion_gaussian._opacity[new_mask]


        indices = gaussian_graph.indices_[new_mask]  # shape (M, K)
        M, K = indices.shape

        src_xyz = xyz.unsqueeze(1).expand(-1, K, -1)               # (M, K, 3)
        neighbor_xyz = motion_gaussian.get_xyz[indices]           # (M, K, 3)
        edge_vectors = neighbor_xyz - src_xyz  

        offsets = torch.rand_like(edge_vectors[..., 0]) * (offset_max - offset_min) + offset_min  # (M, K)
        offsets = offsets.unsqueeze(-1)     

        new_xyz = src_xyz + offsets * edge_vectors                # (M, K, 3)
        new_xyz = new_xyz.reshape(-1, 3)                          # (M*K, 3)
        if random:
            stds = scaling.repeat(K,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(rotation).repeat(K,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(K, 1)

        new_scaling = torch.clamp(
            torch.abs(offsets * edge_vectors * 0.9),  # Compute element-wise absolute values
            min=0.001,  # Enforce lower bound
        ).reshape(-1, 3).cuda()

        
        CONSOLE.log(new_scaling.max(axis = 1)[0].min())

        new_rotation = rotation.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4)
        if sanitize_new_points:
            # Interpolate attributes along the same edge used for placement.
            # This avoids copying an isolated white/high-opacity parent into
            # all K children.  Opacity is divided across children so their
            # aggregate contribution remains bounded.
            edge_weights = offsets
            neighbor_features_dc = motion_gaussian._features_dc[indices]
            neighbor_features_rest = motion_gaussian._features_rest[indices]
            neighbor_opacity = motion_gaussian._opacity[indices]
            new_features_dc = (
                features_dc.unsqueeze(1) * (1.0 - edge_weights.unsqueeze(-1))
                + neighbor_features_dc * edge_weights.unsqueeze(-1)
            ).reshape(-1, *features_dc.shape[1:])
            new_features_rest = (
                features_rest.unsqueeze(1) * (1.0 - edge_weights.unsqueeze(-1))
                + neighbor_features_rest * edge_weights.unsqueeze(-1)
            ).reshape(-1, *features_rest.shape[1:])
            opacity_prob = torch.sigmoid(opacity).unsqueeze(1)
            neighbor_opacity_prob = torch.sigmoid(neighbor_opacity)
            blended_opacity = (
                opacity_prob * (1.0 - edge_weights)
                + neighbor_opacity_prob * edge_weights
            )
            child_opacity = torch.clamp(blended_opacity / float(K), 1e-4, 0.5)
            new_opacity = inverse_sigmoid(child_opacity).reshape(
                -1, *opacity.shape[1:]
            )
            # f_dc is converted to RGB as 0.5 + SH_C0 * f_dc.  Keep newly
            # created points just inside the visible range so a child cannot
            # start as an exactly saturated white sample.
            dc_limit = (0.95 - 0.5) / 0.28209479177387814
            new_features_dc = torch.nan_to_num(
                new_features_dc, nan=0.0, posinf=dc_limit, neginf=-dc_limit
            ).clamp(-dc_limit, dc_limit)
            new_features_rest = torch.nan_to_num(
                new_features_rest, nan=0.0, posinf=1.0, neginf=-1.0
            )
            CONSOLE.log(
                "Sanitized edge-init attributes: children=%d opacity_divisor=%d"
                % (new_xyz.shape[0], K)
            )
        else:
            new_features_dc = features_dc.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, *features_dc.shape[1:])
            new_features_rest = features_rest.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, *features_rest.shape[1:])
            new_opacity = opacity.unsqueeze(1).expand(-1, K, -1).reshape(-1, *opacity.shape[1:])
        self.spatial_lr_scale = spatial_lr_scale

        new_indices = torch.arange(M * K, device=new_xyz.device).reshape(M, K) + self.get_xyz.shape[0] # (M, K)

        # Insert new gaussians into the model
        self.initialize_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            self.scaling_inverse_activation(new_scaling),
            new_rotation
        )
        appearace_graph.update_father_graph(new_indices)

        # Prune original points: keep only the new points
        CONSOLE.log(f'Edge-based init done: {new_xyz.shape[0]} new points created.')

    def random_densify(self, motion_gaussian, gaussian_graph, new_mask,  spatial_lr_scale : float, offset_min=1/3, offset_max=1/2, appearace_graph=None):
        """
        For each Gaussian, create new Gaussians along edges to neighbors.
        New Gaussians are placed between [offset_min, offset_max] along each edge.
        After that, prune the original Gaussians.
        """
        new_mask = new_mask.bool()
        print("new mask: ", new_mask.sum())
        if new_mask.sum() == 0:
            CONSOLE.log("[edge_based_init] No points selected for splitting (mask is empty).")
            return
        xyz = motion_gaussian.get_xyz[new_mask]
        scaling = motion_gaussian.get_scaling[new_mask]
        rotation = motion_gaussian._rotation[new_mask]
        features_dc = motion_gaussian._features_dc[new_mask]
        features_rest = motion_gaussian._features_rest[new_mask]
        opacity = motion_gaussian._opacity[new_mask]
        
        indices = gaussian_graph.indices_[new_mask]  # shape (M, K)
        M, K = indices.shape
        N = K

        stds = scaling.repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(rotation).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(N, 1)
        new_scaling = scaling.repeat(N,1) / (0.8*N)
        new_rotation = rotation.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4)
        new_features_dc = features_dc.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, *features_dc.shape[1:])
        new_features_rest = features_rest.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, *features_rest.shape[1:])
        new_opacity = opacity.unsqueeze(1).expand(-1, K, -1).reshape(-1, *opacity.shape[1:])

        # Insert new gaussians into the model
        self.initialize_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            self.scaling_inverse_activation(new_scaling),
            new_rotation
        )
        new_indices = torch.arange(M * K, device=new_xyz.device).reshape(M, K) + self.get_xyz.shape[0] # (M, K)
        appearace_graph.update_father_graph(new_indices)

        # Prune original points: keep only the new points
        CONSOLE.log(f'Edge-based init done2: {new_xyz.shape[0]} new points created.')

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_RT(self, viewspace_point_tensor, update_filter):
        self.RT_gradient_accum[update_filter] += torch.norm(self._xyz.grad[update_filter, :], dim=-1, keepdim=True)
        self.RT_gradient_accum[update_filter] += 10 * torch.norm(self._rotation.grad[update_filter, :], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_residual(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        torch.cuda.empty_cache()

    def lock_gradient(self, lock_xyz=False, lock_features=False, lock_opacity=False,
                      lock_scaling=False, lock_rotation=False, grad_scale=0.001):
        """Attenuate selected previous-frame gradients; new points are unchanged.

        Zero disables their gradients. This is gradient scaling, not an Adam
        learning-rate multiplier (nor a guarantee of zero momentum updates).
        """
        for parameter, restricted in (
            (self._xyz, lock_xyz),
            (self._features_dc, lock_features),
            (self._features_rest, lock_features),
            (self._opacity, lock_opacity),
            (self._scaling, lock_scaling),
            (self._rotation, lock_rotation),
        ):
            if restricted and parameter.grad is not None:
                parameter.grad[:self.last_frame_points] *= grad_scale
