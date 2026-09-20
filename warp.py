import numpy as np
import torch
import copy
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors

from rich.console import Console
from utils.calc_utils import quaternion_multiply, quaternion_inverse, build_rotation
import os
CONSOLE = Console(width=120)

def norm_quaternion(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    return q


def batch_qvec2rotmat_torch(qvecs):
    q0, q1, q2, q3 = qvecs[:, 0], qvecs[:, 1], qvecs[:, 2], qvecs[:, 3]
    
    R = torch.empty(qvecs.shape[0], 3, 3, device=qvecs.device)

    R[:, 0, 0] = 1 - 2 * q2**2 - 2 * q3**2
    R[:, 0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    R[:, 0, 2] = 2 * q1 * q3 + 2 * q0 * q2

    R[:, 1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    R[:, 1, 1] = 1 - 2 * q1**2 - 2 * q3**2
    R[:, 1, 2] = 2 * q2 * q3 - 2 * q0 * q1

    R[:, 2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    R[:, 2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    R[:, 2, 2] = 1 - 2 * q1**2 - 2 * q2**2

    return R


def batch_rotmat2qvec_torch(Rs):
    epsilon = 1e-6  # 小的正数以避免开平方时的负数
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + Rs[:, 0, 0] + Rs[:, 1, 1] + Rs[:, 2, 2], min=epsilon)).unsqueeze(1)
    qx = (Rs[:, 2, 1] - Rs[:, 1, 2]).unsqueeze(1) / (4.0 * qw + epsilon)
    qy = (Rs[:, 0, 2] - Rs[:, 2, 0]).unsqueeze(1) / (4.0 * qw + epsilon)
    qz = (Rs[:, 1, 0] - Rs[:, 0, 1]).unsqueeze(1) / (4.0 * qw + epsilon)
    qvecs = torch.cat([qw, qx, qy, qz], dim=1)

    # 确保四元数的标量部分为正
    qvecs[qvecs[:, 0] < 0] *= -1

    return qvecs


def read_ply_and_export_matrix(file_path):
    # 读取PLY文件
    plydata = PlyData.read(file_path)

    # 获取属性数量和顶点数量
    num_vertices = len(plydata.elements[0])
    num_attributes = len(plydata.elements[0].properties)

    # 初始化一个空的n x k矩阵
    data_matrix = np.zeros((num_vertices, num_attributes))

    # 填充矩阵
    for i, attribute in enumerate(plydata.elements[0].properties):
        attribute_name = attribute.name
        attribute_data = np.asarray(plydata.elements[0][attribute_name])
        data_matrix[:, i] = attribute_data

    return data_matrix


class S2NRaySamples():
    @classmethod
    def compute_next_frame(cls, frame_idx):
        return frame_idx + 1

    @staticmethod
    def _detach_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        if isinstance(obj, (list, tuple)):
            return type(obj)(S2NRaySamples._detach_tensors(item) for item in obj)
        if isinstance(obj, dict):
            return {key: S2NRaySamples._detach_tensors(value) for key, value in obj.items()}
        if hasattr(obj, "__dict__"):
            detached = copy.copy(obj)
            for key, value in obj.__dict__.items():
                setattr(detached, key, S2NRaySamples._detach_tensors(value))
            return detached
        return obj

    @classmethod
    def update_points_num(cls, points_num):
        if not hasattr(cls, "previous_points_num"):
            cls.previous_points_num = []
        cls.previous_points_num.append(points_num)
        cls.previous_points_num = cls.previous_points_num[-5:]

    @classmethod
    def update_history_gaussians(cls, gaussians, motion_folder, frame_idx):
        if not hasattr(cls, "previous_gaussians"):
            cls.previous_gaussians = []
        cls.previous_gaussians.append(cls._detach_tensors(gaussians))
        if len(cls.previous_gaussians) > 5:
            cls.previous_gaussians[0].save_ply(
                os.path.join(motion_folder, "ckt_prune", f"point_cloud_{frame_idx - 5}.ply")
            )
            cls.previous_gaussians.pop(0)
    
    def loadMotion(self, motionFolder, frame_idx, aval_indices_path=None):
        
        src = os.path.join(motionFolder, 'ckt', f'point_cloud_{frame_idx}.ply')
        src_gs = torch.from_numpy( read_ply_and_export_matrix(src)).cuda().to(torch.float32)
        print("motion gs: ", src_gs.shape)
        if aval_indices_path is not None:
            aval_indices = np.load(aval_indices_path)
            src_gs_aval = src_gs[aval_indices]
        else:
            src_gs_aval = src_gs
        self.next_frame_idx = self.compute_next_frame(frame_idx)
        dst = os.path.join(motionFolder, 'ckt', f'point_cloud_{self.next_frame_idx}.ply')
        dst_gs = torch.from_numpy( read_ply_and_export_matrix(dst)).cuda().to(torch.float32)
        self.next_xyz = dst_gs[:, :3]

        self.current_number = src_gs.shape[0]
        self.next_number = dst_gs.shape[0]

        self.gaussian_number = min(src_gs.shape[0], dst_gs.shape[0])
        
        self.current_aval_number = src_gs_aval.shape[0]
        print("current_aval_number: ", self.current_aval_number)
        src_gs = src_gs_aval[:self.current_aval_number]
        dst_gs = dst_gs[:self.current_aval_number]

        self.joint = src_gs[:, :3]
        rel_rotations = quaternion_multiply(norm_quaternion(dst_gs[:, -4:]), quaternion_inverse(src_gs[:, -4:]))
        rel_rotations = norm_quaternion(rel_rotations)
        rel_rots = build_rotation(rel_rotations)

        src_xyz = src_gs[:, :3].reshape(-1, 3, 1)
        dst_xyz = dst_gs[:, :3].reshape(-1, 3, 1)
        rel_xyz = dst_xyz - torch.einsum("ijk,ikn->ijn", rel_rots, src_xyz)

        rel_trans = torch.cat([rel_rots, rel_xyz], dim=2)
        self.rel_trans = rel_trans.reshape(-1, 3, 4)
        self.dx = dst_xyz - src_xyz

    
    def skin2JointInterpolation_np(self, points, k=8):
        """Skin to joint interpolation using sklearn's NearestNeighbors for KNN"""
        
        # Convert to numpy if not already (keep on CPU)
        points_np = points.detach().cpu().numpy()  # [N, 3]
        joints_np = self.joint.detach().cpu().numpy()  # [M, 3] (assuming self.joint exists)
        print(points_np.shape)
        print(joints_np.shape)
        
        # Find the k nearest motion points for each appearance point.
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(joints_np)
        dists, indices = nbrs.kneighbors(points_np)  # [N, k]
        
        # Keep all k neighbors and move them back to the input device.
        self.dist_ = torch.from_numpy(dists[:, ]).float().to(points.device)  # [N, k]
        self.indices_ = torch.from_numpy(indices[:, ]).long().to(points.device)  # [N, k]
        
        # Compute weights (inverse distance with epsilon for stability)
        epsilon = 1e-6
        self.graph_weights_ = 1.0 / (self.dist_ + epsilon)
        self.graph_weights_ = self.graph_weights_ / self.graph_weights_.sum(dim=1, keepdim=True)
        self.graph_weights_ = self.graph_weights_.unsqueeze(-1).detach()  # [N, k, 1]

        
        return self.graph_weights_, self.indices_

    
    def skin2JointInterpolation(self, points, k = 8):
        ##################   skin to joint interpolation ####################################
        dist_matrix = torch.cdist(points, self.joint)  # Shape: [N, N]
        # Get top-k nearest neighbors (excludingself )
        self.dist_, indices = torch.topk(dist_matrix, k , largest=False, dim=1)
        self.dist_, self.indices_ = self.dist_[:, ].detach(), indices[:, ].detach()  # Exclude self and detach
        # Compute graph weights using exponential similarity
        CONSOLE.log("dist shape:", self.dist_.shape)

        epsilon = 1e-6  # 避免除零错误
        self.graph_weights_ = 1.0 / (self.dist_  + epsilon)
        self.graph_weights_ = self.graph_weights_ / self.graph_weights_.sum(dim=1, keepdim=True)

        self.graph_weights_ = self.graph_weights_.unsqueeze(-1).detach()  # Detach to remove from computation graph

        # Ensure indices are detached and converted to long type
        self.indices_ = self.indices_.to(dtype=torch.long).detach()


    @classmethod
    def setInterval(cls, stFrame=1, edFrame=2):
        cls.stFrame_ = stFrame
        cls.edFrame_ = edFrame
        cls.step_ = 1
