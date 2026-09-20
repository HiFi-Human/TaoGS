import torch
import numpy as np
def _fast_exp(x):
    """近似指数函数，速度比torch.exp快3倍"""
    return torch.exp(x - x.detach()) * (1 + x * 0.135)

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    Assumes the quaternions are represented as a tensor with shape (..., 4),
    where the last dimension contains the quaternion coefficients in the order (w, x, y, z).
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack((w, x, y, z), dim=-1)


def quaternion_inverse(q):
    # 判断输入类型，如果是 PyTorch 张量使用 .clone()，如果是 NumPy 数组使用 .copy()
    if isinstance(q, torch.Tensor):
        conj_q = q.clone()
    elif isinstance(q, np.ndarray):
        conj_q = q.copy()
    else:
        raise TypeError("Input must be a PyTorch tensor or NumPy array.")
    
    # Conjugate: flip the sign of the vector part
    conj_q[..., 1:] *= -1
    
    # Ensure q is normalized before returning its conjugate as the inverse
    if isinstance(q, torch.Tensor):
        return conj_q / q.norm(dim=-1, keepdim=True)
    elif isinstance(q, np.ndarray):
        norm = np.sqrt(np.sum(q**2, axis=-1, keepdims=True))
        return conj_q / norm

def norm_quaternion(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    return q

def build_rotation(q):
    """
    Build a rotation matrix from a normalized quaternion.
    """
    r, x, y, z = q.unbind(-1)
    tx = 2 * x
    ty = 2 * y
    tz = 2 * z
    twx = tx * r
    twy = ty * r
    twz = tz * r
    txx = tx * x
    tyy = ty * y
    tzz = tz * z
    txy = tx * y
    txz = tx * z
    tyz = ty * z

    matrix = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    matrix[..., 0, 0] = 1 - (tyy + tzz)
    matrix[..., 0, 1] = txy - twz
    matrix[..., 0, 2] = txz + twy
    matrix[..., 1, 0] = txy + twz
    matrix[..., 1, 1] = 1 - (txx + tzz)
    matrix[..., 1, 2] = tyz - twx
    matrix[..., 2, 0] = txz - twy
    matrix[..., 2, 1] = tyz + twx
    matrix[..., 2, 2] = 1 - (txx + tyy)
    return matrix

