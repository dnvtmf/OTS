import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from extension.utils.utils import extend_list, to_list

__all__ = [
    'make_3d_grid', 'normalize', 'to_homo', 'dot', 'reflect', 'xfm', 'to_4x4', 'align_camera_poses',
    'compute_camera_algin', 'camera_poses_error'
]


def make_3d_grid(n, delta=0.5, sacle=1., offset=0., device=None, stack_last=True) -> Tensor:
    """ 生成3D网格的坐标
    
    先生成n**3的立方体网格的坐标，即 [0, 0,0], ..., [n-1, n-1, n-1],
    再将这些坐标 变换为 (points + delta) * sacle + offset

    返回形状为 [n, n, n, 3] 或 [3, n, n, n]的坐标
    """
    if isinstance(n, (np.ndarray, Tensor)):
        n = n.tolist()
    n = extend_list(to_list(n), 3)
    points = torch.stack(
        torch.meshgrid(
            torch.arange(n[0], device=device),
            torch.arange(n[1], device=device),
            torch.arange(n[2], device=device),
            indexing='ij'
        ),
        dim=-1 if stack_last else 0
    )
    points = (points.float() + delta) * sacle + offset
    return points


def normalize(x: Tensor, dim=-1, eps=1e-20):
    return F.normalize(x, dim=dim, eps=eps)
    # return x / torch.sqrt(torch.clamp(torch.sum(x * x, dim, keepdim=True), min=eps))


def to_homo(x: Tensor) -> Tensor:
    """get homogeneous coordinates"""
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def dot(x: Tensor, y: Tensor, keepdim=True, dim=-1) -> Tensor:
    """Computes the dot product of two tensors in the given dimension dim"""
    return torch.sum(x * y, dim, keepdim=keepdim)
    # return torch.linalg.vecdot(x, y, dim=dim)


def bmv(mat: Tensor, vec: Tensor) -> Tensor:
    """Performs a batch matrix-vector product of the matrix mat and the vector vec """
    return torch.sum(vec[..., None, :] * mat, -1, keepdim=False)


def reflect(l: Tensor, n: Tensor) -> Tensor:
    """给定法线n和入射方向l, 计算反射方向"""
    return 2 * dot(l, n) * n - l


def xfm(points: Tensor, *T: Tensor):
    """
    坐标变换 transform (T[n] @ ... @ T[0] @ points.T).T

    Args:
        points: shape: [..., P, M]
        T: shape [..., N, N], M == N or M + 1 == N or M - 1 == N

    Returns:
        Tensor: shape: [..., P, N]
    """
    if len(T) == 0:
        return points
    T_ = T[0]
    for i in range(1, len(T)):
        T_ = T[i] @ T_
    if points.shape[-1] + 1 == T_.shape[-1]:
        points = to_homo(points)
    elif points.shape[-1] - 1 == T_.shape[-1]:
        points = points[..., :-1] / points[..., -1:]
    else:
        assert points.shape[-1] == T_.shape[-1]
    return points @ T_.transpose(-1, -2)
    # return torch.sum(points[..., None, :] * T_[..., None, :, :], -1, keepdim=False)


def to_4x4(T: Tensor):
    """convert T from 3x4 matrices to 4x4 matrics"""
    if T.shape[-2] == 4:
        return T
    pad = torch.zeros_like(T[..., :1, :])
    pad[..., :, -1] = 1
    T = torch.cat([T, pad], dim=-2)
    return T


def compute_camera_algin(X0: Tensor, X1: Tensor):
    """
    compute camera algin parameters using procrustes analysis

    Args:
        X0: shape: [N, 3]
        X1: shape: [N, 3]

    Returns:
        t0, t1, s0, s1, R: X1to0 = (X1 - t1) / s1 @ R.t() * s0 + t0
    """
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = (X0c ** 2).sum(dim=-1).mean().sqrt()
    s1 = (X1c ** 2).sum(dim=-1).mean().sqrt()
    X0cs = X0c / s0
    X1cs = X1c / s1
    try:
        # rotation (use double for SVD, float loses precision)
        # U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
        # R = (U @ V.t()).float()  # type:Tensor
        U, S, Vh = torch.linalg.svd((X0cs.t() @ X1cs).double())
        R = (U @ Vh).float()  # type:Tensor
        if R.det() < 0:
            R[2] *= -1
    except:
        print("warning: SVD did not converge...")
        return 0, 0, 1, 1, torch.eye(3, device=X0.device, dtype=X0.dtype)
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    t0, t1 = t0[0], t1[0]  # shape: [3], [3]
    return t0, t1, s0, s1, R


def align_camera_poses(poses1: Tensor, poses2: Tensor):
    assert poses1.shape == poses2.shape and poses1.ndim == 3 and poses1.shape[1:] == (4, 4)
    # center = poses1.new_zeros(3)
    # X1 = xfm(center, poses1)
    # X2 = xfm(center, poses2)
    # X1 = X1[..., :3] / X1[..., 3:]
    # X2 = X2[..., :3] / X2[..., 3:]
    X1 = poses1[:, :3, 3]
    X2 = poses2[:, :3, 3]
    t1, t2, s1, s2, R = compute_camera_algin(X1, X2)
    R_aligned = poses1[..., :3, :3] @ R
    t_aligned = (X1 - t1) / s1 @ R * s2 + t2
    pose_aligned = poses1.clone()
    pose_aligned[..., :3, :3] = R_aligned
    pose_aligned[..., :3, 3] = t_aligned
    return pose_aligned


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def camera_poses_error(poses: Tensor, gt_poses: Tensor, aligned=False, degree=True, reduction='mean'):
    """ calculate the rotation error and translation error between poses and gt_poses

    Args:
        poses: shape: [..., 4, 4]
        gt_poses:  shape: [..., 4, 4]
        aligned: poses are aligned?. Defaults to False.
        degree: convert rotation error to degree. Defaults to True.
        reduction: 'mean', 'sum', None
    Returns:
        rotation_error, translation_error
    """
    pose_aligned = align_camera_poses(poses, gt_poses) if not aligned else poses
    R_aligned, t_aligned = pose_aligned[:, :3, :3], pose_aligned[:, :3, 3]
    R_GT, t_GT = gt_poses[:, :3, :3], gt_poses[:, :3, 3]

    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned - t_GT).norm(dim=-1)  # type: Tensor
    if degree:
        R_error.rad2deg_()
    if reduction == 'mean':
        return R_error.mean(), t_error.mean()
    elif reduction == 'sum':
        return R_error.sum(), t_error.sum()
    else:
        return R_error, t_error


def test():
    points = torch.randn(2, 1, 4, 3)
    M = torch.randn(3, 4, 4, 3)
    assert dot(points[..., None, :], M, keepdim=False).shape == (2, 3, 4, 4)
    assert bmv(M, points).shape == (2, 3, 4, 4)
    assert xfm(points, M).shape == (2, 3, 4, 4)
