import torch
from torch import Tensor

__all__ = ['distance_point_to_line_2d', 'quanernion_to_matrix']


def distance_point_to_line_2d(point: Tensor, line_a: Tensor, line_b: Tensor):
    """点point到直线(line_a, line_b)的距离, shape[..., 2] """
    # 直线方程: Ax + By + C = 0
    A = line_b[..., 1] - line_a[..., 1]
    B = line_a[..., 0] - line_b[..., 0]
    C = -line_a[..., 0] * A - line_a[..., 1] * B
    # print('input:', point, line_a, line_b)
    # print('line:', A, B, C, torch.rsqrt(A.square() + B.square()))
    # print('check:', A * line_a[..., 0] + B * line_a[..., 1] + C, A * line_b[..., 0] + B * line_b[..., 1] + C)
    # 距离公式： d = |Ax+By+C|/sqrt(A^2 + B^2)
    d = torch.abs(A * point[..., 0] + B * point[..., 1] + C) * torch.rsqrt(A.square() + B.square())
    return d


def project_3d_points_to_2d(points_3d, R, T):
    pass


def quanernion_to_matrix(w: float, x: float, y: float, z: float, device=None) -> Tensor:
    """四元数转旋转矩阵"""
    return torch.tensor([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
    ], dtype=torch.float32, device=device)


def test():
    print()
    d = distance_point_to_line_2d(
        torch.tensor([[0, 0], [1, 1], [1, 1], [44.8920, -10.2155]]),  # point
        torch.tensor([[0, 1], [0, 0], [0, 0], [44.8920, -10.2155]]),  # line A
        torch.tensor([[1, 0], [1, 0], [0, 1], [49.0228, -26.5623]]),  # line B
    )
    print(d)
