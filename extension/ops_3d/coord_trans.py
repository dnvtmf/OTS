""" coordinate transformation 三维坐标变换

世界空间World space和观察空间view space为右手系(OpenGL坐标系)，+x向右手边, +y指向上方, +z指向屏幕外
裁剪空间Clip space为左手系: +1指向右手边, +y 指向上方, +z指向屏幕内; z 的坐标值越小，距离观察者越近
屏幕坐标系： X 轴向右为正，Y 轴向下为正，坐标原点位于窗口的左上角 (左手系: z轴向屏幕内，表示深度)

坐标变换矩阵: T{s}2{d} Transform from {s} space to {d} space
{s} 和 {d} 包括世界World坐标系、观察View坐标系、裁剪Clip坐标系、屏幕Screen坐标

Tv2s即相机内参，Tw2v即相机外参
"""

from typing import Tuple, Union
import math
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F

from extension.utils.torch_utils import to_tensor


def _rotate(angle: Union[float, Tensor], device=None, a=0, b=1) -> Tensor:
    if isinstance(angle, Tensor):
        s, c = torch.sin(angle), torch.cos(angle)
        T = torch.eye(4, dtype=s.dtype, device=s.device if device is None else device)
        T = T.expand(list(s.shape) + [4, 4]).contiguous()
        T[..., a, a] = c
        T[..., a, b] = -s
        T[..., b, a] = s
        T[..., b, b] = c
    else:
        s, c = math.sin(angle), math.cos(angle)
        T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        T[a][a] = c
        T[b][b] = c
        T[a][b] = -s
        T[b][a] = s
        T = torch.tensor(T, device=device)
    return T


def rotate_x(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 1, 2)


def rotate_y(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 0, 2)


def rotate_z(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 0, 1)


def rotate(x: float = None, y: float = None, z: float = None, device=None):
    R = torch.eye(4, device=device)
    if x is not None:
        R = R @ rotate_x(x, device)
    if y is not None:
        R = R @ rotate_y(y, device)
    if z is not None:
        R = R @ rotate_z(z, device)
    return R


def scale(s: float, device=None):
    return torch.tensor([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)


## 世界坐标系相关
def coord_spherical_to(radius: Tensor, thetas: Tensor, phis: Tensor) -> Tensor:
    """ 球坐标系 转 笛卡尔坐标系(OpenGL)

    Args:
        radius (Tensor): 径向半径 radial distance, 原点O到点P的距离 [0, infity]
        thetas (Tensor): 极角 polar angle, 正y轴与连线OP的夹角 [0, pi]
        phis (Tensor): 方位角 azimuth angle, 正x轴与连线OP在xz平面的投影的夹角, [0, 2 * pi]

    Returns:
        Tensor: 点P的笛卡尔坐标系, shape: [..., 3]
    """
    radius = to_tensor(radius, dtype=torch.float32)
    thetas = to_tensor(thetas, dtype=torch.float32)
    phis = to_tensor(phis, dtype=torch.float32)
    # yapf: disable
    return torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)
    # yapf: enable


def coord_to_spherical(points: Tensor):
    """ 笛卡尔坐标系(OpenGL) 转 球坐标系

    Args:
        points (Tensor): 点P的笛卡尔坐标系, shape: [..., 3]

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 点P的球坐标系: 径向半径、极角和方位角
    """
    raduis = points.norm(p=2, dim=-1)  # type: Tensor
    thetas = torch.arccos(points[..., 1] / raduis)
    phis = torch.arctan2(points[..., 0], points[..., 2])
    return raduis, thetas, phis


## 相机坐标系相关
def look_at(eye: Tensor, at: Tensor = None, up: Tensor = None, inv=False):
    if at is None:
        dir_vec = torch.zeros_like(eye)
        dir_vec[..., 2] = -1.
    else:
        dir_vec = F.normalize(eye - at, dim=-1)

    if up is None:
        up = torch.zeros_like(dir_vec)
        # if dir is parallel with y-axis, up dir is z axis, otherwise is y-axis
        y_axis = dir_vec.new_tensor([0, 1., 0]).expand_as(dir_vec)
        y_axis = torch.cross(dir_vec, y_axis).norm(dim=-1, keepdim=True) < 1e-6
        up = torch.scatter_add(up, -1, y_axis + 1, 1 - y_axis.float() * 2)
    right_vec = F.normalize(torch.cross(up, dir_vec, dim=-1), dim=-1)  # 相机空间x轴方向
    up_vec = torch.cross(dir_vec, right_vec, dim=-1)  # 相机空间y轴方向
    shape = eye.shape
    if inv:
        Tv2w = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
        Tv2w[..., :3, 0] = right_vec
        Tv2w[..., :3, 1] = up_vec
        Tv2w[..., :3, 2] = dir_vec
        Tv2w[..., :3, 3] = eye
        return Tv2w
    R = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
    T = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
    R[..., 0, :3] = right_vec
    R[..., 1, :3] = up_vec
    R[..., 2, :3] = dir_vec
    T[..., :3, 3] = -eye
    world2view = R @ T
    return world2view


## 像素空间/裁剪空间相关
## fovx/ fovy 弧度值
def fovx_to_fovy(fovx, aspect=1.) -> Tuple[np.ndarray, Tensor]:
    if isinstance(fovx, Tensor):
        return torch.arctan(torch.tan(fovx * 0.5) / aspect) * 2.0
    else:
        return np.arctan(np.tan(fovx * 0.5) / aspect) * 2.0


def focal_length_to_fovy(focal_length: float, sensor_height: float):
    t = 0.5 * sensor_height / focal_length
    return 2 * (torch.arctan(t) if isinstance(t, Tensor) else np.arctan(t))


def fovy_to_focal_length(fovy: float, H: float):
    return H / (2 * (torch.tan(fovy * 0.5) if isinstance(fovy, Tensor) else np.tan(fovy * 0.5)))


def camera_intrinsics(focal=None, size=None, fovy=np.pi, inv=False, opengl=True, **kwargs) -> Tensor:
    """生成相机内参"""
    W, H = size
    if focal is None:
        focal = fovy_to_focal_length(fovy, H)
    cx, cy = 0.5 * W, 0.5 * H
    shape = [x.shape for x in [focal, cx, cy] if isinstance(x, Tensor)]
    if len(shape) > 0:
        shape = list(torch.broadcast_shapes(*shape))

    if inv:  # Ts2v
        fr = 1. / focal
        Ts2v = torch.zeros(shape + [3, 3], **kwargs)
        Ts2v[..., 0, 0] = fr
        Ts2v[..., 0, 2] = (-cx if opengl else -cx) * fr
        Ts2v[..., 1, 1] = -fr if opengl else fr
        Ts2v[..., 1, 2] = (cy if opengl else -cy) * fr
        Ts2v[..., 2, 2] = -1 if opengl else 1
        return Ts2v

    K = torch.zeros(shape + [3, 3], **kwargs)  # Tv2s
    K[..., 0, 0] = focal
    K[..., 0, 2] = -cx if opengl else cx
    K[..., 1, 1] = -focal if opengl else focal
    K[..., 1, 2] = -cy if opengl else cy
    K[..., 2, 2] = -1 if opengl else 1
    return K


def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None, size=None):
    """透视投影矩阵

    Args:
        fovy: 弧度. Defaults to 0.7854.
        aspect: 长宽比W/H. Defaults to 1.0.
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.
        size: (W, H)

    Returns:
        Tensor: 透视投影矩阵
    """
    shape = []
    if size is not None:
        aspect = size[0] / size[1]
    for x in [fovy, aspect, n, f]:
        if isinstance(x, Tensor):
            shape = x.shape
    Tv2c = torch.zeros(*shape, 4, 4, dtype=torch.float, device=device)
    y = np.tan(fovy * 0.5)
    Tv2c[..., 0, 0] = 1. / (y * aspect)
    Tv2c[..., 1, 1] = -1. / y
    Tv2c[..., 2, 2] = -(f + n) / (f - n)
    Tv2c[..., 2, 3] = -(2 * f * n) / (f - n)
    Tv2c[..., 3, 2] = -1
    return Tv2c


def ortho(size=1., aspect=1.0, n=0.1, f=1000.0, device=None):
    """正交投影矩阵

    Args:
        size: 长度. Defaults to 1.0.
        aspect: 长宽比W/H. Defaults to 1.0.
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.

    Returns:
        Tensor: 正交投影矩阵
    """
    # yapf: disable
    return torch.tensor([
            [1 / (size * aspect), 0,        0,                  0                ], # noqa
            [0,                   1 / size, 0,                  0                ], # noqa
            [0,                   0,       -(f + n) / (f - n), -(f + n) / (f - n)],
            [0,                   0,        0,                  0                ], # noqa
    ], dtype=torch.float32, device=device)
    # yapf: enable


def test():
    from extension.utils import set_printoptions
    set_printoptions()
    print('fovy <--> focal:', fovy_to_focal_length(focal_length_to_fovy(10, 10), 10))
    # print(translate(0, 0, -torch.ones(2, 3)).cuda().shape)
    # eye = torch.tensor([1, 0, 1.])
    eye = torch.randn(3)
    # eye = coord_spherical_to(0.1, np.deg2rad(30), np.deg2rad(270))
    print(eye, coord_spherical_to(*coord_to_spherical(eye)))
    at = torch.tensor([0, 0, 0.])
    up = torch.tensor([0, 1, 0.])
    pose = look_at(eye, at, up, True)
    print(pose @ look_at(eye, at, up))
    # with vis3d:
    #     vis3d.add_camera_poses(pose, color=(1, 0, 0))
