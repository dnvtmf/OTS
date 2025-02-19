"""旋转表示方法
matrix: 3x3矩阵
eluer: 欧拉角 有万向节死锁(Gimbal Lock)的问题
quaternion: 四元数 q=(x, y, z, w)
axis_angle: (u, theta) 向量u表示旋转轴, theta表示绕旋转轴逆时针旋转的弧度
rotvec: v 其归一化向量表示旋转轴, 模长表示绕旋转轴逆时针旋转的弧度

reference: 
- https://zhuanlan.zhihu.com/p/45404840
- http://motion.pratt.duke.edu/RoboticSystems/3DRotations.html
"""
import torch
from torch import Tensor
from . import quaternion


def _vec2ss_matrix(vector: Tensor):
    """vector to skew-symmetric 3x3 matrix"""
    a, b, c = vector.unbind(-1)
    zeros = torch.zeros_like(a)
    return torch.stack([zeros, -c, b, c, zeros, -a, -b, a, zeros], dim=-1).reshape(*a.shape, 3, 3)


def euler_to_R(xyz, order='xyz'):
    assert len(order) == 3 and xyz.shape[-1] == 3
    R = None
    for axis, angle in zip(order, xyz.unbind(-1)):
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        if axis == "x":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either x, t or z.")

        Ro = torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
        R = Ro if R is None else R @ Ro
    return R


def euler_to_quaternion(x=0, y=0, z=0., order='xyz'):
    q = None
    x, y, z = [(i if isinstance(i, Tensor) else torch.tensor(i)) for i in [x, y, z]]
    zeros = torch.zeros_like(x)
    for o in order:
        if o == 'x':
            x = 0.5 * x
            qo = torch.cat([x.sin(), zeros, zeros, x.cos()], dim=-1)
        elif o == 'y':
            y = 0.5 * y
            qo = torch.cat([zeros, y.sin(), zeros, y.cos()], dim=-1)
        else:
            z = 0.5 * z
            qo = torch.cat([zeros, zeros, z.sin(), z.cos()], dim=-1)
        q = qo if q is None else quaternion.mul(q, qo)
    return q


def quaternion_to_R(q: Tensor):
    if q.shape[-1] == 7:
        q = q[..., 3:]
    assert q.shape[-1] == 4  # [x, y, z, w]
    x, y, z, w = q.unbind(-1)
    R = torch.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * w * y + 2 * x * z,
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x,
        2 * x * z - 2 * w * y, 2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y,
    ], dim=-1).reshape(*x.shape, 3, 3) # yapf: disable
    return R


def quaternion_to_axis_angle(q: Tensor):
    if q.shape[-1] == 7:
        q = q[..., 3:]
    assert q.shape[-1] == 4  # [x, y, z, w]
    u, theta = quaternion.to_rotate(q)
    return u, theta


def axis_angle_to_R(u: Tensor, theta: Tensor = None):
    """convert axis_angle (w, theta) and translate (v) to 3x3 Rotation Matrix  

    Args:
        w : rotate axis, shape: [..., 3]
        theta: rotate radian, shape: [...], if None theta = |w|
    Returns:
        maxtrix with shape [..., 3, 3]
    """
    u_nrom = u.norm(dim=-1, keepdim=True)
    w_skewsym = _vec2ss_matrix(u / (u_nrom + 1e-15))
    theta = (u_nrom[..., None] if theta is None else theta[..., None, None])
    I = torch.eye(3, dtype=theta.dtype, device=theta.device)
    R = I + theta.sin() * w_skewsym + (1 - theta.cos()) * torch.matmul(w_skewsym, w_skewsym)
    return R


def axis_angle_to_quaternion(u: Tensor, theta: Tensor = None):
    if theta is None:
        theta = u.norm(dim=-1, keepdim=False)
        u = u / theta[..., None]
    q = quaternion.from_rotate(u, theta)
    return q


def lie_to_R(phi: Tensor) -> Tensor:
    theta = phi.norm(dim=-1, keepdim=False)
    a = phi / theta[..., None]
    I = torch.eye(3, device=phi.device)
    cos = theta.cos()[..., None, None]
    sin = theta.sin()[..., None, None]
    R = cos * I + (1 - cos) * a[..., None, :] * a[..., :, None] + sin * _vec2ss_matrix(a)
    return R


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"x": (2, 1), "y": (0, 2), "z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["xy", "yz", "zx"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def R_to_euler(R: Tensor, order='xyz'):
    """未处理万向锁问题"""
    i0 = {'x': 0, 'y': 1, 'z': 2}[order[0]]
    i2 = {'x': 0, 'y': 1, 'z': 2}[order[2]]
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(R[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(R[..., i0, i0])

    o = (
        _angle_from_tan(order[0], order[1], R[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(order[2], order[1], R[..., i0, :], True, tait_bryan),
    )
    return torch.stack(o, -1)


def R_to_quaternion(R: Tensor):
    w = 0.5 * torch.sqrt(R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] + 1)
    w_ = 1. / (4 * w + 1e-10)  # 避免数值不稳定, 存在不用除法的算法
    x = (R[..., 2, 1] - R[..., 1, 2]) * w_
    y = (R[..., 0, 2] - R[..., 2, 0]) * w_
    z = (R[..., 1, 0] - R[..., 0, 1]) * w_
    return torch.stack([x, y, z, w], dim=-1)


def R_to_axis_angle(R: Tensor):
    return quaternion_to_axis_angle(R_to_quaternion(R))


def R_to_lie(R: Tensor):

    import pytorch3d.transforms
    from lietorch import SO3
    quat = pytorch3d.transforms.matrix_to_quaternion(R[..., :3, :3])
    quat = torch.cat((quat[..., 1:], quat[..., 0:1]), -1)  # swap real first to real last
    Ps = SO3.InitFromVec(quat)
    return Ps.log()


def R_to_lie_np(matrix: Tensor):
    from lietorch import SO3
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(matrix.detach().cpu().numpy()).as_quat()
    T = SO3.InitFromVec(torch.from_numpy(quat).to(matrix))
    return T


def test():
    import pytorch3d.transforms
    from scipy.spatial.transform import Rotation
    from tree_segmentation.extension import utils
    torch.set_default_dtype(torch.float64)
    utils.set_printoptions(6)
    print()
    n = 10
    matrix = pytorch3d.transforms.random_rotations(n)
    R = R_to_lie_np(matrix)
    error = (R.matrix()[..., :3, :3] - matrix).sum()
    print(error)

    u = torch.randn(n, 3)
    u = u / u.norm(dim=-1, keepdim=True)
    theta = torch.rand(n) * 2 * torch.pi
    gt = torch.from_numpy(Rotation.from_rotvec((u * theta[..., None]).numpy()).as_matrix())
    gt2 = pytorch3d.transforms.axis_angle_to_matrix(u * theta[..., None])
    print(gt.shape, gt2.shape)

    def cmp(m, eps=1e-6):
        return (m - gt).abs().max() < eps and (m - gt2).abs().max() < eps

    m1 = axis_angle_to_R(u, theta)
    assert cmp(m1)
    gt_tq = R_to_lie_np(m1).data
    my_q = axis_angle_to_quaternion(u, theta)
    print(gt_tq[0], my_q[0])
    assert torch.min((gt_tq + my_q).abs(), (gt_tq - my_q).abs()).abs().max() < 1e-6
    m2 = quaternion_to_R(my_q)
    assert cmp(m2)
    my_q2 = R_to_quaternion(m2)
    assert torch.min((gt_tq + my_q2).abs(), (gt_tq - my_q2).abs()).abs().max() < 1e-5

    m3 = lie_to_R(R_to_lie(m2))
    assert cmp(m3)

    for xyz in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
        eluer = R_to_euler(gt, order=xyz)
        gt_eluer = pytorch3d.transforms.matrix_to_euler_angles(gt, xyz.upper())
        assert (eluer - gt_eluer).abs().max() < 1e-6
        m4 = euler_to_R(eluer, order=xyz)
        # m4 = pytorch3d.transforms.euler_angles_to_matrix(eluer, xyz.upper())
        print(m4[0], gt[0])
        assert cmp(m4), xyz