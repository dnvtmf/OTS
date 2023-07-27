"""Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/camera.py"""
import torch
from torch import Tensor
from .misc import to_4x4


class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    @classmethod
    def so3_to_SO3(cls, w):  # [...,3]
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    @classmethod
    def SO3_to_so3(cls, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        # ln(R) will explode if theta==pi
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[..., None, None] % torch.pi
        lnR = 1 / (2 * cls.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    @classmethod
    def se3_to_SE3(cls, wu: Tensor):  # [...,6]
        w, u = wu.split([3, 3], dim=-1)
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        C = cls.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return to_4x4(Rt)

    @classmethod
    def SE3_to_se3(cls, Rt: Tensor, eps=1e-8):  # [...,3/4,4]
        R, t = Rt[..., :3, :].split([3, 1], dim=-1)
        w = cls.SO3_to_so3(R)
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    @classmethod
    def skew_symmetric(cls, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([
            torch.stack([O, -w2, w1], dim=-1),
            torch.stack([w2, O, -w0], dim=-1),
            torch.stack([-w1, w0, O], dim=-1)
        ],
            dim=-2)
        return wx

    @classmethod
    def taylor_A(cls, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_B(cls, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_C(cls, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans
