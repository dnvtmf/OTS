"""" prepare normals for shading
base on https://github.com/NVlabs/nvdiffrec
"""
import nvdiffrast.torch as dr
import torch
from torch import Tensor

from .misc import dot, normalize

NORMAL_THRESHOLD = 0.1

__all__ = ['prepare_shading_normal', 'compute_shading_normal', 'compute_shading_normal_face']


def _bend_normal(view_vec, smooth_nrm, geom_nrm, two_sided_shading):
    # Swap normal direction for backfacing surfaces
    if two_sided_shading:
        # smooth_nrm = torch.where(dot(geom_nrm, view_vec) > 0, smooth_nrm, -smooth_nrm)
        # geom_nrm = torch.where(dot(geom_nrm, view_vec) > 0, geom_nrm, -geom_nrm)
        sign = dot(geom_nrm, view_vec).sign()
        smooth_nrm *= sign
        geom_nrm *= sign

    t = torch.clamp(dot(view_vec, smooth_nrm) / NORMAL_THRESHOLD, min=0, max=1)
    return torch.lerp(geom_nrm, smooth_nrm, t)


def _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl):
    smooth_bitang = normalize(torch.cross(smooth_tng, smooth_nrm))
    shading_nrm = smooth_tng * perturbed_nrm[..., 0:1]
    if opengl:
        shading_nrm -= smooth_bitang * perturbed_nrm[..., 1:2]
    else:
        shading_nrm += smooth_bitang * perturbed_nrm[..., 1:2]
    shading_nrm += smooth_nrm * torch.clamp(perturbed_nrm[..., 2:3], min=0.0)
    return normalize(shading_nrm)


def bsdf_prepare_shading_normal(
    pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl
):
    smooth_nrm = normalize(smooth_nrm)
    smooth_tng = normalize(smooth_tng)
    view_vec = normalize(view_pos - pos)
    shading_nrm = _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl)
    return _bend_normal(view_vec, shading_nrm, geom_nrm, two_sided_shading)


#
# class _prepare_shading_normal_func(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
#         ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
#         out = _get_plugin().prepare_shading_normal_fwd(
#             pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl, False
#         )
#         ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
#         return out
#
#     @staticmethod
#     def backward(ctx, dout):
#         pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
#         return _get_plugin().prepare_shading_normal_bwd(
#             pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl
#         ) + (None, None, None)


def prepare_shading_normal(
    pos: Tensor,
    view_pos: Tensor,
    perturbed_nrm: Tensor,
    smooth_nrm: Tensor,
    smooth_tng: Tensor,
    geom_nrm: Tensor,
    two_sided_shading=True,
    opengl=True,
    use_python=True
) -> Tensor:
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''

    if perturbed_nrm is None:
        perturbed_nrm = smooth_tng.new_tensor([0, 0, 1], requires_grad=False)[None, None, None, ...]

    # if use_python:
    out = bsdf_prepare_shading_normal(
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl
    )
    # else:
    #     out = _prepare_shading_normal_func.apply(
    #         pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl
    #     )

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out


def compute_shading_normal(
    mesh, view_pos: Tensor, rast: Tensor, perturbed_nrm: Tensor = None, two_sided_shading=True
) -> Tensor:
    # dr.interpolate(
    #     attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

    # Interpolate world space position
    gb_pos, _ = dr.interpolate(mesh.v_pos[None, ...], rast, mesh.f_pos.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.f_pos[:, 0], :]
    v1 = mesh.v_pos[mesh.f_pos[:, 1], :]
    v2 = mesh.v_pos[mesh.f_pos[:, 2], :]
    face_normals = normalize(torch.linalg.cross(v1 - v0, v2 - v0))
    face_normal_indices = torch.arange(0, face_normals.shape[0], dtype=torch.int32, device=v0.device)
    face_normal_indices = face_normal_indices[:, None].repeat(1, 3)
    gb_geometric_normal, _ = dr.interpolate(face_normals[None, ...], rast, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = dr.interpolate(mesh.v_nrm[None, ...], rast, mesh.f_nrm.int())
    gb_tangent, _ = dr.interpolate(mesh.v_tng[None, ...], rast, mesh.f_tng.int())  # Interpolate tangents

    gb_normal = prepare_shading_normal(
        gb_pos,
        view_pos,
        perturbed_nrm,
        gb_normal,
        gb_tangent,
        gb_geometric_normal,
        two_sided_shading=two_sided_shading,
        opengl=True
    )
    return gb_normal


def compute_shading_normal_face(mesh, view_pos: Tensor, rast: Tensor, v_pos: Tensor = None, two_sided_shading=True):
    """使用triangle的法线作为渲染法线"""
    v_pos = mesh.v_pos if v_pos is None else v_pos
    v0 = v_pos[mesh.f_pos[:, 0], :3]
    v1 = v_pos[mesh.f_pos[:, 1], :3]
    v2 = v_pos[mesh.f_pos[:, 2], :3]
    v_nrm = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, v_nrm.new_tensor([0.0, 0.0, 1.0]))

    nrm = torch.constant_pad_nd(v_nrm, (0, 0, 1, 0), 0)[rast[..., -1].long()]
    # nrm = torch.zeros_like(rast[..., :3])
    # mask = rast[..., -1] > 0
    # nrm[mask, :] = v_nrm[rast[mask, :][:, -1].long() - 1]
    nrm = normalize(nrm)

    if two_sided_shading:  # Swap normal direction for backfacing surfaces
        assert view_pos.ndim == rast.ndim, f"view_pos must be same dimention with rast"
        # nrm = torch.where(dot(nrm, view_pos) < 0, -nrm, nrm)
        nrm = nrm * dot(nrm, view_pos).sign()
    return nrm
