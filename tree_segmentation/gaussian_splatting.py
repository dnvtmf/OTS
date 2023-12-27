"""
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code: https://github.com/graphdeco-inria/gaussian-splatting

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
"""
import logging
import math
import os
from typing import NamedTuple, Mapping, Any, Dict, Optional, Union

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from torch import nn, Tensor

from tree_segmentation.extension import utils
from tree_segmentation.extension.gaussian_splatting import render, GaussianRasterizationSettings, topk_weights
from tree_segmentation.extension._C import get_C_function

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


class BasicPointCloud(NamedTuple):  # TODO: replace by PointClouds
    points: np.array
    colors: np.array
    normals: np.array


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class GaussianSplatting(nn.Module):
    param_names_map = {
        '_xyz': 'xyz',
        '_features_dc': 'f_dc',
        '_features_rest': 'f_rest',
        '_scaling': 'scaling',
        '_rotation': 'rotation',
        '_opacity': 'opacity'
    }
    max_radii2D: Tensor
    xyz_gradient_accum: Tensor
    denom: Tensor

    def __init__(
        self,
        sh_degree: int = 3,
        convert_SHs_python=False,
        compute_cov3D=False,
        loss_cfg: dict = None,
        **kwargs
    ):
        super().__init__()
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D = compute_cov3D
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, 3)
        self._features_dc = torch.empty(0, 1, 3)
        self._features_rest = torch.empty(0, (self.max_sh_degree + 1) ** 2 - 1, 3)
        self._scaling = torch.empty(0, 3)
        self._rotation = torch.empty(0, 4)
        self._opacity = torch.empty(0, 1)

        # self.max_radii2D = torch.empty(0)
        # self.xyz_gradient_accum = torch.empty(0)
        # self.denom = torch.empty(0)
        self.register_buffer('max_radii2D', torch.empty(0), persistent=False)
        self.register_buffer('xyz_gradient_accum', torch.empty(0, 1), persistent=False)
        self.register_buffer('denom', torch.empty(0, 1), persistent=False)
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self._step = -1

    def set_from_dataset(self, dataset):
        self.tanfovx = math.tan(0.5 * dataset.fovx)
        self.tanfovy = math.tan(0.5 * dataset.fovy)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        distCUDA2 = get_C_function('simple_knn')
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D.data = torch.zeros((self.points.shape[0]), device=self._xyz.device)

    def forward(self, scaling_modifier=1, campos: Tensor = None, **kwargs):
        points = self._xyz
        features = torch.cat((self._features_dc, self._features_rest), dim=1)
        outputs = {'points': points, 'opacity': self.opacity_activation(self._opacity)}
        if self.convert_SHs_python and campos is not None:
            shs_view = features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
            dir_pp = (points - campos.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            outputs['colors'] = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            outputs['sh_features'] = features

        if self.compute_cov3D:
            outputs['covariance'] = self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        else:
            outputs['scales'] = self.scaling_activation(self._scaling)
            outputs['rotations'] = self.rotation_activation(self._rotation)
        return outputs

    def render(self, *args, info, background: Tensor = None, scale_modifier=1.0, **kwargs):
        raster_settings = GaussianRasterizationSettings(
            image_width=info['size'][0],
            image_height=info['size'][1],
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            scale_modifier=scale_modifier,
            viewmatrix=info['Tw2v'].view(4, 4).transpose(-1, -2),
            projmatrix=info['Tw2c'].view(4, 4).transpose(-1, -2),
            sh_degree=self.active_sh_degree,
            campos=info['campos'],
            prefiltered=False,
            debug=False
        )
        outputs = render(**self(campos=info['campos']), raster_settings=raster_settings)
        images = torch.permute(outputs['images'], (1, 2, 0))
        if background is not None:
            images = images + (1 - outputs['opacity'][..., None]) * background.squeeze(0)
        outputs['images'] = images[None]
        return outputs

    def change_with_training_progress(self, step=0, num_steps=1, epoch=0, num_epochs=1):
        total_step = epoch * num_steps + step + 1
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if total_step % 1000 == 0 and self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        self._step = total_step

    def loss(self, inputs, outputs, targets):
        image = outputs['images']
        gt_img = targets['images'][..., :3]
        H, W, C = image.shape[-3:]
        image, gt_img = image.view(1, H, W, C), gt_img.view(1, H, W, C)
        losses = {'rgb': self.loss_funcs('image', image, gt_img), 'ssim': self.loss_funcs('ssim', image, gt_img)}
        return losses

    def construct_list_of_attributes(self):
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append('f_rest_{}'.format(i))
        attrs.append('opacity')
        for i in range(self._scaling.shape[1]):
            attrs.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            attrs.append('rot_{}'.format(i))
        # for name, opt_name in self.param_names_map.items():
        #     if name == 'xyz':
        #         continue
        #     channels = getattr(self, name)[0].numel()
        #     if channels == 1:
        #         attrs.append(opt_name)
        #     else:
        #         for i in range(channels):
        #             attrs.append(f"{opt_name}_{i}")
        return attrs

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1,
            2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1,
            2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        return plydata

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs):
        for name in self.param_names_map.keys():
            if name in state_dict:
                setattr(self, name, nn.Parameter(getattr(self, name).new_empty(state_dict[name].shape)))
                logging.debug(f'change the shape of parameters of {name}')
        super().load_state_dict(state_dict, strict=strict, **kwargs)
        N = self._xyz.shape[0]
        self.max_radii2D.data = torch.zeros((N,), device=self._xyz.device)

    def reset_opacity(self, optimizer: torch.optim.Optimizer):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        # optimizable_tensors = self.replace_tensor_to_optimizer(optimizer, opacities_new, "opacity")
        optimizable_tensors = self.change_optimizer(optimizer, opacities_new, name='opacity', op='replace')
        self._opacity = optimizable_tensors["opacity"]

    def densify_and_split(self, optimizer: torch.optim.Optimizer, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.points.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points,), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_params = {
            'xyz': torch.bmm(rots, samples[..., None]).squeeze(-1) + self.points[selected_pts_mask].repeat(N, 1),
            'scaling': self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        }
        for param_name, opt_name in self.param_names_map.items():
            if param_name == 'xyz' or param_name == 'scaling':
                continue
            param = getattr(self, param_name)
            new_params[opt_name] = param[selected_pts_mask].repeat(N, *[1] * (param.ndim - 1))

        self.densification_postfix(optimizer, **new_params, mask=selected_pts_mask, N=N)

        prune_filter = torch.cat((selected_pts_mask, selected_pts_mask.new_zeros(N * selected_pts_mask.sum())))
        self.prune_points(optimizer, prune_filter)

    def densify_and_clone(self, optimizer: torch.optim.Optimizer, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        masked_params = {}
        for param_name, opt_name in self.param_names_map.items():
            masked_params[opt_name] = getattr(self, param_name)[selected_pts_mask]
        self.densification_postfix(optimizer, **masked_params, mask=selected_pts_mask)

    def get_params(self, cfg):
        lr = cfg.lr
        # yapf: off
        w = 0.16 * self.spatial_lr_scale
        params_groups = [
            {'params': [self._xyz],           'lr': w * lr,     "name": "xyz"},
            {'params': [self._features_dc],   'lr': 2.5 * lr,   "name": "f_dc",     'fix': True},
            {'params': [self._features_rest], 'lr': 0.125 * lr, "name": "f_rest",   'fix': True},
            {'params': [self._opacity],       'lr': 50 * lr,    "name": "opacity",  'fix': True},
            {'params': [self._scaling],       'lr': 5 * lr,     "name": "scaling",  'fix': True},
            {'params': [self._rotation],      'lr': lr,         "name": "rotation", 'fix': True},
        ]
        # yapf: on
        # {'params': itertools.chain(m.parameters() for m in self.modules()), 'lr': lr}
        return params_groups

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def points(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def densify_and_prune(self, optimizer: torch.optim.Optimizer, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(optimizer, grads, max_grad, extent)
        self.densify_and_split(optimizer, grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(optimizer, prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def training_setup(self, percent_dense):
        self.percent_dense = percent_dense
        num_points = self._xyz.shape[0]
        self.xyz_gradient_accum.data = self._xyz.new_zeros((num_points, 1))
        self.denom.data = self._xyz.new_zeros((num_points, 1))

    @staticmethod
    def change_optimizer(optimizer, tensor, name=None, op='replace'):
        # type: (torch.optim.Optimizer, Union[Tensor, Dict[str, Tensor]], Optional[str],str) -> Dict[str, Tensor]
        """replace, prune, concat a tensor or tensor list in optimizer"""
        assert op in ['replace', 'prune', 'concat']
        optimizable_tensors = {}
        mask = None
        for group in optimizer.param_groups:
            if (('name' not in group) or (name is not None and group["name"] != name) or
                (isinstance(tensor, dict) and group['name'] not in tensor)):
                continue

            old_tensor = group['params'][0]
            new_tensor = tensor if isinstance(tensor, Tensor) else tensor[group['name']]
            if op == 'concat':
                group["params"][0] = nn.Parameter(torch.cat([old_tensor, new_tensor]).requires_grad_(True))
            elif op == 'prune':
                mask = new_tensor
                group["params"][0] = nn.Parameter(old_tensor[mask].requires_grad_(True))
            else:
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

            stored_state = optimizer.state.get(old_tensor, None)
            if stored_state is None:
                continue
            del optimizer.state[old_tensor]
            if op == 'concat':
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], torch.zeros_like(new_tensor)], dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    [stored_state["exp_avg_sq"], torch.zeros_like(new_tensor)], dim=0)
            elif op == 'prune':
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
            else:  # replace
                stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)
            optimizer.state[group['params'][0]] = stored_state
        return optimizable_tensors

    @staticmethod
    def replace_tensor_to_optimizer(optimizer: torch.optim.Optimizer, tensor: Tensor, name: str):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if 'name' in group and group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    @staticmethod
    def _prune_optimizer(optimizer: torch.optim.Optimizer, mask):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if 'name' not in group:
                continue
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, optimizer: torch.optim.Optimizer, mask):
        valid_points_mask = ~mask
        # optimizable_tensors = self._prune_optimizer(optimizer, valid_points_mask)
        optimizable_tensors = self.change_optimizer(optimizer, valid_points_mask, op='prune')

        for param_name, opt_name in self.param_names_map.items():
            setattr(self, param_name, optimizable_tensors[opt_name])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    @staticmethod
    def cat_tensors_to_optimizer(optimizer: torch.optim.Optimizer, tensors_dict: Dict[str, Tensor]):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if 'name' not in group:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor),
                    dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor),
                    dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, optimizer: torch.optim.Optimizer, mask=None, N=None, **kwargs):
        # optimizable_tensors = self.cat_tensors_to_optimizer(optimizer, kwargs)
        optimizable_tensors = self.change_optimizer(optimizer, tensor=kwargs, op='concat')
        for param_name, optim_name in self.param_names_map.items():
            setattr(self, param_name, optimizable_tensors[optim_name])

        num_points = self.points.shape[0]
        self.xyz_gradient_accum.data = self._xyz.new_zeros((num_points, 1))
        self.denom.data = self._xyz.new_zeros((num_points, 1))
        self.max_radii2D.data = self._xyz.new_zeros(num_points)
        return mask


def vis_trained():
    from tree_segmentation.extension.utils.gui.viewer_3D import simple_3d_viewer
    from tree_segmentation.extension import ops_3d

    path = '/home/wan/Projects/NeRF/gaussian-splatting/output/7f156431-5/point_cloud/iteration_30000/point_cloud.ply'
    model = GaussianSplatting(3)
    model.load_ply(path)

    @torch.no_grad()
    def rendering(Tw2v, fovy, size):
        Tw2v = Tw2v.cuda()
        Tw2v = ops_3d.convert_coord_system(Tw2v, 'opengl', 'colmap')
        Tv2c = ops_3d.perspective(size=size, fovy=fovy).cuda()
        # print(Tv2c)
        fovx = ops_3d.fovx_to_fovy(fovy, size[1] / size[0])
        Tv2c = ops_3d.perspective_v2(fovy, size=size).cuda()
        # print(Tv2c_2)
        # exit()
        Tw2c = Tv2c @ Tw2v
        Tv2w = torch.inverse(Tw2v)
        tanfovx = math.tan(0.5 * fovx)
        tanfovy = math.tan(0.5 * fovy)
        bg_color = [1, 1, 1]  # if dataset.background == 'white' else [0, 0, 0]
        bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        raster_settings = GaussianRasterizationSettings(
            image_height=size[1],
            image_width=size[0],
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=1.0,
            viewmatrix=Tw2v.T,
            projmatrix=Tw2c.T,
            sh_degree=model.max_sh_degree,
            campos=Tv2w[:3, 3],
            prefiltered=False,
            debug=False
        )
        return render(**model(), raster_settings=raster_settings)['images']

    simple_3d_viewer(rendering, size=(1024, 1024))


if __name__ == '__main__':
    vis_trained()
