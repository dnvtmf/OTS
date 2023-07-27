import os
from pathlib import Path
from typing import List

import numpy as np
import nvdiffrast.torch as dr
import torch
from torch import Tensor, nn

from extension.ops_3d.misc import normalize
from extension.utils.io.image import avg_pool_nhwc, load_image, rgb_to_srgb, save_image, scale_img_nhwc, srgb_to_rgb
from extension.utils.io.material import MTL


######################################################################################
# Wrapper to make materials behave like a python dict, but register textures as torch.nn.Module parameters.
######################################################################################
class Material(nn.Module):

    def __init__(self, mat_dict):
        super(Material, self).__init__()
        self.mat_keys = set()
        for key in mat_dict.keys():
            self.mat_keys.add(key)
            self[key] = mat_dict[key]

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        self.mat_keys.add(key)
        setattr(self, key, val)

    def __delitem__(self, key):
        self.mat_keys.remove(key)
        delattr(self, key)

    def keys(self):
        return self.mat_keys

    @classmethod
    @torch.no_grad()
    def from_mtl(cls, mtl: MTL, clear_ks=False):
        mat = Material({'name': mtl.name})
        if not 'bsdf' in mat:
            mat['bsdf'] = 'pbr'

        def _deal(v, m):
            v = torch.from_numpy(v) if v is not None else 1.
            return v * m[..., :3] if m is not None else v

        mat['ka'] = Texture2D(_deal(mtl.Ka, mtl.map_Ka))
        mat['kd'] = Texture2D(_deal(mtl.Kd, mtl.map_Kd))
        mat['ks'] = Texture2D(_deal(mtl.Ks, mtl.map_Ks))
        if mtl.bump is not None:
            mat['normal'] = Texture2D(torch.from_numpy(mtl.bump[..., :3] * 2 - 1.))

        # Convert Kd from sRGB to linear RGB
        mat['kd'] = mat['kd'].srgb_to_rgb()
        if clear_ks:
            # Override ORM occlusion (red) channel by zeros. We hijack this channel
            for mip in mat['ks'].getMips():
                mip[..., 0] = 0.0
        return mat

    @classmethod
    @torch.no_grad()
    def load(cls, filename, clear_ks=True):
        import re
        mtl_path = Path(filename).parent

        # Read file
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Parse materials
        materials = []
        for line in lines:
            split_line = re.split(' +|\t+|\n+', line.strip())
            prefix = split_line[0].lower()
            data = split_line[1:]
            if 'newmtl' in prefix:
                material = Material({'name': data[0]})
                materials += [material]
            elif materials:
                if 'bsdf' in prefix or 'map_kd' in prefix or 'map_ks' in prefix or 'bump' in prefix:
                    material[prefix] = data[0]
                else:
                    material[prefix] = torch.tensor(tuple(float(d) for d in data), dtype=torch.float32)
        # Convert everything to textures.
        for mat in materials:
            if not 'bsdf' in mat:
                mat['bsdf'] = 'pbr'

            if 'map_kd' in mat:
                mat['kd'] = Texture2D.load(os.path.join(mtl_path, mat['map_kd']))
            else:
                mat['kd'] = Texture2D(mat['kd'])

            if 'map_ks' in mat:
                mat['ks'] = Texture2D.load(os.path.join(mtl_path, mat['map_ks']), channels=3)
            else:
                mat['ks'] = Texture2D(mat['ks'])

            if 'bump' in mat:
                mat['normal'] = Texture2D.load(mtl_path / mat['bump'], lambda_fn=lambda x: x * 2 - 1, channels=3)

            # Convert Kd from sRGB to linear RGB
            mat['kd'] = mat['kd'].srgb_to_rgb()

            if clear_ks:
                # Override ORM occlusion (red) channel by zeros. We hijack this channel
                for mip in mat['ks'].getMips():
                    mip[..., 0] = 0.0
        return materials

    @torch.no_grad()
    def save(self, filename):
        folder = os.path.dirname(filename)
        with open(filename, "w") as f:
            f.write('newmtl defaultMat\n')
            f.write('bsdf   %s\n' % self['bsdf'])
            if 'kd' in self.keys():
                f.write('map_Kd texture_kd.png\n')
                self['kd'].rgb_to_srgb().save(os.path.join(folder, 'texture_kd.png'))
            if 'ks' in self:
                f.write('map_Ks texture_ks.png\n')
                self['ks'].save(os.path.join(folder, 'texture_ks.png'))
            if 'normal' in self.keys():
                f.write('bump texture_n.png\n')
                self['normal'].save(os.path.join(folder, 'texture_n.png'), lambda_fn=lambda x: (normalize(x) + 1) * 0.5)
            else:
                f.write('Kd 1 1 1\n')
                f.write('Ks 0 0 0\n')
                f.write('Ka 0 0 0\n')
                f.write('Tf 1 1 1\n')
                f.write('Ni 1\n')
                f.write('Ns 0\n')


class Texture2D(nn.Module):
    """ Simple texture class. 
    
    A texture can be either
        - A 3D tensor (using auto mipmaps)
        - A list of 3D tensors (full custom mip hierarchy)
    """

    def __init__(self, init, min_max=None):
        """ Initializes a texture from image data.
        
        Input can be constant value (1D array) or texture (3D array) or mip hierarchy (list of 3d arrays)
        """
        super(Texture2D, self).__init__()

        if isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32)
        elif isinstance(init, list) and len(init) == 1:
            init = init[0]

        if isinstance(init, list):
            self.data = nn.ModuleList(nn.Parameter(mip.clone().detach(), requires_grad=True) for mip in init)
        elif isinstance(init, Tensor):
            assert init.ndim <= 4
            self.data = nn.Parameter(init.clone().detach().view([1] * (4 - init.ndim) + list(init.shape)))
        else:
            raise ValueError("Invalid texture object")

        self.min_max = min_max

    # Filtered (trilinear) sample texture at a given location
    def sample(self, texc, texc_deriv, filter_mode='linear-mipmap-linear'):
        if isinstance(self.data, list):
            out = dr.texture(self.data[0], texc, texc_deriv, mip=self.data[1:], filter_mode=filter_mode)
        else:
            if self.data.shape[1] > 1 and self.data.shape[2] > 1:
                mips = [self.data]
                while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                    mips += [texture2d_mip.apply(mips[-1])]
                out = dr.texture(mips[0], texc, texc_deriv, mip=mips[1:], filter_mode=filter_mode)
            else:
                out = dr.texture(self.data, texc, texc_deriv, filter_mode=filter_mode)
        return out

    @property
    def size(self):
        height, width = self.getMips()[0].shape[1:3]
        return width, height

    @property
    def channels(self):
        return self.getMips()[0].shape[3]

    @property
    def width(self):
        return self.getMips()[0].shape[2]

    @property
    def height(self):
        return self.getMips()[0].shape[1]

    def getRes(self):
        return self.getMips()[0].shape[1:3]

    def getChannels(self):
        return self.getMips()[0].shape[3]

    def getMips(self):
        if isinstance(self.data, list):
            return self.data
        else:
            return [self.data]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        if self.min_max is not None:
            for mip in self.getMips():
                for i in range(mip.shape[-1]):
                    mip[..., i].clamp_(min=self.min_max[0][i], max=self.min_max[1][i])

    # In-place clamp with no derivative to make sure values are in valid range after training
    def normalize_(self):
        with torch.no_grad():
            for mip in self.getMips():
                mip = torch.nn.functional.normalize(mip)

    def extra_repr(self) -> str:
        return f"{list(self.data.shape)}"

    def srgb_to_rgb(self):
        return Texture2D(list(srgb_to_rgb(mip) for mip in self.getMips()))

    def rgb_to_srgb(self):
        return Texture2D(list(rgb_to_srgb(mip) for mip in self.getMips()))

    @classmethod
    @torch.no_grad()
    def create_trainable(cls, init, res=None, auto_mipmaps=True, min_max=None) -> 'Texture2D':
        if isinstance(init, cls):
            assert isinstance(init.data, torch.Tensor)
            min_max = init.min_max if min_max is None else min_max
            init = init.data
        elif isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')

        # Pad to NHWC if needed
        if len(init.shape) == 1:  # Extend constant to NHWC tensor
            init = init[None, None, None, :]
        elif len(init.shape) == 3:
            init = init[None, ...]

        # Scale input to desired resolution.
        if res is not None:
            init = scale_img_nhwc(init, res)

        # Genreate custom mipchain
        if not auto_mipmaps:
            mip_chain = [init.clone().detach().requires_grad_(True)]
            while mip_chain[-1].shape[1] > 1 or mip_chain[-1].shape[2] > 1:
                new_size = [max(mip_chain[-1].shape[1] // 2, 1), max(mip_chain[-1].shape[2] // 2, 1)]
                mip_chain += [scale_img_nhwc(mip_chain[-1], new_size)]
            return cls(mip_chain, min_max=min_max)
        else:
            return cls(init, min_max=min_max)

    @classmethod
    def load(cls, filename, lambda_fn=None, channels=None):
        def _load_mip2D(fn, lambda_fn=None, channels=None):
            image = load_image(fn)
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image)
            if channels is not None:
                image = image[..., 0:channels]
            if lambda_fn is not None:
                image = lambda_fn(image)
            return image

        base, ext = os.path.splitext(filename)
        if os.path.exists(base + "_0" + ext):
            mips = []
            while os.path.exists(base + ("_%d" % len(mips)) + ext):
                mips += [_load_mip2D(base + ("_%d" % len(mips)) + ext, lambda_fn, channels)]
            return Texture2D(mips)
        else:
            return Texture2D(_load_mip2D(filename, lambda_fn, channels))

    @torch.no_grad()
    def save(self, filename, lambda_fn=None):
        def _save_mip2D(fn: Path, mip, mipidx=None, lambda_fn=None):
            data = (mip if lambda_fn is None else lambda_fn(mip)).detach().cpu().numpy()
            save_image(fn.with_name(f"{fn.stem}{'' if mipidx is None else f'_{mipidx:d}'}{fn.suffix}"), data)

        if isinstance(self.data, list):
            for i, mip in enumerate(self.data):
                _save_mip2D(filename, mip[0, ...], i, lambda_fn)
        else:
            _save_mip2D(filename, self.data[0, ...], None, lambda_fn)


class texture2d_mip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, texture):
        # y = texture.permute(0, 3, 1, 2)  # NHWC -> NCHW
        # y = torch.nn.functional.avg_pool2d(y, (2, 2))
        # return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
        return avg_pool_nhwc(texture, (2, 2))

    @staticmethod
    def backward(ctx, dout):
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 0.25 / dout.shape[1], 1.0 - 0.25 / dout.shape[1], dout.shape[1] * 2, device="cuda"),
            torch.linspace(0.0 + 0.25 / dout.shape[2], 1.0 - 0.25 / dout.shape[2], dout.shape[2] * 2, device="cuda"),
            indexing='ij'
        )
        uv = torch.stack((gx, gy), dim=-1)
        return dr.texture(dout * 0.25, uv[None, ...].contiguous(), filter_mode='linear', boundary_mode='clamp')


######################################################################################
# Merge multiple materials into a single uber-material
######################################################################################


def _upscale_replicate(x, full_res):
    x = x.permute(0, 3, 1, 2)
    x = torch.nn.functional.pad(x, (0, full_res[1] - x.shape[3], 0, full_res[0] - x.shape[2]), 'replicate')
    return x.permute(0, 2, 3, 1).contiguous()


def merge_materials(materials: List[Material], v_tex: Tensor, f_tex: Tensor, f_mat: Tensor, mode='wrap', eps=0.01):
    """ 合并多个Material

    Args:
        materials: 材质Material列表
        v_tex: 纹理坐标 shape: [V, 2], range: [0., 1]
        f_tex: 纹理面片 shape: [E, 3], range: [0, E-1]
        f_mat: 每个面皮使用的纹理编号 shape: [E]
        mode: 对在值在[0, 1]之外的纹理坐标的处理方法. 'wrap': 重复, 'clamp': 约束到[0, 1], 'zero': 置为0
        eps: 将纹理坐标约束在[eps, 1-eps]之间, 避免纹理插值错误

    Returns:
        Material, Tensor, Tensor: 合并后的材质, 新的纹理坐标, 新的纹理面片
    """
    assert len(materials) > 0
    for mat in materials:
        assert mat['bsdf'] == materials[0]['bsdf'], "All materials must have the same BSDF (uber shader)"
        assert ('normal'
                in mat) is ('normal' in materials[0]), "All materials must have either normal map enabled or disabled"

    uber_material = Material({
        'name': 'uber_material',
        'bsdf': materials[0]['bsdf'],
    })

    textures = ['ka', 'kd', 'ks', 'normal']

    # Find maximum texture resolution across all materials and textures
    max_res = None
    for mat in materials:
        for tex in textures:
            tex_res = np.array(mat[tex].getRes()) if tex in mat else np.array([1, 1])
            max_res = np.maximum(max_res, tex_res) if max_res is not None else tex_res

    # Compute size of compund texture and round up to nearest PoT
    full_res = 2 ** np.ceil(np.log2(max_res * np.array([1, len(materials)]))).astype(np.int32)

    # Normalize texture resolution across all materials & combine into a single large texture
    for tex in textures:
        if tex in materials[0]:
            # Lay out all textures horizontally, NHWC so dim2 is x
            tex_data = torch.cat(tuple(scale_img_nhwc(mat[tex].data, tuple(max_res)) for mat in materials), dim=2)
            tex_data = _upscale_replicate(tex_data, full_res)
            uber_material[tex] = Texture2D(tex_data)

    # Recompute texture coordinates to cooincide with new composite texture
    if mode == 'wrap':
        v_tex = torch.remainder(v_tex, 1.0)
    elif mode == 'clamp':
        # v_tex = v_tex.clamp(eps, 1. - eps)
        pass
    elif mode == 'zero':
        v_tex = torch.where(0 <= v_tex and v_tex <= 1, v_tex, torch.zeros_like(v_tex))
    else:
        raise NotImplementedError(f"mode '{mode}' is not supported when merge materials")
    v_tex = v_tex.clamp(eps, 1. - eps)
    output, new_f_tex = torch.unique(f_tex + f_mat[:, None] * v_tex.shape[0], return_inverse=True)
    new_v_tex = v_tex[output % v_tex.shape[0], :]
    new_v_tex[:, 0] = (new_v_tex[:, 0] + output // v_tex.shape[0]) * (max_res[1] / full_res[1])
    return uber_material, new_v_tex, new_f_tex
