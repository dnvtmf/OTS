""" Swin Transformer
based on timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py

A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030
Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below
S3 (AutoFormerV2, https://arxiv.org/abs/2111.14725) Swin weights from
    - https://github.com/microsoft/Cream/tree/main/AutoFormerV2
Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from extension.utils import n_tuple
from extension.utils.timm_utils import trunc_normal_
from .base import BackboneBase


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = n_tuple(patch_size, 2)
        self.patch_size = patch_size
        self.flatten = flatten
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if H % self.patch_size[1] != 0 or W % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[0],
                          0, self.patch_size[1] - H % self.patch_size[1]))
        # assert H == self.img_size[1], f"Input image height ({H}) doesn't match model ({self.img_size[1]})."
        # assert W == self.img_size[0], f"Input image width ({W}) doesn't match model ({self.img_size[0]})."
        x = self.proj(x)
        H, W = x.shape[-2:]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(patch_size={self.patch_size}, embed_dim={self.embed_dim}, " \
               f"flatten={self.flatten})"


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# @register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = n_tuple(bias, 2)
        drop_probs = n_tuple(drop, 2)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self, dim, num_heads, window_size=7, shift_size=0,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=n_tuple(self.window_size, 2), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}, " \
               f"window_size={self.window_size})"


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H:
            W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H:
            W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            return x, H, W, self.downsample(x, H, W), (H + 1) // 2, (W + 1) // 2
        else:
            return x, H, W, x, H, W

    def __repr__(self):
        if self.downsample is None:
            return f"{self.__class__.__name__}({self.blocks[0]} * {self.depth})"
        else:
            return f"{self.__class__.__name__}(\n  {self.blocks[0]} * {self.depth}\n  {self.downsample}\n)"


# @BACKBONES.register()
class SwinTransformer(BackboneBase):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        in_channels=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        head_dim=None,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        global_pool='avg',
        output_strides=(4, 8, 16, 32),
        **kwargs
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = n_tuple(image_size, 2)
            patch_size = n_tuple(patch_size, 2)
            patch_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patch_resolution[0], patch_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        window_size = n_tuple(window_size, self.num_layers)
        layers = []
        for i in range(self.num_layers):
            layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            ))
        self.layers = nn.ModuleList(layers)

        # self.num_features = embed_dim * 2**(self.num_layers-1)
        # self.norm = norm_layer(self.num_features)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        for i in range(self.num_layers):
            self._channels[patch_size * 2 ** i] = self.embed_dim * 2 ** i
        self.output_strides = output_strides
        for stride in self.output_strides:
            self.add_module(f"norm_{stride}", norm_layer(self.channels[stride]))
        self.pretrained_cfg = dict(key='models', replace={
            r".*.attn_mask": '',
            # r"norm.(.*)": "layers.7.{0}",
        })
        self.reset_parameters()
        return

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        features = {}
        if 1 in self.output_strides:
            features['1'] = x
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        H, W = (H + self.patch_size - 1) // self.patch_size, (W + self.patch_size - 1) // self.patch_size
        if self.ape:
            # interpolate the position embedding to the corresponding size
            # x = x + F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic")
            raise NotImplementedError()
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            feature, Hf, Wf, x, H, W = self.layers[i](x, H, W)
            stride = self.patch_size * 2 ** i
            # features[str(stride)] = feature
            if stride in self.output_strides:
                feature = getattr(self, f"norm_{stride}")(feature)
                features[str(stride)] = feature.transpose(1, 2).view(B, -1, Hf, Wf).contiguous()
        # x = self.norm(x)
        # if self.global_pool == 'avg':
        #     x = x.mean(dim=1)
        # x = self.head(x)
        return features

    @property
    def stages(self, by_stride=True):
        return NotImplemented


_swin_url_prefix = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/'


# @BACKBONES.register()
def SwinT(variant='base', **kwargs):
    """
    base (swin_tiny_patch4_window7_224): Swin-T @ 224x224, trained ImageNet-1k
    S3 (swin_s3_tiny_224): Swin-S3-T @ 224x224, ImageNet-1k. https://arxiv.org/abs/2111.14725
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('window_size', (7, 7, 14, 7) if variant == 'S3' else 7)
    kwargs.setdefault('embed_dim', 96)
    kwargs.setdefault('depths', (2, 2, 6, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('image_size', 224)
    net = SwinTransformer(**kwargs)
    if variant == 'base':
        net.download_url = _swin_url_prefix + 'swin_tiny_patch4_window7_224.pth'
    elif variant == 'S3':
        net.download_url = 'https://github.com/silent-chen/AutoFormerV2-model-zoo/releases/download/v1.0.0/S3-T.pth'
    return net


# @BACKBONES.register()
def SwinS(variant='base', **kwargs):
    """
    base (swin_small_patch4_window7_224): Swin-S @ 224x224, trained ImageNet-1k
    S3 (swin_s3_small_224) :Swin-S3-S @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('window_size', (14, 14, 14, 7) if variant == 'S3' else 7)
    kwargs.setdefault('embed_dim', 96)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('image_size', 224)
    net = SwinTransformer(**kwargs)
    if variant == 'base':
        net.download_url = _swin_url_prefix + 'swin_small_patch4_window7_224.pth'
    elif variant == 'S3':
        net.download_url = 'https://github.com/silent-chen/AutoFormerV2-model-zoo/releases/download/v1.0.0/S3-S.pth'
    return net


# @BACKBONES.register()
def SwinB(variant='base224_22K', **kwargs):
    """
    base384 (swin_base_patch4_window12_384): Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    base224 (swin_base_patch4_window7_224): Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    base384_22K (swin_base_patch4_window12_384_in22k): Swin-B @ 384x384, trained ImageNet-22k
    base224_22K (swin_base_patch4_window7_224_in22k): Swin-B @ 224x224, trained ImageNet-22k
    S3 (swin_s3_base_224): Swin-S3-B @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('window_size', (7, 7, 14, 7) if variant == 'S3' else 7 if '224' in variant else 12)
    kwargs.setdefault('embed_dim', 96 if variant == 'S3' else 128)
    kwargs.setdefault('depths', (2, 2, 30, 2) if variant == 'S3' else (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24) if variant == 'S3' else (4, 8, 16, 32))
    kwargs.setdefault('image_size', 384 if '384' in variant else 224)
    kwargs.setdefault('num_classes', 21841 if variant.endswith('_22K') else 1000)
    net = SwinTransformer(**kwargs)
    if variant == 'base384':
        net.download_url = _swin_url_prefix + 'swin_base_patch4_window12_384_22kto1k.pth'
    elif variant == 'base224':
        net.download_url = _swin_url_prefix + 'swin_base_patch4_window7_224_22kto1k.pth'
    elif variant == 'base384_22K':
        net.download_url = _swin_url_prefix + 'swin_base_patch4_window12_384_22k.pth'
    elif variant == 'base224_22K':
        net.download_url = _swin_url_prefix + 'swin_base_patch4_window7_224_22k.pth'
    elif variant == 'S3':
        net.download_url = 'https://github.com/silent-chen/AutoFormerV2-model-zoo/releases/download/v1.0.0/S3-B.pth'
    return net


# @BACKBONES.register()
def SwinL(variant='base224_22K', **kwargs):
    """
    base384 (swin_large_patch4_window12_384): Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k
    base224 (swin_large_patch4_window7_224): Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k
    base384_22K (swin_large_patch4_window12_384_in22k): Swin-L @ 384x384, trained ImageNet-22k
    base224_22K (swin_large_patch4_window7_224_in22k):  Swin-L @ 224x224, trained ImageNet-22k
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('patch_size', 4)
    kwargs.setdefault('window_size', 12 if '384' in variant else 7)
    kwargs.setdefault('embed_dim', 192)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (6, 12, 24, 48))
    kwargs.setdefault('image_size', 384 if '384' in variant else 224)
    kwargs.setdefault('num_classes', 21841 if variant.endswith('_22K') else 1000)
    net = SwinTransformer(**kwargs)
    if variant == 'base384':
        net.download_url = _swin_url_prefix + 'swin_large_patch4_window12_384_22kto1k.pth'
    elif variant == 'base224':
        net.download_url = _swin_url_prefix + 'swin_large_patch4_window7_224_22kto1k.pth'
    elif variant == 'base384_22K':
        net.download_url = _swin_url_prefix + 'swin_large_patch4_window12_384_22k.pth'
    elif variant == 'base224_22K':
        net.download_url = _swin_url_prefix + 'swin_large_patch4_window7_224_22k.pth'
    return net


def test():
    from extension.utils import show_shape
    print()
    image_size = 512
    net = SwinB(variant='base384_22K', image_size=(384, 384))
    # net = SwinB(variant='S3', image_size=(224, 224))
    print(net)
    print(net.channels)
    net.load_pretrained_model(strict=False)

    x = torch.randn(2, 3, image_size, image_size)
    print('input:', x.shape)
    y = net(x)
    print('output:', show_shape(y))
