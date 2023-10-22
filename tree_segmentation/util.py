from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import Tensor

from tree_segmentation import Tree2D, MaskData, Tree3Dv2, Tree3D
from tree_segmentation.extension import utils

__all__ = [
    'color_mask', 'get_colored_masks', 'show_masks', 'image_add_points', 'image_add_mask_boundary', 'show_all_levels',
    'get_hash_name', 'search_folder'
]


def color_mask(mask: np.ndarray, max_value=None, channels=3):
    if max_value is None:
        max_value = mask.max()  # noqa
    mask = mask.astype(np.int32)
    # cmap = matplotlib.colormaps['viridis'].resampled(max_value + 2)
    # random_order = np.arange(max_value + 2)
    # random_order[1:] = np.random.permutation(random_order[1:])
    colors = np.array([[1, 1, 1]] + sns.color_palette(n_colors=max_value))
    if channels == 4:
        colors = np.concatenate([colors, np.ones_like(colors[:, :1])], axis=-1)
    mask_image = colors[mask]
    # mask_image = np.where(mask[:, :, None] == 0, 1., mask_image)
    return mask_image.astype(np.float32)


def get_colored_masks(*masks, channels=3):
    if len(masks) == 1 and isinstance(masks[0], (tuple, list)):
        masks = masks[0]
    all_masks = []
    for mask in masks:
        if isinstance(mask, (Tree2D, MaskData)):
            mask = mask['masks']
        elif isinstance(mask, dict):
            mask = mask['segmentation']
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.ndim == 2:
            all_masks.append(mask[None])
        else:
            assert mask.ndim == 3
            all_masks.append(mask)
    if len(all_masks) == 0:
        return None
    else:
        all_masks = np.concatenate(all_masks, axis=0)
        one_mask = np.max(all_masks.astype(np.uint8) * np.arange(1, all_masks.shape[0] + 1)[:, None, None], axis=0)
    return color_mask(one_mask, len(all_masks), channels=channels)


def show_masks(image, *masks, mask=None, alpha=None):
    if image is not None:
        if isinstance(image, Tensor):
            image = utils.as_np_image(image)
        plt.imshow(image)
        alpha = 0.356 if alpha is None else alpha
    else:
        alpha = 1.0
    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    if mask is None:
        mask_image = get_colored_masks(*masks)
    else:
        assert len(masks) == 0
        if isinstance(mask, Tensor):
            mask = mask.detach().cpu().numpy()
        mask_image = color_mask(mask, mask.max())
    if mask_image is not None:
        plt.imshow(mask_image, alpha=alpha)
    plt.axis('off')
    # plt.show()


def image_add_points(image: np.ndarray, points: np.ndarray, s=5, color=(1, 0, 0)):
    image = image.copy()
    H, W = image.shape[:2]
    if points.max() <= 1.1:  # noqa
        points = points * np.array([W, H])
    color = np.array(color)
    assert 0 <= color.min() and color.max() <= 1 and color.shape == (3,)
    if image.dtype == np.uint8:
        color = (color * 255).astype(np.uint8)
    else:
        color = color.astype(image.dtype)
    for i in range(points.shape[0]):
        x, y = points[i]
        for dx in range(-s, s + 1):
            for dy in range(-s, s + 1):
                if dx * dx + dy * dy <= s * s and 0 <= x + dx < W and 0 <= y + dy < H:
                    image[int(y + dy), int(x + dx), :] = color
    return image


def image_add_mask_boundary(image: np.ndarray, mask: Tensor, color=(1., 0, 0), kernel_size=7):
    mask = F.interpolate(mask[None, None, :, :].float(), size=image.shape[:2], mode='nearest')
    assert kernel_size % 2 == 1
    boundary = F.avg_pool2d(mask, kernel_size, 1, kernel_size // 2)
    boundary = torch.logical_and(boundary.ne(mask), mask).cpu().numpy()[0, 0]
    image = image.copy()
    image[boundary, :3] = color
    return image


def show_all_levels(image, tree: Union[Tree3D, Tree3Dv2, Tree2D], tri_id=None, dpi=None, width=5., alpha=0.3, **kwargs):
    is_tree_2d = type(tree).__name__ == 'Tree2D'  # isinstance(tree, Tree2D)
    if is_tree_2d:
        levels = tree.get_levels()
        aux_data = None
    else:
        aux_data = tree.get_aux_data(tri_id)
        levels = tree.get_levels(aux_data)
    if isinstance(image, Tensor):
        image = image.cpu().numpy()
    # if image.dtype == np.uint8:
    #     image = image.astype(np.float32) / 255.
    num_level = len(levels) - 1
    if dpi is not None:
        plt.figure(dpi=dpi, **kwargs)
    else:
        if image is not None:
            height = image.shape[0] / image.shape[1] * width
        elif tri_id is not None:
            height = tri_id.shape[0] / tri_id.shape[1] * width
        else:
            height = width
        plt.figure(figsize=(2 * width, num_level * height), **kwargs)
    for level, nodes in enumerate(levels):
        if level == 0:
            continue
        # print(f"level [{level}]: {nodes.tolist()}")
        plt.subplot(num_level, 2, level * 2 - 1)
        if is_tree_2d:
            show_masks(image, tree.masks[nodes - 1], alpha=alpha)
        else:
            show_masks(image, [aux_data[i.item()][0] for i in nodes], alpha=alpha)
        plt.title(f"level={level}")
        plt.subplot(num_level, 2, level * 2)
        if is_tree_2d:
            show_masks(None, tree.masks[nodes - 1])
        else:
            show_masks(None, [aux_data[i.item()][0] for i in nodes])
        plt.title(f"{len(nodes)} masks")
    return num_level + 1


def get_hash_name(filepath: Path) -> str:
    # return hashlib.md5(str(filepath.absolute()).encode()).hexdigest()
    parts = filepath.parts
    return f"{parts[-3]}_{parts[-2]}_{filepath.stem}"


def search_folder(d: Path, extenstions=None):
    outputs = []
    for x in d.iterdir():
        if x.is_dir():
            outputs.extend(search_folder(x, extenstions))
        else:
            if extenstions is None:
                outputs.append(x)
            elif isinstance(extenstions, str):
                if x.suffix == extenstions:
                    outputs.append(x)
            elif x.suffix in extenstions:
                outputs.append(x)
    return outputs
