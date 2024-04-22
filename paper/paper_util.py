from typing import Union
from pathlib import Path

import torch
import numpy as np
from torch import Tensor
from trimesh.exchange.export import export_mesh

from tree_segmentation import Tree2D, Tree3D
from tree_segmentation.extension import utils, Mesh


def get_cropped(image: Tensor, mask: Tensor):
    # print(utils.show_shape(image, mask))
    mask = mask.cpu().numpy() if isinstance(mask, Tensor) else mask
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    image = image.cpu().numpy() if isinstance(image, Tensor) else image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    masked_image = np.concatenate([image, mask], axis=-1)
    row = np.any(mask, axis=0)
    col = np.any(mask, axis=1)
    # print(row.shape, col.shape)
    row = np.nonzero(row)[0]
    col = np.nonzero(col)[0]
    # print(row, col)
    top, down = row[0], row[-1]
    left, right = col[0], col[-1]
    # print(left, right, top, down)
    masked_image = masked_image[left:right + 1, top:down + 1]
    return masked_image


def save_2d_masks(save_dir: Path, image: Union[np.ndarray, Tensor], tree2d: Tree2D):
    utils.dir_create_and_clear(save_dir, '*.png')
    for level, nodes in enumerate(tree2d.get_levels()):
        if level == 0:
            utils.save_image(save_dir.joinpath(f"0.png"), image)
            continue
        for x in nodes:
            x = x.item()
            mask = tree2d.masks[x - 1].to(torch.uint8) * 255
            # TODO: save part mask in the level
            utils.save_image(save_dir.joinpath(f"{x}_{tree2d.parent[x].item()}.png"), mask.cpu().numpy())
        print(f'Saved 2d level {level} with nodes: {nodes.tolist()}')


def get_2d_tree_from_3d(tree3d: Tree3D, tri_id: Tensor):
    aux_data = tree3d.get_aux_data(tri_id)
    levels = tree3d.get_levels(aux_data)
    masks = []
    for level in levels:
        for x in level:
            masks.append(aux_data[x.item()][0])
    masks = torch.stack(masks)
    tree2d = Tree2D(masks, torch.ones(masks.shape[0], device=tri_id.device), device=tri_id.device)
    tree2d.update_tree()
    tree2d.node_rearrange()
    tree2d.post_process()
    return tree2d


def save_3d_view(save_dir: Path, tree3d: Tree3D, image: Union[np.ndarray, Tensor], tri_id: Tensor):
    tree2d = get_2d_tree_from_3d(tree3d, tri_id)
    save_2d_masks(save_dir, image, tree2d)
    print(f'There are {tree2d.cnt} masks in the view of tree3d ')


def save_3d_part_meshs(save_dir: Path, tree3d: Tree3D, mesh: Mesh):
    utils.dir_create_and_clear(save_dir, '*.obj')
    tree3d.node_rearrange()
    for level, nodes in enumerate(tree3d.get_levels()):
        if level == 0:
            export_mesh(mesh.to_trimesh(), save_dir.joinpath('0.obj'))
            continue
        for x in nodes:
            x = x.item()
            part_mesh = Mesh(mesh.v_pos, mesh.f_pos[tree3d.masks[x - 1, 1:]])
            export_mesh(part_mesh.to_trimesh(), save_dir.joinpath(f'{x}_{tree3d.parent[x].item()}.obj'))
        print(f'Saved level {level} with nodes: {nodes.tolist()}')
