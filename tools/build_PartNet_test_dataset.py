import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import trimesh
import yaml
from rich.console import Console
from rich.tree import Tree
from scipy.spatial.distance import cdist
from torch import Tensor, nn
from tqdm import tqdm

import tree_segmentation.extension as ext
from tree_segmentation.extension import Mesh, ops_3d, utils
from tree_segmentation.render import render_mesh

torch.set_grad_enabled(False)
console = Console()
device = torch.device("cuda")
# device = torch.device("cpu")
# cache_root = Path('~/wan_code/segmentation/tree_segmentation/results').expanduser()
utils.set_printoptions(linewidth=120)

data_root = Path('~/data/PartNet/data_v0').expanduser()
print(f"Data Root: {data_root}")
shapes = list(os.scandir(data_root))
print(f'There are {len(shapes)} shapes')

ShapeNet_root = Path('~/data/ShapeNet/ShapeNetCore.v1/').expanduser()
print('ShapeNet root:', ShapeNet_root)
glctx = dr.RasterizeCudaContext()


def correct_face_orinent(mesh: Mesh, glctx, Tw2c, image_size=512):
    v_pos = mesh.v_pos.float() if Tw2c is None else ops_3d.xfm(mesh.v_pos.float(), Tw2c)
    v_pos = v_pos[None] if v_pos.ndim == 2 else v_pos
    rast, _ = dr.rasterize(glctx, v_pos, mesh.f_pos.int(), (image_size, image_size))

    unchanged = torch.ones(len(mesh.f_pos), dtype=torch.bool, device=device)
    for i in range(Tw2c.shape[0]):
        f_indices = torch.unique(rast[i, :, :, -1].int())
        if f_indices[0] == 0:
            f_indices = f_indices[1:]
        f_indices = f_indices - 1
        f_indices = f_indices[unchanged[f_indices]]
        if len(f_indices) == 0:
            continue
        unchanged[f_indices] = 0
        v0, v1, v2 = v_pos[i, mesh.f_pos[f_indices], :3].unbind(1)
        face_normals = torch.cross(v1 - v0, v2 - v0)
        f_indices = f_indices[face_normals[:, -1] < 0]  # need change faces
        if len(f_indices) == 0:
            continue
        # print(len(f_indices))
        mesh.f_pos[f_indices, :] = mesh.f_pos[f_indices, :].flip(-1)
        if mesh.f_tex is not None:
            mesh.f_tex[f_indices, :] = mesh.f_tex[f_indices, :].flip(-1)
        if mesh.f_nrm is not None:
            mesh.f_nrm[f_indices, :] = mesh.f_nrm[f_indices, :].flip(-1)
        # mesh.f_tng[f_indices, :] = mesh.f_tng[f_indices, :].flip(-1)
    print('Completed correct face')


def get_textrue_from_shapenet(filepath: Path, part: Mesh):
    #  Load Mesh
    # whole = Mesh.load(mesh_path_shapenet.joinpath('models/model_normalized.obj'))
    assert filepath.exists(), filepath
    whole = Mesh.load(filepath)
    whole = whole.to(device)
    whole.check()
    # Translate
    v_pos = part.v_pos.clone()
    v_pos[:, 0], v_pos[:, 2] = v_pos[:, 2], -v_pos[:, 0]
    mi, mx = v_pos.aminmax(dim=0)
    print(utils.show_shape(mi, mx), mi, mx)
    center = 0.5 * (mi + mx)
    scale = torch.sqrt(torch.sum((mx - mi)**2))
    T = torch.tensor([
        [0, 0, 1 / scale, -center[0] / scale],
        [0, 1 / scale, 0, -center[1] / scale],
        [-1 / scale, 0, 0, -center[2] / scale],
        [0, 0, 0, 1],
    ],
                     device=device).inverse()
    whole.v_pos = ops_3d.xfm(whole.v_pos, T)[:, :3]
    print(part, whole)

    part_ = part.to_trimesh()
    print(utils.show_shape(part_.vertices, part_.faces))
    whole_ = whole.to_trimesh()
    points_p = trimesh.sample.sample_surface(part_, 20000)[0]
    points_w = trimesh.sample.sample_surface(whole_, 20000)[0]

    dist_mat = cdist(points_p, points_w)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    print('CD:', chamfer_dist)
    assert chamfer_dist < 0.1
    #
    closest, distance, triangle_id = trimesh.proximity.closest_point(whole_, part.v_pos.cpu().numpy())
    closest = torch.from_numpy(closest).cuda().float()
    triangle_id = torch.from_numpy(triangle_id).cuda()
    print('Triangles_id:', utils.show_shape(triangle_id))
    assert 0 <= triangle_id.min() and triangle_id.max() < len(whole.f_pos)
    A, B, C = whole.v_pos[whole.f_pos[triangle_id]].unbind(1)
    print('Find closed points', distance.mean(), F.pairwise_distance(A, closest).mean())

    n = torch.cross(B - A, C - A)
    an = ops_3d.dot(n, n)
    alpha = ops_3d.dot(torch.cross(C - closest, C - B), n) / an
    beta = ops_3d.dot(torch.cross(A - closest, A - C), n) / an
    gamma = 1 - alpha - beta
    # check = alpha * A + beta * B + gamma * C
    # print((closest - check).square().max())
    # print((part.v_pos - check).abs().max())
    if whole.f_tex is not None:
        print('v_tex:', triangle_id.min(), triangle_id.max(), len(whole.f_tex))
        assert 0 <= triangle_id.min() and triangle_id.max() < len(whole.f_tex)
        v_tex = whole.v_tex[whole.f_tex[triangle_id]]
        # print(v_tex.shape)
        part.v_tex = v_tex[:, 0] * alpha + v_tex[:, 1] * beta + v_tex[:, 2] * gamma
        # print(part.v_tex.shape)
        part.f_tex = part.f_pos.clone()
        part.material = whole.material
        if whole.f_mat is not None:
            v_mat = whole.f_mat[triangle_id]
            print('f_mat:', part.f_pos[:, 0].min(), part.f_pos[:, 0].max(), len(v_mat))
            assert 0 <= part.f_pos[:, 0].min() and part.f_pos[:, 0].max() < len(v_mat)
            part.f_mat = v_mat[part.f_pos[:, 0]]
        # print(part.f_mat.shape, part.f_pos.shape, utils.show_shape(whole.f_mat, triangle_id))
    # if whole.f_nrm is not None:
    #     assert part.f_nrm is None
    #     part.f_nrm = whole.f_nrm
    print('Get texture from ShapeNet')
    return whole


@torch.no_grad()
def run_one(obj_dir: Path, save_dir: Path):
    with obj_dir.joinpath('meta.json').open('r') as f:
        meta = json.load(f)
    print('meta:', meta)
    mesh_path_shapenet = None
    for categroy in os.listdir(ShapeNet_root):
        if ShapeNet_root.joinpath(categroy, meta['model_id']).is_dir():
            mesh_path_shapenet = ShapeNet_root.joinpath(categroy, meta['model_id'])
            break
    print(f"Find corresponding shape in ShapeNet:", mesh_path_shapenet)
    assert mesh_path_shapenet is not None

    # Load Parts
    part_paths = sorted(list(obj_dir.joinpath('objs').glob('*.obj')))
    part_names = [part_path.stem for part_path in part_paths]
    print(part_names)
    part_meshes = [Mesh.load(part_path, mtl=False) for part_path in sorted(part_paths)]
    part = Mesh.merge(*part_meshes).cuda()
    # get texture
    get_textrue_from_shapenet(mesh_path_shapenet.joinpath('model.obj'), part)
    # camera pose
    image_size = 1024
    fovy = math.radians(60)
    num_views = 100
    radius = torch.rand((num_views,), device=device) * 0.1 + 2.5
    thetas = torch.arccos(torch.linspace(0, 1, num_views, device=device) * 2. - 1.)
    phis = torch.linspace(0, 1, num_views, device=device) * 2.0 * torch.pi
    # thetas = torch.tensor([90.], device=device).deg2rad_()
    # phis = torch.ones((num,), device=device) * 0.5 * torch.pi
    eye = ops_3d.coord_spherical_to(radius, thetas, phis).to(device)
    Tw2v = ops_3d.look_at(eye, torch.zeros_like(eye))
    Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=device)
    Tw2c = Tv2c @ Tw2v

    part.check()
    correct_face_orinent(part, glctx, Tw2c, image_size=image_size)
    part.check()
    part = part.unit_size()
    part.compuate_normals_()
    part.compute_tangents_()
    images, tri_ids = render_mesh(glctx, part, Tw2v=Tw2v, fovy=fovy, image_size=image_size, use_face_normal=True)
    print('Complet Rendering')
    save_dir.mkdir(exist_ok=True, parents=True)
    torch.save(Tw2v, save_dir.joinpath('Tw2v.pth'))

    images = images.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    for i in range(num_views):
        utils.save_image(save_dir.joinpath(f"{i:03d}.png"), images[i])
    console.print('[green]Complete save')


def main():
    save_dir = data_root.parent.joinpath('tree_seg')
    object_paths = list(data_root.glob('*'))
    for obj_path in tqdm(object_paths):
        console.print('-' * 40, obj_path, '-' * 40)
        try:
            run_one(obj_path, save_dir.joinpath(obj_path.name))
        except AssertionError as e:
            console.print(f'[red]AssertionError {e}')


if __name__ == '__main__':
    main()