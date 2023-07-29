import json
import math
import os
from pathlib import Path
from typing import Optional

import nvdiffrast.torch as dr
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.tree import Tree
import torch_geometric as pyg

from semantic_sam import semantic_sam_l
from tree_segmentation.extension import Mesh
from tree_segmentation.extension import ops_3d
from tree_segmentation import Tree3Dv2, Tree3D, TreePredictor, render_mesh
from tree_segmentation.metric import TreeSegmentMetric

device = torch.device("cuda")
glctx = dr.RasterizeCudaContext()
ckpt_path = Path("~/models/segmentation/Semantic-SAM/swinl_only_sam_many2many.pth").expanduser()
predictor: Optional[TreePredictor] = None


def get_ground_truth(data, gt_tree: Tree3D, tree: Tree, part_names, part_map, node=0):
    added_leaf = np.zeros(len(part_names), dtype=bool)

    if 'children' in data:
        for child in data['children']:
            child_node = gt_tree.node_new()
            tree_i = tree.add(f"{child['name']}_{child['id']}")
            gt_tree.node_insert(child_node, node)
            added_leaf |= get_ground_truth(child, gt_tree, tree_i, part_names, part_map, child_node)

    for part_name in data['objs']:
        part_idx = part_names.index(part_name) + 1
        if added_leaf[part_idx - 1]:
            continue
        added_leaf[part_idx - 1] = True
        if 'children' in data or len(data['objs']) > 1:
            leaf = gt_tree.node_new()
            gt_tree.node_insert(leaf, node)
        else:
            leaf = node
        gt_tree.face_parent[part_map == part_idx] = leaf
        tree.add(f"[red]{part_name}")
    # remove chain
    if len(gt_tree.get_children(node)) == 1:
        if node == 0:
            gt_tree.node_delete(gt_tree.first[node].item(), move_children=Tree)
        else:
            gt_tree.node_delete(node, move_children=True)
    return added_leaf


def get_mesh_and_gt_tree(obj_dir: Path, cache_dir: Path, ):
    # load meta
    with obj_dir.joinpath('meta.json').open('r') as f:
        meta = json.load(f)
    print('meta:', meta)
    with obj_dir.joinpath('result_after_merging.json').open('r') as f:
        meta_parts = json.load(f)
    # print(list(os.scandir(example)))
    # Load Mesh
    part_paths = sorted(list(obj_dir.joinpath('objs').glob('*.obj')))
    part_names = [part_path.stem for part_path in part_paths]
    print(part_names)
    part_meshes = [Mesh.load(part_path, mtl=False) for part_path in sorted(part_paths)]
    print(f"There are {len(part_meshes)} parts", part_meshes[0])

    for part in part_meshes:
        assert 0 <= part.f_pos.min() and part.f_pos.max() < len(part.v_pos)
    mesh = Mesh.merge(*part_meshes)
    mesh = mesh.to(device).unit_size()
    torch.save(mesh, cache_dir.joinpath(obj_dir.name).with_suffix('.mesh_cache'))

    part_map = torch.zeros(mesh.f_pos.shape[0] + 1, device=device, dtype=torch.int)
    num = 1
    for part_idx, part in enumerate(part_meshes, 1):
        part_map[num:num + part.f_pos.shape[0]] = part_idx
        num += part.f_pos.shape[0]
    # print(mesh)
    # print(part_map.shape, part_map.unique())

    # build gt Tree
    gt = Tree3D(mesh, device=device)
    show_tree = Tree(f"{meta_parts[0]['name']}_{meta_parts[0]['id']}")

    part_map = part_map.to(device)
    assert len(meta_parts) == 1
    get_ground_truth(meta_parts[0], gt, show_tree, part_names, part_map)
    gt.save(cache_dir.joinpath('gt.tree3d'))
    gt_v2 = Tree3Dv2.convert(gt)
    gt_v2.save(cache_dir.joinpath('gt.tree3dv2'))
    print('Save gt result')
    # console.print(show_tree)
    # gt.print_tree()
    return mesh, gt_v2


def get_images(mesh: Mesh, image_size=256, num_views=1, num_split=10, seed=42, fovy=60):
    if seed > 0:
        torch.manual_seed(seed)
    fovy = math.radians(fovy)
    # Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=device)
    radius = torch.rand((num_views,), device=device) * 0.1 + 2.5
    thetas = torch.arccos(torch.rand((num_views,), device=device) * 2. - 1.)
    phis = torch.rand((num_views,), device=device) * 2.0 * torch.pi
    # thetas = torch.tensor([90.], device=device).deg2rad_()
    # phis = torch.ones((num,), device=device) * 0.5 * torch.pi
    eye = ops_3d.coord_spherical_to(radius, thetas, phis).to(device)
    Tw2vs = ops_3d.look_at(eye, torch.zeros_like(eye))
    # Tv2w = ops_3d.look_at(eye, torch.zeros_like(eye), inv=True)
    # Tw2c = Tv2c @ Tw2vs

    images = []
    tri_ids = []
    for s in range(0, num_views, num_split):
        e = min(s + num_split, num_views)
        images_i, tri_ids_i = render_mesh(glctx, mesh, Tw2v=Tw2vs[s:e], fovy=fovy, image_size=image_size)
        images.append(images_i)
        # tri_ids.append(part_map[rast[..., -1].long()])
        tri_ids.append(tri_ids_i)
    images, tri_ids = torch.cat(images, dim=0), torch.cat(tri_ids, dim=0)
    return images, tri_ids, Tw2vs


def run_2d_segmentation(cache_dir: Path, images: Tensor, tri_ids: Tensor, Tw2vs: Tensor):
    global predictor
    if predictor is None:
        predictor = TreePredictor(semantic_sam_l(ckpt_path).eval().to(device))
    num_views = images.shape[0]
    for index in tqdm(range(num_views)):
        if cache_dir.joinpath(f"view_{index:04d}.data").exists():
            continue
        torch.cuda.empty_cache()
        tree_data = predictor.generate(
            (images[index, :, :, :3].cpu().numpy() * 255).astype(np.uint8),
            max_iters=100,
            in_threshold=0.9,
            union_threshold=0.1,
            min_mask_region_area=100,
            points_per_update=256,
            device=device,
            in_thre_area=50,
        )
        data = {
            'tree_data': tree_data.save(filename=None),
            'tri_id': tri_ids[index].clone(),
            'image': images[index].clone(),
            'Tw2v': Tw2vs[index].clone(),
        }
        torch.save(data, cache_dir.joinpath(f"view_{index:04d}.data"))
    return


@torch.no_grad()
def eval_one(
    example,
    cache_root,
    metric: TreeSegmentMetric,
    num_views=100,
    force_2d=False,
    force_3d=False,
    force_gt=False
):
    print(f"Example dir", example)
    cache_dir = cache_root.joinpath(f"PartNet_{example.name}")
    cache_dir.mkdir(exist_ok=True)
    print('Cache Dir:', cache_dir)

    if 0 and cache_dir.joinpath('gt.tree3dv2').exists():
        mesh = torch.load(cache_dir.joinpath(example.stem + '.mesh_cache'), map_location=device)
        gt = Tree3Dv2(mesh, device=device)
        gt.load(cache_dir.joinpath('gt.tree3dv2'))
    else:
        mesh, gt = get_mesh_and_gt_tree(example, cache_dir)

    if force_2d or len(list(cache_dir.glob(f"view_*.data"))) < num_views:
        images, tri_ids, Tw2vs = get_images(mesh, image_size=1024, num_views=num_views, seed=42)
        run_2d_segmentation(cache_dir, images, tri_ids, Tw2vs)
    # print(utils.get_GPU_memory())
    # 3D Tree segmentation
    tree3d = Tree3Dv2(mesh, device=device)
    if not force_3d and cache_dir.joinpath('my.tree3dv2').exists():
        tree3d.load(cache_dir.joinpath('my.tree3dv2'))
    else:
        tree3d.load_2d_results(cache_dir)
        Gv = tree3d.build_view_graph()
        Gm = tree3d.build_graph(Gv)
        X, autoencoder = tree3d.compress_masks(epochs=3000)
        K = tree3d.Lmax * 2
        gnn = pyg.nn.GCN(
            in_channels=X.shape[1],
            hidden_channels=128,
            num_layers=2,
            out_channels=K,
            norm='BatchNorm'
        ).cuda()
        # print(gnn)
        tree3d.run(epochs=5000, K=K, gnn=gnn, A=Gm * Gm.ge(0.5), X=X)
        tree3d.save(cache_dir.joinpath('my.tree3dv2'))
    metric.update(tree3d, gt)
    return


console = None


def main():
    global console
    torch.set_grad_enabled(False)
    console = Console()
    # cache_root = Path('~/wan_code/segmentation/tree_segmentation/results').expanduser()
    cache_root = Path('/data5/wan/TreeSeg_cache').expanduser()
    data_root = Path('~/data/PartNet/data_v0').expanduser()

    print(f"Data Root: {data_root}")
    shapes = list(os.scandir(data_root))
    print(f'There are {len(shapes)} shapes')
    torch.manual_seed(42)
    metric = TreeSegmentMetric()

    index = torch.randint(0, len(shapes), (10,))
    for i in index:
        eval_one(data_root.joinpath(shapes[i.item()].name), cache_root, metric, force_3d=False)
        print('PQ: ', metric.summarize())


if __name__ == '__main__':
    main()
