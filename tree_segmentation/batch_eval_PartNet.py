import json
import math
import os
from pathlib import Path

import nvdiffrast.torch as dr
import torch
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.tree import Tree

from tree_segmentation.extension import Mesh
from tree_segmentation.extension import ops_3d
from semantic_sam import semantic_sam_l
from tree_segmentation import Tree3Dv2, Tree3D, TreePredictor
from tree_segmentation.metric import TreeSegmentMetric

torch.set_grad_enabled(False)
console = Console()
device = torch.device("cuda")
glctx = dr.RasterizeCudaContext()
# cache_root = Path('~/wan_code/segmentation/tree_segmentation/results').expanduser()
cache_root = Path('/data5/wan/TreeSeg_cache').expanduser()
data_root = Path('~/data/PartNet/data_v0').expanduser()
ckpt_path = Path("~/models/segmentation/Semantic-SAM/swinl_only_sam_many2many.pth").expanduser()
model = semantic_sam_l(ckpt_path).eval().to(device)
print(model)
predictor = TreePredictor(model)


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


@torch.no_grad()
def render_mesh(mesh, image_size=256, num=1, num_split=10):
    fovy = math.radians(60)
    Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=device)
    radius = torch.rand((num,), device=device) * 0.1 + 2.5
    thetas = torch.arccos(torch.rand((num,), device=device) * 2. - 1.)
    phis = torch.rand((num,), device=device) * 2.0 * torch.pi
    # thetas = torch.tensor([90.], device=device).deg2rad_()
    # phis = torch.ones((num,), device=device) * 0.5 * torch.pi
    eye = ops_3d.coord_spherical_to(radius, thetas, phis).to(device)
    Tw2v = ops_3d.look_at(eye, torch.zeros_like(eye))
    Tv2w = ops_3d.look_at(eye, torch.zeros_like(eye), inv=True)
    Tw2c = Tv2c @ Tw2v

    images = []
    tri_ids = []
    for s in range(0, num, num_split):
        e = min(s + num_split, num)
        v_pos = ops_3d.xfm(mesh.v_pos, Tw2c[s:e])
        # assert v_pos.ndim == 3
        rast, _ = dr.rasterize(glctx, v_pos, mesh.f_pos.int(), (image_size, image_size))
        view_pos = Tv2w[s:e, None, None, :3, 3]
        # nrm = ops_3d.compute_shading_normal(mesh,view_pos, rast, None)
        nrm = ops_3d.compute_shading_normal_face(mesh, view_pos, rast)
        view_dir = ops_3d.normalize(view_pos)
        light = ops_3d.normalize(view_dir + torch.randn(3, device=device) * 0.01)
        img = ops_3d.Blinn_Phong(
            nrm, light, view_dir, (nrm.new_full((3,), 0.2), nrm.new_full((3,), 0.5), nrm.new_full((3,), 0.1))
        ).clamp(0, 1)
        img = dr.antialias(img, rast, v_pos, mesh.f_pos.int())
        images.append(img)
        # tri_ids.append(part_map[rast[..., -1].long()])
        tri_ids.append(rast[..., -1].long())
    images, tri_ids = torch.cat(images, dim=0), torch.cat(tri_ids, dim=0)
    mask = tri_ids[..., None] > 0
    images = torch.where(mask, images, torch.ones_like(images))  # white backbground
    images = torch.cat([images, mask.to(images)], dim=-1)
    return images, tri_ids, Tw2v


@torch.no_grad()
def eval_one(example, metric: TreeSegmentMetric, num_views=100):
    print(f"Example dir", example)
    cache_dir = cache_root.joinpath(f"PartNet_{example.name}")
    cache_dir.mkdir(exist_ok=True)
    print('Cache Dir:', cache_dir)
    # load meta
    with example.joinpath('meta.json').open('r') as f:
        meta = json.load(f)
    print('meta:', meta)
    with example.joinpath('result_after_merging.json').open('r') as f:
        meta_parts = json.load(f)
    # print(list(os.scandir(example)))
    # Load Mesh
    part_paths = sorted(list(example.joinpath('objs').glob('*.obj')))
    part_names = [part_path.stem for part_path in part_paths]
    print(part_names)
    part_meshes = [Mesh.load(part_path, mtl=False) for part_path in sorted(part_paths)]
    print(f"There are {len(part_meshes)} parts", part_meshes[0])

    for part in part_meshes:
        assert 0 <= part.f_pos.min() and part.f_pos.max() < len(part.v_pos)
    mesh = Mesh.merge(*part_meshes)
    mesh = mesh.to(device).unit_size()

    part_map = torch.zeros(mesh.f_pos.shape[0] + 1, device=device, dtype=torch.int)
    num = 1
    for part_idx, part in enumerate(part_meshes, 1):
        part_map[num:num + part.f_pos.shape[0]] = part_idx
        num += part.f_pos.shape[0]
    # print(mesh)
    # print(part_map.shape, part_map.unique())
    del part_meshes

    # build gt Tree
    gt = Tree3D(mesh, device=device)
    show_tree = Tree(f"{meta_parts[0]['name']}_{meta_parts[0]['id']}")

    part_map = part_map.to(device)
    assert len(meta_parts) == 1
    get_ground_truth(meta_parts[0], gt, show_tree, part_names, part_map)
    gt.save(cache_dir.joinpath('gt.tree3d'))
    print('Save gt result')
    console.print(show_tree)
    gt.print_tree()
    del part_map
    # print(utils.get_GPU_memory())
    # torch.manual_seed(42)
    images, tri_ids, Tw2vs = render_mesh(mesh, image_size=1024, num=num_views)
    images, tri_ids, Tw2vs = images.cpu(), tri_ids.cpu(), Tw2vs.cpu()
    # print(utils.get_GPU_memory())
    # 2D Tree segmentation
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
        del tree_data, data
        # print(utils.get_GPU_memory())
    # 3D Tree segmentation
    tree3d = Tree3Dv2(mesh, cache_dir, device=device)
    tree3d.area.to(device)
    if cache_dir.joinpath('my.tree3dv2').exists():
        tree3d.load(cache_dir.joinpath('my.tree3dv2'))
    else:
        with torch.enable_grad():
            tree3d.run(20000)
        tree3d.save(cache_dir.joinpath('my.tree3dv2'))
    metric.update(tree3d, gt)
    return


def main():
    print(f"Data Root: {data_root}")
    shapes = list(os.scandir(data_root))
    print(f'There are {len(shapes)} shapes')
    torch.manual_seed(42)
    metric = TreeSegmentMetric()

    index = torch.randint(0, len(shapes), (10,))
    for i in index:
        eval_one(data_root.joinpath(shapes[i.item()].name), metric)
        print('PQ: ', metric.summarize())


if __name__ == '__main__':
    main()
