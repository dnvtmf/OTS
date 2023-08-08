import argparse
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

from tree_segmentation.extension import Mesh, utils, ops_3d
from tree_segmentation import Tree3Dv2, Tree3D, render_mesh
from tree_segmentation.metric import TreeSegmentMetric

from evaluation.util import run_predictor, predictor_options, get_predictor

device = torch.device("cuda")
glctx = dr.RasterizeCudaContext()


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


def get_mesh_and_gt_tree(obj_dir: Path, cache_dir: Path):
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
    # gt.save(cache_dir.joinpath('gt.tree3d'))
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
    return images.cpu(), tri_ids.cpu(), Tw2vs.cpu()


def run_2d_segmentation(cache_dir: Path, images: Tensor, tri_ids: Tensor, Tw2vs: Tensor):
    num_views = images.shape[0]
    for index in tqdm(range(num_views)):
        if cache_dir.joinpath(f"view_{index:04d}.data").exists():
            continue
        torch.cuda.empty_cache()
        tree_data = run_predictor((images[index, :, :, :3].cpu().numpy() * 255).astype(np.uint8))
        data = {
            'tree_data': tree_data.save(filename=None),
            'tri_id': tri_ids[index].clone(),
            # 'image': images[index].clone(),
            'Tw2v': Tw2vs[index].clone(),
        }
        torch.save(data, cache_dir.joinpath(f"view_{index:04d}.data"))
    return


@torch.no_grad()
def eval_one(
    args,
    example,
    cache_root,
    metric: TreeSegmentMetric,
    num_views=100,
    epochs_ea=3000,
    epochs_run=5000,
    force_2d=False,
    force_3d=False,
    use_gt_2d=False,
    print=print,
):
    print(f"Example dir", example)
    cache_dir = cache_root.joinpath(f"PartNet_{example.name}")
    cache_dir.mkdir(exist_ok=True)
    print('Cache Dir:', cache_dir)

    if 1 and cache_dir.joinpath('gt.tree3dv2').exists():
        mesh = torch.load(cache_dir.joinpath(example.stem + '.mesh_cache'), map_location=device)
        gt = Tree3Dv2(mesh, device=device)
        gt.load(cache_dir.joinpath('gt.tree3dv2'))
    else:
        mesh, gt = get_mesh_and_gt_tree(example, cache_dir)

    if len(gt.masks) == 0 or len(gt.masks) >= 200:
        return

    if use_gt_2d:
        images, tri_ids, Tw2vs = get_images(mesh, image_size=1024, num_views=num_views, seed=42)
    elif force_2d or len(list(cache_dir.glob(f"view_*.data"))) < num_views:
        images, tri_ids, Tw2vs = get_images(mesh, image_size=1024, num_views=num_views, seed=42)
        run_2d_segmentation(cache_dir, images, tri_ids, Tw2vs)
    # return
    # print(utils.get_GPU_memory())
    # 3D Tree segmentation
    save_path = cache_dir.joinpath('gt_seg.tree3dv2' if use_gt_2d else 'my.tree3dv2')
    tree3d = Tree3Dv2(mesh, device=device)
    if not force_3d and save_path.exists():
        tree3d.load(save_path)
    else:
        if use_gt_2d:
            if not tree3d.build_gt_segmentation(gt, tri_ids.cuda()):
                return
        else:
            tree3d.load_2d_results(cache_dir)
        gt.to(torch.device('cpu'))
        # Gv = tree3d.build_view_graph()
        # Gm = tree3d.build_graph(Gv)
        A = tree3d.build_all_graph()
        X, _ = tree3d.compress_masks(epochs=epochs_ea)
        K = int(tree3d.Lmax * args.K_ratio)
        gnn = pyg.nn.GCN(
            in_channels=X.shape[1], hidden_channels=128, num_layers=2, out_channels=K, norm='BatchNorm').cuda()
        # print(gnn)
        tree3d.run(
            epochs=epochs_run,
            K=K,
            gnn=gnn,
            A=A * A.ge(0.5),
            X=X,
            weights=args.loss_weights,
            print=print,
        )
        tree3d.save(save_path)
        print(f"save tree3d results to {save_path}")
    metric.update(tree3d, gt.to(device))
    return


def load_instance_segmentaion_gt_for_point_clouds(data_root: Path):
    import h5py
    ins_seg_root = data_root.joinpath('../ins_seg_h5').resolve()
    print(ins_seg_root)
    categories = os.listdir(ins_seg_root)
    print(len(categories))
    for cat in categories:
        test_json = ins_seg_root.joinpath(cat, 'test-00.json')
        with open(test_json, 'r') as f:
            test_info = json.load(f)
        print(cat, len(test_info))
        print(test_info[0])
        with h5py.File(ins_seg_root.joinpath(cat, 'test-00.h5'), 'r') as f:
            print(list(f.keys()))
            print(np.array(f['label']).shape)
            print(np.array(f['rgb']).shape)
            print(np.array(f['pts']).shape)
            print(np.array(f['opacity']).shape)
            print(np.array(f['nor']).shape)
        # print(test_info[0])
        break


console = None


def options():
    parser = argparse.ArgumentParser('3D Tree Segmentation for PartNet')
    parser.add_argument('-o', '--output', default='/data5/wan/PartNet', help='The directory to cache tree3d results')
    parser.add_argument('--data-root', default='~/data/PartNet/data_v0', help="The root path of PartNet dataset")
    parser.add_argument('-n', '--epochs', default=5000, type=int, help='The number of epochs when run tree3d')
    parser.add_argument('-ae', '--ae-epochs', default=3000, type=int, help='The number of epochs when run autoencoder')
    parser.add_argument('-v', '--num-views', default=100, type=int, help='The number of rendered views')
    parser.add_argument('-ns', '--num-shapes', default=10, type=int, help='The number of shapes per category')
    parser.add_argument('--seed', default=42, type=int, help='The seed to random choose evaluation shapes')
    # parser.add_argument('--print-interval', default=10, type=int, help='Print results every steps')
    parser.add_argument('--gt-2d', action='store_true', default=False, help='Use GT 2D segmentation results')
    parser.add_argument('-k', '--K-ratio', default=2, type=float, help='Set the default ')
    parser.add_argument('--log', default=None, help='The filepath for log file')
    parser.add_argument(
        '--force-3d',
        action='store_true',
        default=False,
        help='Force run 3d tree segment rather than use cached results')
    parser.add_argument(
        '--force-2d',
        action='store_true',
        default=False,
        help='Force run 2d tree segment rather than use cached results')
    utils.add_cfg_option(parser, '--loss-weights', default={}, help='The weigths of loss')

    predictor_options(parser)
    args = parser.parse_args()
    return args


def get_shapes(root: Path, num_max_per_shape=100, print=print):
    ins_seg_root = root.joinpath('../ins_seg_h5').resolve()
    print('Dataset split root:', ins_seg_root)
    # categories = os.listdir(ins_seg_root)
    categories = [
        # 'Bag',
        'Bed',
        # 'Bottle',
        # 'Bowl',
        'Chair',
        'Clock',
        'Dishwasher',
        'Display',
        'Door',
        'Earphone',
        'Faucet',
        # 'Hat',
        # 'Keyboard',
        'Knife',
        'Lamp',
        # 'Laptop',
        'Microwave',
        # 'Mug',
        'Refrigerator',
        # 'Scissors',
        'StorageFurniture',
        'Table',
        # 'TrashCan',
        # 'Vase',
    ]
    print('The number of categories:', len(categories))
    anno_ids = []
    for cat in categories:
        test_json = ins_seg_root.joinpath(cat, 'test-00.json')
        with open(test_json, 'r') as f:
            test_info = json.load(f)
        print(f'Category {cat} have {len(test_info)} shapes')
        anno_ids.append([x['anno_id'] for x in test_info])
    eval_ids = []
    for i in range(num_max_per_shape):
        for j in range(len(categories)):
            if i < len(anno_ids[j]):
                eval_ids.append(anno_ids[j][i])
    print(f"There are {len(eval_ids)} to evaluate")
    return eval_ids


@torch.no_grad()
def main():
    global console
    args = options()
    console = Console(record=bool(args.log))
    console.print(args)

    data_root = Path(args.data_root).expanduser()
    console.print(f"Data Root: {data_root}")

    cache_root = Path(args.output).expanduser()
    cache_root.mkdir(exist_ok=True)
    console.print(f"The middle results will cache in: {cache_root}")

    shapes = list(os.scandir(data_root))
    console.print(f'There are {len(shapes)} shapes')

    get_predictor(args, print=console.print)

    torch.manual_seed(args.seed)
    # index = torch.randint(0, len(shapes), (args.num_shapes,))
    # console.print(f"Evaluate {len(index)} shapes of PartNet")
    # shapes[i.item()].name
    metric = TreeSegmentMetric()
    shapes = get_shapes(data_root, num_max_per_shape=args.num_shapes, print=print)

    for shape in shapes:
        eval_one(
            args,
            data_root.joinpath(shape),
            cache_root=cache_root,
            metric=metric,
            num_views=args.num_views,
            epochs_ea=args.ae_epochs,
            epochs_run=args.epochs,
            force_3d=args.force_3d,
            force_2d=args.force_2d,
            use_gt_2d=args.gt_2d,
            print=console.print,
        )
        console.print(', '.join(f'{k}: {utils.float2str(v)}' for k, v in metric.summarize().items()))
    if args.log:
        console.save_text(args.log)


if __name__ == '__main__':
    main()
