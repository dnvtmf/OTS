import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import nvdiffrast.torch as dr
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.tree import Tree
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

from tree_segmentation.extension import Mesh, utils, ops_3d
from tree_segmentation import Tree3Dv2, Tree3D, render_mesh, TreePredictor, Tree2D, choose_best_views
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


def get_images_best_view(mesh: Mesh, image_size=1024, num_views=100, num_split=10, seed=0, fovy=60, more_ratio=10):
    if seed > 0:
        torch.manual_seed(seed)
    fovy = math.radians(fovy)
    max_views = int(num_views * more_ratio)
    radius = torch.rand((max_views,), device=device) * 0.1 + 2.5
    thetas = torch.arccos(torch.rand((max_views,), device=device) * 2. - 1.)
    phis = torch.rand((max_views,), device=device) * 2.0 * torch.pi
    eye = ops_3d.coord_spherical_to(radius, thetas, phis).to(device)
    Tw2vs = ops_3d.look_at(eye, torch.zeros_like(eye))
    Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=device)
    Tw2c = Tv2c @ Tw2vs

    best_indices = choose_best_views(glctx, mesh, Tw2c, num_views, image_size, num_split)
    Tw2vs = Tw2vs[best_indices]

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


def get_images(mesh: Mesh, image_size=256, num_views=1, num_split=10, seed=0, fovy=60):
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


def load_images(mesh: Mesh, image_dir: Path, num_views=1, fovy=60):
    Tw2vs = torch.load(image_dir.joinpath("Tw2v.pth"), map_location='cpu')[:num_views]
    images = []
    for i in range(num_views):
        # if colored:
        image = utils.load_image(image_dir.joinpath(f"{i:03d}.png"))  # type: np.ndarray
        # else:
        #     image = utils.load_image(image_dir.joinpath(f"{i:03d}_g.png"))  # type: np.ndarray
        image = torch.from_numpy(image.copy()).float() / 255.
        images.append(image)
    images = torch.stack(images)  #NxHxWx3
    image_size = images.shape[2]
    Tv2c = ops_3d.perspective(fovy=math.radians(fovy), size=(image_size, image_size), device=device)
    Tw2c = Tv2c @ Tw2vs.to(device)
    v_pos = ops_3d.xfm(mesh.v_pos, Tw2c)
    rast, _ = dr.rasterize(glctx, v_pos, mesh.f_pos.int(), (image_size, image_size))
    return images, rast[..., -1].int(), Tw2vs


def run_2d_segmentation(cache_dir: Path, images: Tensor, tri_ids: Tensor, Tw2vs: Tensor):
    num_views = images.shape[0]
    for index in tqdm(range(num_views), desc='Tree2D'):
        if cache_dir.joinpath(f"view_{index:04d}.data").exists():
            continue
        torch.cuda.empty_cache()
        tree_data = run_predictor((images[index, :, :, :3].cpu().numpy() * 255).astype(np.uint8), device=device)
        data = {
            'tree_data': tree_data.save(filename=None),
            'tri_id': tri_ids[index].clone(),
            # 'image': images[index].clone(),
            'Tw2v': Tw2vs[index].clone(),
        }
        torch.save(data, cache_dir.joinpath(f"view_{index:04d}.data"))
        del data
        del tree_data
    return


def build_view_graph(area, tri_ids: Tensor, threshold=0.5, num_nearest=5):
    N_view = tri_ids.shape[0]
    view_masks = torch.zeros((N_view, area.shape[0] + 1), device=device)
    for i in range(N_view):
        v_faces = tri_ids[i].unique()
        if v_faces[0] == 0:
            v_faces = v_faces[1:]
        view_masks[i, v_faces] = 1
    # print(area.shape)
    A = F.linear(view_masks[:, 1:], view_masks[:, 1:] * area)
    mask_area = torch.mv(view_masks[:, 1:], area)
    # A = A / (area[:, None] + area[None, :] - A).clamp_min(1e-7)
    A = A / mask_area[:, None]
    indices = torch.topk(A, num_nearest + 1, dim=0)[1]
    # print(utils.show_shape(indices))
    A = A.ge(threshold)
    A[torch.arange(N_view), indices] = 1
    return A


def run_fast_2d_semgentation(
    cache_dir: Path,
    mesh: Mesh,
    images: Tensor,
    tri_ids: Tensor,
    Tw2vs: Tensor,
    num_points=10,
    steps=10,
    predictor: TreePredictor = None,
):
    if predictor is None:
        predictor = get_predictor()
    v3 = mesh.v_pos[mesh.f_pos]  # shape: (F, 3, 3)
    area = torch.cross(v3[:, 0] - v3[:, 1], v3[:, 0] - v3[:, 2], dim=-1).norm(dim=-1) * 0.5
    A = build_view_graph(area, tri_ids)
    features = []
    results = []
    N = len(images)
    images = (images[:, :, :, :3] * 255).to(torch.uint8).cpu().numpy()
    _, H, W, _ = images.shape
    # first stage
    for i in range(N):
        predictor.set_image(images[i])
        features.append(utils.tensor_to(predictor.features, device=torch.device('cpu')))
        tree2d = Tree2D(
            # device=device,
            min_area=predictor.generate_cfg.min_area,
            in_threshold=predictor.generate_cfg.in_threshold,
            in_thres_area=predictor.generate_cfg.in_area_threshold,
            union_threshold=predictor.generate_cfg.union_threshold,
        )
        points = tree2d.sample_grid(predictor.generate_cfg.points_per_side)
        output = predictor.process_points(points)
        num_ignored = tree2d.insert_batch(output)
        tree2d.remove_not_in_tree()
        tree2d.remove_background(tri_ids[i].eq(0))
        tree2d.compress()
        # predictor.reset_image()
        results.append(tree2d)
    print('complete stage1')
    # second stage
    for step in range(steps):
        # predictor.set_image(images[0])
        for i in range(N):
            points = []
            for j in range(N):
                if i == j or A[i, j] <= 0:
                    continue
                results[j]
                masks = results[j].masks
                for l, x in enumerate(results[j].get_levels()):
                    if l == 0:
                        continue
                    temp = torch.zeros(area.shape[0] + 1, dtype=torch.int, device=device)
                    for k in x:
                        temp[torch.unique(tri_ids[j][masks[k - 1]].to(device))] = k
                    temp[0] = 0
                    mask = temp[tri_ids[i]]
                    for k in x:
                        mask_k = torch.nonzero(mask == k)
                        if len(mask_k) > 0:
                            points.append(mask_k[torch.randint(0, len(mask_k), (num_points,))])
            points = torch.cat(points, dim=0).flip(1).cpu().numpy() / np.array([W, H])
            predictor.features = utils.tensor_to(features[i], device=device)
            results[i].insert_batch(predictor.process_points(points))
            tree2d.remove_background(tri_ids[i].eq(0))
            results[i].remove_not_in_tree()
            results[i].compress()
        print(f'complete stage2 step={step}')
    # save
    for index in range(N):
        data = {
            'tree_data': results[i].save(filename=None),
            'tri_id': tri_ids[index].clone(),
            # 'image': images[index].clone(),
            'Tw2v': Tw2vs[index].clone(),
        }
        torch.save(data, cache_dir.joinpath(f"view_{index:04d}.data"))
    print('complete save reults')
    return


@torch.no_grad()
def eval_one(args,
             shape_root: Path,
             cache_root: Path,
             metric: TreeSegmentMetric,
             num_views=100,
             epochs_ea=3000,
             epochs_run=5000,
             print=print):
    print(f"Shape Dir", shape_root)
    cache_dir = cache_root.joinpath(f"{shape_root.name}")
    cache_dir.mkdir(exist_ok=True)
    print('Cache Dir:', cache_dir)
    image_dir = cache_dir.joinpath('images')
    image_dir.mkdir(exist_ok=Tree)

    if 1 and cache_dir.joinpath('gt.tree3dv2').exists():
        mesh = torch.load(cache_dir.joinpath(shape_root.stem + '.mesh_cache'), map_location=device)
        gt = Tree3Dv2(mesh, device=device)
        gt.load(cache_dir.joinpath('gt.tree3dv2'))
    else:
        mesh, gt = get_mesh_and_gt_tree(shape_root, cache_dir)
    gt = gt.to('cpu')
    # if len(gt.masks) == 0 or len(gt.masks) >= 200:
    #     return
    print("Mesh:", mesh)
    print('GPU {0:.4f}/{1:.4f}'.format(*utils.get_GPU_memory()))

    run_2d = args.force_2d or len(list(cache_dir.glob(f"view_*.data"))) < num_views
    if args.force_view or args.gt_2d or run_2d:
        if args.force_view or len(list(image_dir.glob('*.png'))) < num_views:
            if args.random_views:
                images, tri_ids, Tw2vs = get_images(mesh, image_size=args.image_size, num_views=num_views)
            else:
                images, tri_ids, Tw2vs = get_images_best_view(mesh, image_size=args.image_size, num_views=num_views)
            torch.save(Tw2vs, image_dir.joinpath('Tw2v.pth'))
            filename_fmt = '{:0%dd}.png' % len(str(num_views))
            for i in range(num_views):
                utils.save_image(image_dir.joinpath(filename_fmt.format(i)), images[i])
            print('Save all images in:', image_dir)
            run_2d = True
        else:
            images, tri_ids, Tw2vs = load_images(mesh, image_dir, num_views=num_views)
        print('[Image] GPU {0:.4f}/{1:.4f}'.format(*utils.get_GPU_memory()))
        if run_2d:
            run_2d_segmentation(cache_dir, images, tri_ids, Tw2vs)
            print('[2D] GPU {0:.4f}/{1:.4f}'.format(*utils.get_GPU_memory()))
        del images, Tw2vs
    else:
        tri_ids = None

    # 3D Tree segmentation
    if args.filename is None:
        save_path = cache_dir.joinpath('gt_seg.tree3dv2' if args.gt_2d else 'my.tree3dv2')
    else:
        save_path = cache_dir.joinpath(args.filename).with_suffix('.tree3dv2')
    tree3d = Tree3Dv2(mesh, device=device)
    if not args.force_3d and save_path.exists():
        tree3d.load(save_path)
    else:
        if args.gt_2d:
            if not tree3d.build_gt_segmentation(gt.to(device), tri_ids.cuda()):
                return
            gt.to(torch.device('cpu'))
        else:
            tree3d.load_2d_results(cache_dir)
        # Gv = tree3d.build_view_graph()
        # Gm = tree3d.build_graph(Gv)
        A = tree3d.build_all_graph()
        X_file = cache_dir.joinpath('X_gt.pth' if args.gt_2d else 'X.pth')
        if X_file.exists() and not run_2d:
            X = torch.load(X_file, map_location=device)
            print('Load X from:', X_file)
        else:
            X, _ = tree3d.compress_masks(epochs=epochs_ea)
            torch.save(X, X_file)
        K = int(tree3d.Lmax * args.K_ratio)
        gnn_type = args.gnn.upper()
        if gnn_type == 'NONE':
            gnn = None
        else:
            gnn_m = {
                'GCN': pyg_nn.GCN,
                'CHEB': pyg_nn.ChebConv,
                'GRAPH': pyg_nn.GraphConv,
                'LG': pyg_nn.LGConv,
                'FA': pyg_nn.FAConv,
                'GCN2': pyg_nn.GCN2Conv,
                'LE': pyg_nn.LEConv,
                'SSG': pyg_nn.SSGConv,
                'SG': pyg_nn.SGConv,
            }[gnn_type]
            gnn = gnn_m(
                in_channels=X.shape[1],
                hidden_channels=args.gnn_hidden_dim,
                num_layers=args.gnn_layers,
                out_channels=K,
                norm='BatchNorm',
            ).cuda()
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
        del A, X
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
    parser.add_argument('-s', '--split', default='', help='The split to test')
    parser.add_argument('-v', '--num-views', default=100, type=int, help='The number of rendered views')
    parser.add_argument('-ns', '--num-shapes', default=-1, type=int, help='The number of shapes to test')
    parser.add_argument('--start-index', default=0, type=int, help='The start index to test')
    parser.add_argument('--seed', default=42, type=int, help='The seed to random choose evaluation shapes')
    # parser.add_argument('--print-interval', default=10, type=int, help='Print results every steps')
    parser.add_argument('--log', default=None, help='The filepath for log file')
    ## Tree3D options
    parser.add_argument('-n', '--epochs', default=5000, type=int, help='The number of epochs when run tree3d')
    parser.add_argument('-ae', '--ae-epochs', default=3000, type=int, help='The number of epochs when run autoencoder')
    parser.add_argument('-k', '--K-ratio', default=2, type=float, help='Set the default ')
    parser.add_argument('--gnn', default='GCN', help='The type of gnn')
    parser.add_argument('--gnn-layers', default=2, type=int, help='The number of layers of gnn')
    parser.add_argument('--gnn-hidden-dim', default=128, type=int, help='The hidden dimension of gnn')
    parser.add_argument('-f', '--filename', default=None, help='The filename of tree3d results')
    ##
    utils.add_bool_option(parser, '--gt-2d', help='Use GT 2D segmentation results')
    utils.add_bool_option(parser, '--force-3d', help='Force run 3d tree segment rather than use cached')
    utils.add_bool_option(parser, '--force-2d', help='Force run 2d tree segment rather than use cached')
    utils.add_bool_option(parser, '--force-view', help='Force run view generateiong rather than use cached')
    utils.add_bool_option(parser, '--random-views', default=False, help='Random generate views')
    # utils.add_bool_option(parser, '--colored', default=False, help='The colored images')

    utils.add_cfg_option(parser, '--loss-weights', default={}, help='The weigths of loss')
    predictor_options(parser)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    global console
    args = options()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    console = Console(record=bool(args.log))
    console.print(args)

    data_root = Path(args.data_root).expanduser()
    console.print(f"Data Root: {data_root}")

    cache_root = Path(args.output).expanduser()
    cache_root.mkdir(exist_ok=True)
    console.print(f"The middle results will cache in: {cache_root}")

    if args.split == '' or not os.path.exists(args.split):
        shapes = sorted(os.listdir(Path(args.split).expanduser()))
    else:
        with open(args.split, 'r') as f:
            shapes = [line.strip() for line in f.readlines() if line.strip()]
    console.print(f'[green]There are {len(shapes)} shapes, test {args.num_shapes} shapes')
    assert 0 <= args.start_index < len(shapes)
    shapes = shapes[args.start_index:]
    if args.num_shapes > 0:
        shapes = shapes[:args.num_shapes]
    get_predictor(args, print=console.print)

    metric = TreeSegmentMetric()
    for shape in tqdm(shapes, desc='Tree3D'):
        try:
            eval_one(
                args,
                data_root.joinpath(shape),
                cache_root=cache_root,
                metric=metric,
                num_views=args.num_views,
                epochs_ea=args.ae_epochs,
                epochs_run=args.epochs,
                print=console.print,
            )
        except Exception as e:
            console.print(f"[red] ERROR: {str(e)}")
            torch.cuda.empty_cache()

        console.print(', '.join(f'{k}: {utils.float2str(v)}' for k, v in metric.summarize().items()))
    if args.log:
        console.save_text(cache_root.joinpath(args.log))


if __name__ == '__main__':
    main()
