import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch_geometric.nn as pyg_nn
import trimesh
import trimesh.proximity
from rich.console import Console
from torch import Tensor
from tqdm import tqdm

from evaluation.util import get_predictor, predictor_options, run_predictor
from tree_segmentation import (
    Tree3D, Tree3D, choose_best_views, random_camera_position, render_mesh,
)
from tree_segmentation.extension import Mesh, ops_3d, utils
from tree_segmentation.metric import TreeSegmentMetric

device = torch.device("cuda")
glctx = dr.RasterizeCudaContext()
console = None


def get_images_best_view(
    mesh: Mesh,
    image_size=1024,
    num_views=100,
    num_split=10,
    fovy=60,
    phi_range=(0, 360),
    theta_range=(30, 150),
    dist_to_surface=0.2,
    more_ratio=10,
):
    view_box = mesh.AABB
    view_box[0] += dist_to_surface
    view_box[1] -= dist_to_surface
    print('view bounding box', view_box, 'mesh box:', mesh.AABB)
    eye = random_camera_position(mesh, view_box, N=num_views * more_ratio, min_dist=dist_to_surface)
    N = eye.shape[0]
    print(eye.shape)

    phi_range = [math.radians(phi_range[0]), math.radians(phi_range[1])]
    ct_range = [math.cos(math.radians(theta_range[0])), math.cos(math.radians(theta_range[1]))]
    phis = torch.rand((N,), device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    thetas = torch.arccos(torch.rand((N,), device=device) * (ct_range[1] - ct_range[0]) + ct_range[0])
    radius = torch.ones((N,), device=device)
    at = ops_3d.coord_spherical_to(radius, thetas, phis).to(device)
    Tw2vs = ops_3d.look_at(eye, eye + at)
    print(utils.show_shape(Tw2vs))

    fovy = math.radians(fovy)
    Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=device)
    indices = choose_best_views(glctx, mesh, Tv2c @ Tw2vs, num_views, steps=num_views * 1000, image_size=image_size)
    assert len(indices) == num_views
    print(utils.show_shape(indices))
    Tw2vs = Tw2vs[indices]
    print('Tw2vs:', Tw2vs.shape)

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
        image = utils.load_image(image_dir.joinpath(f"{i:03d}.png"))
        image = torch.from_numpy(image.copy()).float() / 255.
        images.append(image)
    images = torch.stack(images)  # NxHxWx3
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


def load_mesh_and_gt(data_dir: Path, cache_dir: Path, force=False):
    mesh = Mesh.load(data_dir.joinpath('simplify.ply'))
    mesh.float()

    if not force and cache_dir.joinpath('gt.tree3dv2').exists():
        gt = Tree3D(mesh=mesh, device=device)
        gt.load(cache_dir.joinpath('gt.tree3dv2'))
        print('Load GT from', cache_dir.joinpath('gt.tree3dv2'))
    else:
        full_mesh = Mesh.load(data_dir.joinpath('mesh.ply')).to_trimesh()
        semantic_mesh_path = data_dir.joinpath('habitat/mesh_semantic.ply')
        assert semantic_mesh_path.exists()
        semantic_data = utils.parse_ply(semantic_mesh_path)
        # print(utils.show_shape(semantic_data))
        assert semantic_data[1]['name'] == 'face' and semantic_data[1]['names'][1] == 'object_id'
        semantic_label = semantic_data[1]['data'][1]  # type: np.ndarray
        semantic_label = np.repeat(
            semantic_label[:, None], semantic_data[1]['data'][0].shape[1] - 2, axis=1).reshape(-1)
        # # print(semantic_label[:10])
        # print('semantic_label', utils.show_shape(semantic_label))

        print(utils.show_shape(full_mesh.vertices, full_mesh.faces))
        v_pos = mesh.v_pos[mesh.f_pos].mean(1).cpu().numpy().reshape(-1, 3)  # center of triangles
        closest, dist, tri_id = trimesh.proximity.closest_point(full_mesh, v_pos)
        # print('closest_point', utils.show_shape(tri_id))
        masks = []
        for i in np.unique(semantic_label):
            obj_i = semantic_label == i
            mask_i = obj_i[tri_id]
            if mask_i.any():
                masks.append(mask_i)

        gt = Tree3D(mesh, device=device)
        gt.masks = torch.zeros((len(masks), gt.num_faces + 1), device=device, dtype=torch.bool)
        gt.scores = torch.ones((len(masks),), device=device)
        gt.resize(len(masks) + 1)
        for i in range(len(masks)):
            gt.node_insert(gt.node_new(), 0)
            gt.masks[i, 1:] = torch.from_numpy(masks[i]).bool().to(device)
        gt.save(cache_dir.joinpath('gt.tree3dv2'))
    return mesh, gt


def load_semantic_gt(data_dir: Path, mesh: Mesh, cache_dir: Path):
    print('Load gt')
    with data_dir.joinpath('semantic.bin').open('rb') as f:
        labels = np.array(list(struct.iter_unpack('@Q', f.read())), dtype=np.uint64).astype(np.int32)
        print(labels.shape, labels.dtype)
        assert 0 <= labels.min() and labels.max() < mesh.f_pos.shape[0] // 2
    with data_dir.joinpath('semantic.json').open('r') as f:
        semantic_info = json.load(f)
    print(list(semantic_info.keys()))
    children = []
    parent = []
    names = []
    numPrimitives = []
    for obj in semantic_info['segmentation']:
        idx = obj['id']
        while len(children) <= idx:
            children.append([])
            parent.append(-1)
            names.append([''])
            numPrimitives.append(0)
        parent[idx] = -1 if obj['parent'] == '' else int(obj['parent'])
        children[idx] = [int(c) for c in obj['children']]
        names[idx] = obj['class']
        numPrimitives[idx] = obj['numPrimitives']
        if len(obj['children']) > 0:
            assert obj['numPrimitives'] == 0
    numPrimitives = np.array(numPrimitives)
    print(numPrimitives.shape, numPrimitives.dtype)
    prefix_sum = np.cumsum(numPrimitives) - numPrimitives
    print(prefix_sum[:10], numPrimitives[:10])
    for i in range(len(children)):
        for c in children[i]:
            assert parent[c] == i

    gt = Tree3D(mesh, device=device)
    gt.resize(len(children) + 1)

    def _make_tree(
        p,
        tree_idx,
    ):
        for c in children[p]:
            ci = gt.node_new()
            gt.node_insert(ci, tree_idx)
            _make_tree(c, ci)
        if len(children[p]) == 0:
            s = prefix_sum[p]
            e = s + numPrimitives[p]
            for i in range(s, e):
                gt.face_parent[labels[i] * 2 + 1] = tree_idx
                gt.face_parent[labels[i] * 2 + 2] = tree_idx

    for i in range(len(parent)):
        if parent[i] == -1:
            idx = gt.node_new()
            gt.node_insert(idx, 0)
            _make_tree(i, idx)
    # console.print(show_tree)
    # gt.print_tree()
    # gt.save(cache_dir.joinpath('gt2.tree3d'))
    gt_v2 = Tree3D.convert(gt)
    gt_v2.save(cache_dir.joinpath('gt2.tree3dv2'))
    return gt_v2


def options():
    parser = argparse.ArgumentParser('3D Tree Segmentation for PartNet')
    parser.add_argument('-o', '--output', default='./results/Replica', help='The directory to cache tree3d results')
    parser.add_argument('--data-root', default='~/data/Replica', help="The root path of Replica dataset")
    parser.add_argument('-s', '--scene', default='room_1', help='The scene to evaluate')
    parser.add_argument('--seed', default=42, type=int, help='The seed to random choose evaluation shapes')
    # parser.add_argument('--print-interval', default=10, type=int, help='Print results every steps')
    parser.add_argument('--log', default='log', help='The filepath for log file')
    # rendering options
    parser.add_argument('-v', '--num-views', default=100, type=int, help='The number of rendered views')
    # parser.add_argument('--num-faces', default=500_000, help='Try to simplify the number of faces for the mesh')
    utils.add_bool_option(parser, '--force-view', help='Force run view generateiong rather than use cached')
    parser.add_argument('--fovy', default=90, type=float)
    utils.add_n_tuple_option(parser, '--thetas', default=[30, 150])
    utils.add_n_tuple_option(parser, '--phis', default=[0, 360])
    parser.add_argument('--dist-to-bound', default=0.2, type=float)
    ## Tree3D options
    parser.add_argument('-n', '--epochs', default=5000, type=int, help='The number of epochs when run tree3d')
    parser.add_argument('-ae', '--ae-epochs', default=3000, type=int, help='The number of epochs when run autoencoder')
    parser.add_argument('-k', '--K-ratio', default=2, type=float, help='Set the default ')
    parser.add_argument('--gnn', default='GCN', help='The type of gnn')
    parser.add_argument('--gnn-layers', default=2, type=int, help='The number of layers of gnn')
    parser.add_argument('--gnn-hidden-dim', default=128, type=int, help='The hidden dimension of gnn')
    utils.add_cfg_option(parser, '--loss-weights', default={}, help='The weigths of loss')
    parser.add_argument('-f', '--filename', default=None, help='The filename of tree3d results')
    utils.add_bool_option(parser, '--force-3d', help='Force run 3d tree segment rather than use cached')
    ## Tree2D options
    predictor_options(parser)
    utils.add_bool_option(parser, '--force-2d', help='Force run 2d tree segment rather than use cached')
    utils.add_bool_option(parser, '--gt-2d', help='Use ground truth 2D segment')
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
    data_dir = data_root.joinpath(args.scene)
    assert data_dir.is_dir()
    console.print(f"Data Dir: {data_dir}")

    cache_dir = Path(args.output).expanduser().joinpath(args.scene)
    cache_dir.mkdir(exist_ok=True, parents=True)
    console.print(f"The results will save in: {cache_dir}")

    ## load predictor
    get_predictor(args, print=console.print)

    ## load mesh
    mesh, gt = load_mesh_and_gt(data_dir, cache_dir)
    mesh = mesh.to(device).unit_size()
    roate_angle = {
        'office_0': 5,
        'office_1': 34,
        'office_2': 66.5,
        'office_3': 52,
        'office_4': 7,
        'room_0': 0,
        'room_1': 26,
        'room_2': -10,
    }[args.scene]
    mesh.v_pos = ops_3d.xfm(mesh.v_pos, ops_3d.rotate_z(-math.radians(roate_angle), device))[:, :3].contiguous()
    mesh.v_pos = mesh.v_pos[:, (0, 2, 1)].contiguous()
    mesh = mesh.unit_size()
    console.print(mesh)
    console.print(mesh.AABB)
    console.print('[Mesh] GPU {0:.4f}/{1:.4f}'.format(*utils.get_GPU_memory()))

    # generate images
    image_dir = cache_dir.joinpath('images')
    image_dir.mkdir(exist_ok=True)
    run_2d = args.force_2d or len(list(cache_dir.glob(f"view_*.data"))) < args.num_views
    if args.force_view or run_2d or args.gt_2d:
        if args.force_view or len(list(image_dir.glob('*.png'))) < args.num_views:
            images, tri_ids, Tw2vs = get_images_best_view(mesh, image_size=args.image_size, num_views=args.num_views)

            torch.save(Tw2vs, image_dir.joinpath('Tw2v.pth'))
            filename_fmt = '{:0%dd}.png' % len(str(args.num_views))
            for i in range(args.num_views):
                utils.save_image(image_dir.joinpath(filename_fmt.format(i)), images[i])
            console.print('Save all images in:', image_dir)
            run_2d = True
        else:
            images, tri_ids, Tw2vs = load_images(mesh, image_dir, num_views=args.num_views)
        console.print('[Image] GPU {0:.4f}/{1:.4f}'.format(*utils.get_GPU_memory()))

        print(mesh.f_pos.shape, tri_ids.max())
        # 2D Segmentation
        if run_2d and not args.gt_2d:
            run_2d_segmentation(cache_dir, images, tri_ids, Tw2vs)
            console.print('[2D] GPU {0:.4f}/{1:.4f}'.format(*utils.get_GPU_memory()))
        del images, Tw2vs

    # 3D Tree segmentation
    if args.filename is None:
        save_path = cache_dir.joinpath('my.tree3dv2')
    else:
        save_path = cache_dir.joinpath(args.filename).with_suffix('.tree3dv2')
    tree3d = Tree3D(mesh, device=device)
    if not args.force_3d and save_path.exists():
        tree3d.load(save_path)
    else:
        if args.gt_2d:
            tree3d.load_2d_results(gt=gt, tri_ids=tri_ids, pack=True)
        else:
            tree3d.load_2d_results(cache_dir, pack=True)
        # save memory
        # view_mask = tree3d.masks_view.any(dim=0)
        # view_mask[0] = 1
        # print(view_mask.shape, view_mask.sum() / view_mask.shape[0])
        # tree3d.area = tree3d.area[view_mask[1:]]
        # tree3d.masks_view = tree3d.masks_view[:, view_mask]
        # tree3d.masks_2d = tree3d.masks_2d[:, view_mask]
        # print(utils.show_shape(tree3d.area, tree3d.masks_view, tree3d.masks_2d))
        # tree3d.num_faces = view_mask.sum() - 1
        # if tree3d._masks_2d_packed:
        #     indices = torch.nonzero(tree3d.masks_2d)
        #     tree3d._masks_2d_sp = torch.sparse.FloatTensor(
        #         torch.stack([tree3d.masks_2d[indices[:, 0], indices[:, 1]] - 1, indices[:, 1]]),
        #         torch.ones(indices.shape[0], device=indices.device),
        #         [tree3d.M, tree3d.num_faces + 1],
        #     )

        # Gv = tree3d.build_view_graph()
        # Gm = tree3d.build_graph(Gv)
        A_file = cache_dir.joinpath('A.pth')
        if A_file.exists() and not run_2d:
            A = torch.load(A_file, map_location=device)
            console.print('Load A from:', A_file)
        else:
            A = tree3d.build_all_graph()
            torch.save(A, A_file)
            console.print('Save A to:', A_file)
        X_file = cache_dir.joinpath('X.pth')
        if X_file.exists() and not run_2d:
            X = torch.load(X_file, map_location=device)
            console.print('Load X from:', X_file)
        else:
            X, _ = tree3d.compress_masks(epochs=args.ae_epochs)
            torch.save(X, X_file)
            console.print('Save X to:', X_file)
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
            epochs=args.epochs,
            K=K,
            gnn=gnn,
            A=A,
            X=X,
            weights=args.loss_weights,
            print=print,
        )

        # masks = tree3d.masks
        # tree3d.masks = torch.zeros((masks.shape[0], mesh.f_pos.shape[0] + 1), dtype=torch.bool, device=masks.device)
        # tree3d.masks[:, view_mask] = masks

        tree3d.save(save_path)
        console.print(f"save tree3d results to {save_path}")
        del A, X
    metric = TreeSegmentMetric()
    gt.mesh = tree3d.mesh
    gt.area = tree3d.area
    metric.update(tree3d, gt.to(device))
    console.print(', '.join(f"{k}={v}" for k, v in metric.summarize().items()))
    if args.log:
        now_date = time.strftime("%m-%d_%H:%M:%S", time.localtime(time.time()))
        console.save_text(cache_dir.joinpath(f"{args.log}_{now_date}.txt").as_posix())


if __name__ == '__main__':
    main()
