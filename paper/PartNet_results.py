import json
from pathlib import Path
import math
import numpy as np
from tqdm import tqdm
import json
import nvdiffrast.torch as dr
import torch
from typing import List

from tree_segmentation.extension import utils, ops_3d
from tree_segmentation import Tree3Dv2, TreeSegmentMetric, Tree2D

from paper.paper_util import get_2d_tree_from_3d
from multiprocessing import Process, Queue, set_start_method

data_root = Path('~/data/PartNet/data_v0').expanduser()


def to_tuple(m: TreeSegmentMetric):
    return (m.SQ_sum, m.RQ_sum, m.PQ_sum, m.TS_sum, m.TQ_sum, m.maxIoU_sum, m.cnt)


def add_from_tuple(m: TreeSegmentMetric, data):
    m.SQ_sum += data[0]
    m.RQ_sum += data[1]
    m.PQ_sum += data[2]
    m.TS_sum += data[3]
    m.TQ_sum += data[4]
    m.maxIoU_sum += data[5]
    m.cnt += data[6]


def eval_one(que: Queue, result_paths: List[Path], gpu_id):
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{gpu_id}")
    # device = torch.device("cpu")
    glctx = dr.RasterizeCudaContext(f"cuda:{gpu_id}")
    torch.set_default_device(device)

    for result_path in result_paths:
        obj_id = result_path.parts[-2]
        # print(result_path, obj_id)
        assert data_root.joinpath(obj_id).exists()
        with data_root.joinpath(obj_id, 'meta.json').open('r') as f:
            meta = json.load(f)
            cat = meta['model_cat']

        m2d = TreeSegmentMetric()
        m3d = TreeSegmentMetric()
        mp = TreeSegmentMetric()
        g2d = TreeSegmentMetric()

        mesh = torch.load(result_path.with_name(f'{result_path.parts[-2]}.mesh_cache'), map_location=device)
        mesh = mesh.to(device)
        gt = Tree3Dv2(mesh, device=device)
        gt.load(result_path.with_name('gt.tree3dv2'))
        prediction = Tree3Dv2(mesh, device=device)
        prediction.load(result_path)
        m3d.update(prediction, gt)

        Tw2vs = torch.load(result_path.parent.joinpath("images", "Tw2v.pth"), map_location=device)
        num_views = len(Tw2vs)
        image_size = (512, 512)
        Tv2c = ops_3d.perspective(fovy=math.radians(60), size=image_size, device=device)
        Tw2c = Tv2c @ Tw2vs.to(device)
        v_pos = ops_3d.xfm(mesh.v_pos, Tw2c)
        rast, _ = dr.rasterize(glctx, v_pos.to(device), mesh.f_pos.int().to(device), image_size)
        tri_ids = rast[..., -1].int().to(device)

        face_mask = torch.zeros(mesh.f_pos.shape[0] + 1, dtype=torch.bool, device=device)
        face_mask[torch.unique(tri_ids)] = 1
        gt.face_mask = face_mask
        # print(utils.show_shape(tri_ids))
        for view_id in range(num_views):
            tree2d_path = result_path.parent.joinpath(f"view_{view_id:04d}.data")  # type: Path
            if not tree2d_path.is_file():
                print(f"file {tree2d_path} is not exists")
                continue
            tree2d_gt = get_2d_tree_from_3d(gt, tri_ids[view_id])

            tree2d_pd = Tree2D(device=device)
            pth = torch.load(tree2d_path, map_location=device)
            # print(utils.show_shape(pth))
            tree2d_pd.load(None, **pth['tree_data'])
            tree2d_pd.remove_background(tri_ids[view_id].eq(0))
            tree2d_pd.post_process()
            tree2d_pd.remove_not_in_tree()
            assert tree2d_pd.parent.lt(0).any()
            assert tree2d_gt.parent.lt(0).any()
            m2d.update(tree2d_pd, tree2d_gt)

            tree2d_p = get_2d_tree_from_3d(prediction, tri_ids[view_id])
            mp.update(tree2d_p, tree2d_gt)

        gt_seg_path = result_path.with_name('gt_seg.tree3dv2')
        if gt_seg_path.exists():
            gt_seg = Tree3Dv2(mesh, device=device)
            gt_seg.load(gt_seg_path)
            g2d.update(gt_seg, gt)

        que.put((cat, to_tuple(m2d), to_tuple(m3d), to_tuple(mp), to_tuple(g2d)))


def main():
    utils.set_printoptions(linewidth=120)

    # cache_root = Path('~/wan_code/segmentation/tree_segmentation/results').expanduser()
    print(f"Data Root: {data_root}")

    save_root = Path('/data5/wan/PartNet_final/').expanduser()
    print(f"Save Root:", save_root)
    all_results = sorted(list(save_root.glob('*/my.tree3dv2')))
    print(f'There are {len(all_results)} results')

    categories = set()
    results = {}

    for result_path in all_results:
        obj_id = result_path.parts[-2]
        # print(result_path, obj_id)
        assert data_root.joinpath(obj_id).exists()
        with data_root.joinpath(obj_id, 'meta.json').open('r') as f:
            meta = json.load(f)
            cat = meta['model_cat']
        categories.add(cat)
        if cat not in results:
            results[cat] = []
        results[cat].append(result_path)

    categories = sorted(list(categories))
    metrics_2d = {'all': TreeSegmentMetric()}
    metrics_3d = {'all': TreeSegmentMetric()}
    metrics_p = {'all': TreeSegmentMetric()}
    metrics_gs = {'all': TreeSegmentMetric()}

    print(f"Categories: {len(categories)}")
    for cat in categories:
        print(f'Cat: {cat} have {len(results[cat])} results')
        metrics_2d[cat] = TreeSegmentMetric()
        metrics_3d[cat] = TreeSegmentMetric()
        metrics_p[cat] = TreeSegmentMetric()
        metrics_gs[cat] = TreeSegmentMetric()

    # show_cats = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone', 'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator', 'Scissors', 'StorageFurniture', 'Table', 'TrashCan', 'Vase']
    show_cats = [
        'Bed', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone', 'Faucet', 'Knife', 'Lamp', 'Microwave',
        'Refrigerator', 'StorageFurniture', 'Table', 'Vase'
    ]
    metric_names = ['SQ', 'RQ', 'TS', 'TQ', 'mTQ']
    # for step, result_path in enumerate(tqdm(all_results), 1):
    process_list = []
    que = Queue()
    num_gpus = 10
    N = len(all_results)
    for i in range(num_gpus):
        p = Process(target=eval_one, args=(que, all_results[i * N // num_gpus:(i + 1) * N // num_gpus], i))
        p.start()
        process_list.append(p)
    for step in tqdm(range(N)):
        cat, m2d_data, m3d_data, mp_data, gs_data = que.get()
        add_from_tuple(metrics_2d['all'], m2d_data)
        add_from_tuple(metrics_2d[cat], m2d_data)
        add_from_tuple(metrics_3d['all'], m3d_data)
        add_from_tuple(metrics_3d[cat], m3d_data)
        add_from_tuple(metrics_p['all'], mp_data)
        add_from_tuple(metrics_p[cat], mp_data)
        add_from_tuple(metrics_gs['all'], gs_data)
        add_from_tuple(metrics_gs[cat], gs_data)

        print(f'Complete {step+1}/{len(all_results)}')
        print(f'3D, {", ".join([f"{k}={v:.4f}" for k, v in metrics_3d["all"].summarize().items()])}')
        print(f'2D, {", ".join([f"{k}={v:.4f}" for k, v in metrics_2d["all"].summarize().items()])}')
        print(f'P , {", ".join([f"{k}={v:.4f}" for k, v in metrics_p["all"].summarize().items()])}')
        print(f'GT, {", ".join([f"{k}={v:.4f}" for k, v in metrics_gs["all"].summarize().items()])}')
        print('num:', metrics_3d['all'].cnt, metrics_2d['all'].cnt, metrics_p['all'].cnt, metrics_gs['all'].cnt)
    [p.join() for p in process_list]
    with open(save_root.joinpath('metrics.csv'), 'w') as f:
        f.write(f"type, cat, {', '.join(metrics_3d['all'].summarize().keys())}\n")
        for k, v in metrics_3d.items():
            f.write(f'3D, {k}, {", ".join([f"{x:.4f}" for x in v.summarize().values()])}\n')
        for k, v in metrics_2d.items():
            f.write(f'2D, {k}, {", ".join([f"{x:.4f}" for x in v.summarize().values()])}\n')
        for k, v in metrics_p.items():
            f.write(f'P , {k}, {", ".join([f"{x:.4f}" for x in v.summarize().values()])}\n')


if __name__ == '__main__':
    set_start_method('spawn')
    main()