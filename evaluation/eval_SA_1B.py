import json
import math
import os
from pathlib import Path
import argparse

import hashlib
import matplotlib.pyplot as plt
import networkx as nx
import nvdiffrast.torch as dr
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import yaml
import cv2
from rich.console import Console
from rich.tree import Tree
import torch_geometric as pyg

import tree_segmentation as ts
import tree_segmentation.extension as ext
from tree_segmentation.extension import ops_3d, Mesh, utils
from semantic_sam import SemanticSAM, semantic_sam_l, semantic_sam_t
from segment_anything import build_sam
from tree_segmentation import TreePredictor, TreeSegmentMetric, Tree2D, MaskData
from tree_segmentation.util import show_masks, show_all_levels, get_hash_name
from evaluation.batch_eval_PartNet import get_mesh_and_gt_tree, get_images
import pycocotools.mask as mask_util
from rich.console import Console


def read_annotations(json_path: Path):
    masks = []
    scores = []
    with open(json_path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    for ann in annotations:
        scores.append(ann['predicted_iou'])
        masks.append(mask_util.decode(ann['segmentation']))
    scores = np.stack(scores)
    masks = np.stack(masks)
    tree2d = Tree2D(MaskData(masks=torch.from_numpy(masks), iou_preds=torch.from_numpy(scores)))
    tree2d.update_tree()
    return tree2d


def options():
    parser = argparse.ArgumentParser('Tree Segmentation for SA-1B dataset')
    parser.add_argument('-sam', '--segment-anything', action='store_true', default=False)
    parser.add_argument('-ssl', '--semantic-sam-l', action='store_true', default=False, help='Default')
    parser.add_argument('-sst', '--semantic-sam-t', action='store_true', default=False)
    parser.add_argument(
        '-w', '--weights', default='./weights', help='The directory stored pretrained model of SAM/Semantic-SAM')
    parser.add_argument('-o', '--output', default=None, help='The directory to cache tree2d results')
    parser.add_argument('--data-root', default='/data5/SA-1B', help="The root path of SA-1B dataset")
    parser.add_argument('--seed', default=42, type=int, help='The seed to random choose evaluation images')
    parser.add_argument('-n', '--number', default=1000, type=int, help='The number of images to evaluate')
    parser.add_argument('--eval-part', default=110, type=int, help='The part index of SA-1B to evaluate')
    parser.add_argument('--log', default='log.txt', help='The filepath for log file')
    parser.add_argument('--print-interval', default=10, type=int, help='Print results every steps')
    # predictor
    parser.add_argument('--pred_iou_thresh', default=0.88, type=float)
    parser.add_argument('--stability_score_thresh', default=0.95, type=float)
    parser.add_argument('--box_nms_thresh', default=0.7, type=float)
    parser.add_argument('--points_per_batch', default=64, type=int)
    parser.add_argument('--image-size', default=1024, type=int)
    # tree segmentation options
    parser.add_argument('--max-steps', default=100, type=int)
    parser.add_argument('--points_per_update', default=256, type=int)
    parser.add_argument('--points_per_side', default=32, type=int)
    parser.add_argument('--in_threshold', default=0.9, type=float)
    parser.add_argument('--in_area_threshold', default=50, type=float)
    parser.add_argument('--union_threshold', default=0.1, type=float)
    parser.add_argument('--min_area', default=100, type=float)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    console = Console(record=True)
    args = options()
    console.print(args)

    if args.output is not None:
        save_root = Path(args.output).expanduser()
        save_root.mkdir(exist_ok=True, parents=True)
        console.print(f'[red]Try to save 2D Tree Segmentation Results in: {save_root}')
    else:
        save_root = None

    data_root = Path(args.data_root).joinpath(f"{args.eval_part:06d}").expanduser()
    images_paths = sorted(list(data_root.glob('*.jpg')))
    console.print(f'There are {len(images_paths)} images in dir: {data_root}')
    np.random.seed(42)
    eval_image_paths = np.random.choice(images_paths, args.number)
    print(f"Try To evaluate {len(eval_image_paths)} image")

    model_dir = Path(args.weights).expanduser()
    if args.segment_anything:
        assert model_dir.joinpath('sam_vit_h_4b8939.pth').exists(), f"Not model 'sam_vit_h_4b8939.pth' in {model_dir}"
        model = build_sam(model_dir.joinpath('sam_vit_h_4b8939.pth'))
        #save_root.joinpath('SAM')
        console.print('Loaded Model SAM')
    elif args.semantic_sam_t:
        assert model_dir.joinpath('swint_only_sam_many2many.pth').exists(), \
            f"Not model 'swint_only_sam_many2many.pth' in {model_dir}"
        model = semantic_sam_t(model_dir.joinpath('swint_only_sam_many2many.pth'))
        console.print('Loaded Model Semantic-SAM-t')
        # save_root.joinpath('Semantic-SAM-l')
    else:  # elif args.semantic_sam_l:
        assert model_dir.joinpath('swinl_only_sam_many2many.pth').exists(), \
            f"Not model 'swinl_only_sam_many2many.pth' in {model_dir}"
        model = semantic_sam_l(model_dir.joinpath('swinl_only_sam_many2many.pth'))
        console.print('Loaded Model Semantic-SAM-l')
        # save_root.joinpath('Semantic-SAM-t')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval()
    model = model.to(device)

    predictor = TreePredictor(
        model=model,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        image_size=args.image_size)
    predictor.generate_cfg = args
    console.print('build predictor')

    metric = TreeSegmentMetric()
    timer = utils.TimeEstimator(args.number)
    time_avg = utils.TimeWatcher()
    timer.start()
    time_avg.start()
    for i, image_path in enumerate(eval_image_paths, 1):
        image = utils.load_image(image_path)
        H, W, _ = image.shape
        scale = min(1024 / H, 1024 / W)
        image = cv2.resize(image, (int(scale * W), int(scale * H)), interpolation=cv2.INTER_AREA)
        time_avg.log('image')
        gt = read_annotations(image_path.with_suffix('.json'))
        time_avg.log('gt')
        if save_root is not None:
            save_path = save_root.joinpath(image_path.name).with_suffix('.tree2d')
            if save_path.exists():
                prediction = Tree2D(device=device)
                prediction.load(save_path)
            else:
                prediction = predictor.generate(image, device=device)
                prediction.save(save_path)
        else:
            prediction = predictor.generate(image, device=device)
        time_avg.log('tree2d')
        metric.update(prediction, gt.to(device), return_match=False)
        time_avg.log('metric')
        timer.step()
        if i % args.print_interval == 0:
            console.print(f'Process [{i}/{args.number}], {timer.progress}')
            console.print(', '.join(f'{k}: {utils.float2str(v)}' for k, v in metric.summarize().items()))

    console.print('Complete Evalution')
    console.print('Time:', time_avg)
    for k, v in metric.summarize().items():
        console.print(f"{k:5s}: {v}")

    console.save_text(args.log)


if __name__ == '__main__':
    main()
