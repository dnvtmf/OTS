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

from semantic_sam import semantic_sam_l, semantic_sam_t
from segment_anything import build_sam
from tree_segmentation.extension import Mesh, utils, ops_3d
from tree_segmentation import Tree3Dv2, Tree3D, TreePredictor, render_mesh
from tree_segmentation.metric import TreeSegmentMetric

predictor = None


def predictor_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Preditor options')
    # model
    group.add_argument('-sam', '--segment-anything', action='store_true', default=False)
    group.add_argument('-ssl', '--semantic-sam-l', action='store_true', default=False, help='Default')
    group.add_argument('-sst', '--semantic-sam-t', action='store_true', default=False)
    group.add_argument(
        '-w', '--weights', default='./weights', help='The directory stored pretrained model of SAM/Semantic-SAM')
    # predictor
    group.add_argument('--pred_iou_thresh', default=0.88, type=float)
    group.add_argument('--stability_score_thresh', default=0.95, type=float)
    group.add_argument('--box_nms_thresh', default=0.7, type=float)
    group.add_argument('--points_per_batch', default=64, type=int)
    group.add_argument('--image-size', default=1024, type=int)
    # tree segmentation options
    group.add_argument('--max-steps', default=100, type=int)
    group.add_argument('--points_per_update', default=256, type=int)
    group.add_argument('--points_per_side', default=32, type=int)
    group.add_argument('--in_threshold', default=0.9, type=float)
    group.add_argument('--in_area_threshold', default=50, type=float)
    group.add_argument('--union_threshold', default=0.1, type=float)
    group.add_argument('--min_area', default=100, type=float)
    return group


def get_predictor(args, print=print):
    global predictor
    if predictor is not None:
        return predictor
    model_dir = Path(args.weights).expanduser()
    if args.segment_anything:
        assert model_dir.joinpath('sam_vit_h_4b8939.pth').exists(), f"Not model 'sam_vit_h_4b8939.pth' in {model_dir}"
        model = build_sam(model_dir.joinpath('sam_vit_h_4b8939.pth'))
        #save_root.joinpath('SAM')
        print('Loaded Model SAM')
    elif args.semantic_sam_t:
        assert model_dir.joinpath('swint_only_sam_many2many.pth').exists(), \
            f"Not model 'swint_only_sam_many2many.pth' in {model_dir}"
        model = semantic_sam_t(model_dir.joinpath('swint_only_sam_many2many.pth'))
        print('Loaded Model Semantic-SAM-t')
        # save_root.joinpath('Semantic-SAM-l')
    else:  # elif args.semantic_sam_l:
        assert model_dir.joinpath('swinl_only_sam_many2many.pth').exists(), \
            f"Not model 'swinl_only_sam_many2many.pth' in {model_dir}"
        model = semantic_sam_l(model_dir.joinpath('swinl_only_sam_many2many.pth'))
        print('Loaded Model Semantic-SAM-l')
        # save_root.joinpath('Semantic-SAM-t')
    model.eval()
    model = model.cuda()

    predictor = TreePredictor(
        model=model,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        image_size=args.image_size)
    predictor.generate_cfg = args
    print('build predictor')
    return predictor


def run_predictor(image: np.ndarray, device=torch.device('cuda')):
    assert predictor is not None
    results = predictor.generate(
        image,
        points_per_side=predictor.generate_cfg.points_per_side,
        points_per_update=predictor.generate_cfg.points_per_update,
        min_mask_region_area=predictor.generate_cfg.min_area,
        max_steps=predictor.generate_cfg.max_steps,
        in_threshold=predictor.generate_cfg.in_threshold,
        in_thre_area=predictor.generate_cfg.in_area_threshold,
        union_threshold=predictor.generate_cfg.union_threshold,
        device=device,
    )
    predictor.reset_image()
    return results
