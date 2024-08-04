import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from segment_anything import build_sam, build_sam_vit_b, build_sam_vit_l
from semantic_sam import semantic_sam_l, semantic_sam_t
from segment_anything_fast import build_sam_fast_vit_l, build_sam_fast_vit_b, build_sam_fast_vit_h
from tree_segmentation import TreePredictor

predictor: Optional[TreePredictor] = None


def predictor_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Preditor options')
    # model
    group.add_argument('-sam', '--segment-anything', action='store_true', default=False)
    group.add_argument('-samH', '--segment-anything-h', action='store_true', default=False)
    group.add_argument('-samL', '--segment-anything-l', action='store_true', default=False)
    group.add_argument('-samB', '--segment-anything-b', action='store_true', default=False)
    group.add_argument('--sam-fast', action='store_true', default=False,
        help='using segment anything fast')

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
    group.add_argument('--explore-ratio', default=0.5, type=float, help='The ratio to explore unsegment part')
    # tree segmentation options
    group.add_argument('--max-steps', default=100, type=int)
    group.add_argument('--points-per-update', default=256, type=int)
    group.add_argument('--points-per-side', default=32, type=int)
    group.add_argument('--in-threshold', default=0.8, type=float)
    group.add_argument('--in-area-threshold', default=50, type=float)
    group.add_argument('--union-threshold', default=0.1, type=float)
    group.add_argument('--min-area', default=100, type=float)
    return group


def get_predictor(args=None, print=print, device=torch.device('cuda')):
    global predictor
    if predictor is not None:
        return predictor
    assert args is not None
    model_dir = Path(args.weights).expanduser()
    if args.segment_anything or args.segment_anything_h:
        assert model_dir.joinpath('sam_vit_h_4b8939.pth').exists(), f"Not model 'sam_vit_h_4b8939.pth' in {model_dir}"
        if not args.sam_fast:
            model = build_sam(model_dir.joinpath('sam_vit_h_4b8939.pth'))
            print('Loaded Origin Model SAM')
        else:
            model = build_sam_fast_vit_h(model_dir.joinpath('sam_vit_h_4b8939.pth'))
            print('Loaded Fast Model SAM')
    elif args.segment_anything_l:
        assert model_dir.joinpath('sam_vit_l_0b3195.pth').exists(), f"Not model 'sam_vit_l_0b3195.pth' in {model_dir}"

        if not args.sam_fast:
            model = build_sam_vit_l(model_dir.joinpath('sam_vit_l_0b3195.pth'))
            # save_root.joinpath('SAM')
            print('Loaded Origin Model SAM-L')
        else:
            model = build_sam_fast_vit_l(model_dir.joinpath('sam_vit_l_0b3195.pth'))
            # save_root.joinpath('SAM')
            print('Loaded Fast Model SAM-L')
    elif args.segment_anything_b:
        assert model_dir.joinpath('sam_vit_b_01ec64.pth').exists(), f"Not model 'sam_vit_b_01ec64.pth' in {model_dir}"
        if not args.sam_fast:
            model = build_sam_vit_b(model_dir.joinpath('sam_vit_b_01ec64.pth'))
            # save_root.joinpath('SAM')
            print('Loaded Origin Model SAM-B')
        else:
            model = build_sam_fast_vit_b(model_dir.joinpath('sam_vit_b_01ec64.pth'))
            # save_root.joinpath('SAM')
            print('Loaded Fast Model SAM-L')
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
    model = model.to(device)

    predictor = TreePredictor(
        model=model,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        # image_size=args.image_size
    )
    predictor.generate_cfg = args
    print('build predictor')
    return predictor


def run_predictor(image: np.ndarray, device=torch.device('cuda'), compress=True):
    assert predictor is not None
    cfg = predictor.generate_cfg  # noqa
    results = predictor.tree_generate(
        image,
        points_per_side=cfg.points_per_side,
        points_per_update=cfg.points_per_update,
        min_mask_region_area=cfg.min_area,
        max_steps=cfg.max_steps,
        in_threshold=cfg.in_threshold,
        in_thre_area=cfg.in_area_threshold,
        union_threshold=cfg.union_threshold,
        ratio=cfg.explore_ratio,
        device=device,
        compress=compress,
    )
    predictor.reset_image()
    return results
