from pathlib import Path
import argparse
import sys

import cv2
import torch
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

sys.path.insert(0, Path(__file__).resolve().absolute().parent.parent.as_posix())

from tree_segmentation.extension import utils
from segment_anything import build_sam
from semantic_sam import semantic_sam_l, semantic_sam_t
from tree_segmentation.predictor import TreePredictor

device = torch.device('cuda')
console = Console()


def process_one_image(image_path, predictor, save_root):
    print('=' * 20, image_path, '=' * 20)
    if save_root.joinpath(image_path.stem + '.tree2d').exists():
        return
    torch.cuda.empty_cache()
    image = utils.load_image(image_path)
    H, W = image.shape[:2]
    if H > 1024 or W > 1024:
        scale = min(1024 / H, 1024 / W)
        H, W = int(H * scale), int(W * scale)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    print(image.shape)
    results = predictor.generate(
        image,
        points_per_side=predictor.generate_cfg.points_per_side,
        points_per_update=predictor.generate_cfg.points_per_update,
        min_mask_region_area=predictor.generate_cfg.min_area,
        max_iters=predictor.generate_cfg.max_steps,
        in_threshold=predictor.generate_cfg.in_threshold,
        in_thre_area=predictor.generate_cfg.in_area_threshold,
        union_threshold=predictor.generate_cfg.union_threshold,
        device=device,
    )
    results.save(save_root.joinpath(image_path.stem + '.tree2d'))
    predictor.reset_image()


def process_images(predictor: TreePredictor, image_paths: list, save_root: Path, name=''):
    image_paths = [p for p in image_paths if not save_root.joinpath(p.stem + '.tree2d').exists()]
    save_root.mkdir(exist_ok=True)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(compact=True),
        TextColumn("-->"),
        TimeElapsedColumn(),
        console=console,
        speed_estimate_period=3600,
    )
    with progress:
        task_id = progress.add_task(name, total=len(image_paths))
        for image_path in image_paths:
            process_one_image(image_path, predictor, save_root)
            progress.advance(task_id)


def deal_VOC():
    voc_part_root = Path('/data/VOC/Part2010/Annotations_Part')
    voc_img_root = Path('/data/VOC/VOC2012/JPEGImages')
    voc_ppart_root = Path('/data/VOC/panopticParts/labels')
    voc_semantic_seg_root = Path('/data/VOC/VOC2012/SegmentationClassAug')
    voc_instance_seg_root = Path('/data/VOC/VOC2012/SegmentationObject')

    voc_part_annotations = sorted(list(voc_part_root.glob('*.mat')))
    panoptic_parts_annotations = sorted(list(voc_ppart_root.rglob('*.tif')))

    image_names = [filename.stem for filename in voc_part_annotations]
    image_names += [filename.stem for filename in panoptic_parts_annotations]
    image_names = set(image_names)
    print(f'There are {len(image_names)} image for PASCAL VOC')

    return [voc_img_root.joinpath(name + '.jpg') for name in image_names]


def deal_CelebAMask():
    face_dataset_root = Path('/data6/CelebAMask-HQ')
    face_img_root = face_dataset_root.joinpath('CelebA-HQ-img')
    face_part_root = face_dataset_root.joinpath('CelebAMask-HQ-mask-anno')

    image_names = sorted([image_path for image_path in face_img_root.glob('*.jpg')])
    print(f'There are {len(image_names)} image for CelebAMask-HQ')

    return image_names,


def deal_Cityscapes():
    cityscapes_ppart_root = Path('/data5/Cityscapes/gtFinePanopticParts')
    cityscapes_image_root = Path('/data5/Cityscapes/images')
    city_annotations = sorted(list(cityscapes_ppart_root.rglob('*.tif')))
    city_image_paths = {path.stem: path for path in cityscapes_image_root.rglob('*.png')}

    images_paths = [city_image_paths[ann_path.stem[:-len('_gtFinePanopticParts')]] for ann_path in city_annotations]
    print(f'There are {len(images_paths)} image for Cityscapes')
    return images_paths


def options():
    parser = argparse.ArgumentParser('Tree Segmentation for Images')
    parser.add_argument('-sam', '--segment-anything', action='store_true', default=False)
    parser.add_argument('-ssl', '--semantic-sam-l', action='store_true', default=False, help='Default')
    parser.add_argument('-sst', '--semantic-sam-t', action='store_true', default=False)
    parser.add_argument(
        '-w', '--weights', default='./weights', help='The directory stored pretrained model of SAM/Semantic-SAM')
    parser.add_argument('-o', '--output', default='.', help='The directory of output')
    parser.add_argument(
        '-d',
        '--dataset',
        default='VOC',
        choices=['VOC', 'CelebAMask', 'Cityscapes'],
        help="The evaluated dataset of [VOC, CelebAMask, Cityscapes]")
    # preditor
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


def main():
    args = options()
    save_root = Path(args.output).expanduser()
    save_root.mkdir(exist_ok=True)
    console.print(f'[red]Try to save to: {save_root}')

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

    if args.dataset == 'VOC':
        process_images(predictor, deal_VOC(), save_root, name='VOC')
    elif args.dataset == 'CelebAMask':
        process_images(predictor, deal_CelebAMask(), save_root, name='CelebAMask')
    elif args.dataset == 'Cityscapes':
        process_images(predictor, deal_Cityscapes(), save_root, name='Cityscapes')


if __name__ == '__main__':
    main()
