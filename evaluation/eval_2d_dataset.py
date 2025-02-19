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
from evaluation.util import run_predictor, predictor_options

device = torch.device('cuda')
console = Console()


def process_one_image(args, image_path, save_root):
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
    results = run_predictor(args, image)
    results.save(save_root.joinpath(image_path.stem + '.tree2d'))


def process_images(args, image_paths: list, save_root: Path, name=''):
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
            process_one_image(args, image_path, save_root)
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

    return image_names


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
    predictor_options(parser)
    args = parser.parse_args()
    return args


def main():
    args = options()
    save_root = Path(args.output).expanduser()
    save_root.mkdir(exist_ok=True)
    console.print(f'[red]Try to save to: {save_root}')
    assert torch.cuda.is_available()

    if args.dataset == 'VOC':
        process_images(args, deal_VOC(), save_root, name='VOC')
    elif args.dataset == 'CelebAMask':
        process_images(args, deal_CelebAMask(), save_root, name='CelebAMask')
    elif args.dataset == 'Cityscapes':
        process_images(args, deal_Cityscapes(), save_root, name='Cityscapes')


if __name__ == '__main__':
    main()
