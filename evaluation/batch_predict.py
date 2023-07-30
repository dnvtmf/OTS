from pathlib import Path

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

from tree_segmentation.extension import utils
from segment_anything import build_sam_vit_h
from tree_segmentation.predictor import TreePredictor

device = torch.device('cuda')


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
        max_iters=100,
        in_threshold=0.9,
        union_threshold=0.1,
        min_mask_region_area=100,
        points_per_update=256,
        device=device,
        in_thre_area=50,
    )
    results.save(save_root.joinpath(image_path.stem + '.tree2d'))
    predictor.reset_image()


def process_images(predictor: TreePredictor, image_paths: list, save_root: Path, name=''):
    image_paths = [p for p in image_paths if not save_root.joinpath(p.stem + '.tree2d').exists()]
    save_root.mkdir(exist_ok=True)
    console = Console()
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

    save_root = Path('../segment_anything').joinpath('results/VOC')
    return [voc_img_root.joinpath(name + '.jpg') for name in image_names], save_root


def deal_CelebAMask():
    face_dataset_root = Path('/data6/CelebAMask-HQ')
    face_img_root = face_dataset_root.joinpath('CelebA-HQ-img')
    face_part_root = face_dataset_root.joinpath('CelebAMask-HQ-mask-anno')

    image_names = sorted([image_path for image_path in face_img_root.glob('*.jpg')])
    print(f'There are {len(image_names)} image for CelebAMask-HQ')

    save_root = Path('../segment_anything').joinpath('results/CelebAMask-HQ')
    return image_names, save_root


def deal_Cityscapes():
    cityscapes_ppart_root = Path('/data5/Cityscapes/gtFinePanopticParts')
    cityscapes_image_root = Path('/data5/Cityscapes/images')
    city_annotations = sorted(list(cityscapes_ppart_root.rglob('*.tif')))
    city_image_paths = {path.stem: path for path in cityscapes_image_root.rglob('*.png')}

    images_paths = [city_image_paths[ann_path.stem[:-len('_gtFinePanopticParts')]] for ann_path in city_annotations]
    print(f'There are {len(images_paths)} image for Cityscapes')

    save_root = Path('../segment_anything').joinpath('results/Cityscapes')
    return images_paths, save_root


def load_predictor():
    # load SAM
    assert torch.cuda.is_available()
    sam_path = Path('~/models/sam_vit_h_4b8939.pth').expanduser()
    sam = build_sam_vit_h(sam_path).cuda()
    # predictor = SamPredictor(sam)
    tree_seg = TreePredictor(sam, box_nms_thresh=0.7)
    print('load predictor, sam:', sam_path)
    return tree_seg


def main():
    predictor = load_predictor()
    # process_images(predictor, *deal_VOC(), name='VOC')
    # process_images(predictor, *deal_CelebAMask(), name='CelebAMask')
    process_images(predictor, *deal_Cityscapes(), name='Cityscapes')


if __name__ == '__main__':
    main()
