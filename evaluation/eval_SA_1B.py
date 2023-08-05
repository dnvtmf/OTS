import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from rich.console import Console

from evaluation.util import run_predictor, predictor_options, get_predictor
from tree_segmentation import MaskData, Tree2D, TreeSegmentMetric
from tree_segmentation.extension import utils


def read_annotations(json_path: Path, image_size=None):
    masks = []
    scores = []
    with open(json_path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    for ann in annotations:
        scores.append(ann['predicted_iou'])
        mask = mask_util.decode(ann['segmentation'])
        if image_size is not None:
            mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        masks.append(mask)
    scores = np.stack(scores)
    masks = np.stack(masks)
    tree2d = Tree2D(MaskData(masks=torch.from_numpy(masks), iou_preds=torch.from_numpy(scores)))
    tree2d.update_tree()
    return tree2d


def options():
    parser = argparse.ArgumentParser('Tree Segmentation for SA-1B dataset')
    parser.add_argument('-o', '--output', default=None, help='The directory to cache tree2d results')
    parser.add_argument('--data-root', default='/data5/SA-1B', help="The root path of SA-1B dataset")
    parser.add_argument('--seed', default=42, type=int, help='The seed to random choose evaluation images')
    parser.add_argument('-n', '--number', default=1000, type=int, help='The number of images to evaluate')
    parser.add_argument('--eval-part', default=110, type=int, help='The part index of SA-1B to evaluate')
    parser.add_argument('--log', default='log.txt', help='The filepath for log file')
    parser.add_argument('--print-interval', default=10, type=int, help='Print results every steps')
    utils.add_bool_option(parser, '--force', default=False, help='Force run generate')
    predictor_options(parser)
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

    get_predictor(args, print=console.print)
    device = torch.device('cuda')

    metric = TreeSegmentMetric(is_resize_2d_as_gt=False)
    timer = utils.TimeEstimator(args.number)
    time_avg = utils.TimeWatcher()
    timer.start()
    time_avg.start()
    ignore_rate = 0
    num_count = 0
    num_masks = []
    for i, image_path in enumerate(eval_image_paths, 1):
        # if i < 106:
        #     continue
        # print(i, image_path)
        image = utils.load_image(image_path)
        H, W, _ = image.shape
        scale = min(args.image_size / H, args.image_size / W)
        image = cv2.resize(image, (int(scale * W), int(scale * H)), interpolation=cv2.INTER_AREA)
        time_avg.log('image')
        gt = read_annotations(image_path.with_suffix('.json'), image_size=(image.shape[1], image.shape[0]))
        time_avg.log('gt')
        if save_root is not None:
            save_path = save_root.joinpath(image_path.name).with_suffix('.tree2d')
            if save_path.exists() and not args.force:
                prediction = Tree2D(device=device)
                prediction.load(save_path)
            else:
                prediction = run_predictor(image, device=device)

                prediction.save(save_path)
                time_avg.log('tree2d')
        else:
            prediction = run_predictor(image, device=device)
            time_avg.log('tree2d')
        if hasattr(prediction, 'ignore_rate'):
            ignore_rate += prediction.ignore_rate
            num_count += 1
        num_masks.append(prediction.num_masks)
        metric.update(prediction, gt.to(device), return_match=False)
        del prediction, gt
        time_avg.log('metric')
        timer.step()
        if i % args.print_interval == 0:
            console.print(f'Process [{i}/{args.number}], {timer.progress}')
            console.print(', '.join(f'{k}: {utils.float2str(v)}' for k, v in metric.summarize().items()))

    console.print('Complete Evalution')
    console.print('Time:', time_avg)
    for k, v in metric.summarize().items():
        console.print(f"{k:5s}: {v}")
    console.print('average masks:', np.mean(num_masks))
    console.print('ignore rate:', ignore_rate / max(1, num_count))
    console.save_text(args.log)


if __name__ == '__main__':
    main()
