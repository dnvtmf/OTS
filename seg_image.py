import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.cuda
from PIL import Image
from tqdm import tqdm

from segment_anything import build_sam
from semantic_sam import semantic_sam_l, semantic_sam_t
from tree_segmentation import TreePredictor
from tree_segmentation.extension import utils


def options():
    parser = argparse.ArgumentParser('Tree Segmentation for Images')
    parser.add_argument('image', help='The path of input image')
    parser.add_argument('-sam', '--segment-anything', action='store_true', default=False)
    parser.add_argument('-ssl', '--semantic-sam-l', action='store_true', default=False)
    parser.add_argument('-sst', '--semantic-sam-t', action='store_true', default=False)
    parser.add_argument(
        '-w', '--weights', default='./weights', help='The directory stored pretrained model of SAM/Semantic-SAM')
    parser.add_argument('-o', '--output', default='.', help='The directory of output')
    parser.add_argument(
        '-f',
        '--format',
        default='.png',
        choices=['.png', '.tiff', '.tree2d'],
        help='The format of output results, choose in ].png, .tiff, .tree2d]')
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
    parser.add_argument('--in_threshold', default=0.8, type=float)
    parser.add_argument('--in_area_threshold', default=50, type=float)
    parser.add_argument('--union_threshold', default=0.1, type=float)
    parser.add_argument('--min_area', default=100, type=float)
    utils.add_bool_option(parser, '--skip-error', default=False, help='Continue segment when error occured')
    utils.add_bool_option(parser, '-s', '--skip-exist', default=True, help='Skip segmented images')
    args = parser.parse_args()
    return args


def load_predictor(args):
    model_dir = Path(args.weights).expanduser()
    if args.segment_anything:
        assert model_dir.joinpath('sam_vit_h_4b8939.pth').exists(), f"Not model 'sam_vit_h_4b8939.pth' in {model_dir}"
        model = build_sam(model_dir.joinpath('sam_vit_h_4b8939.pth'))
        print('Use Model SAM')
    elif args.semantic_sam_t:
        assert model_dir.joinpath('swint_only_sam_many2many.pth').exists(), \
            f"Not model 'swint_only_sam_many2many.pth' in {model_dir}"
        model = semantic_sam_t(model_dir.joinpath('swint_only_sam_many2many.pth'))
        print('Use Model Semantic-SAM-t')
    else:  # elif args.semantic_sam_l:
        assert model_dir.joinpath('swinl_only_sam_many2many.pth').exists(), \
            f"Not model 'swinl_only_sam_many2many.pth' in {model_dir}"
        model = semantic_sam_l(model_dir.joinpath('swinl_only_sam_many2many.pth'))
        print('Use Model Semantic-SAM-l')
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
    print('build predictor')
    return predictor, device


def deal_one_image(args, predictor: TreePredictor, device, image_path: Path, save_dir: Path):
    if args.skip_exist:
        if args.format == '.tree2d' and save_dir.joinpath(image_path.stem + '.tree2d').exists():
            return
        if args.format == '.tiff' and save_dir.joinpath(image_path.stem + '.tiff').exists():
            return
        if args.format == '.png' and len(list(save_dir.glob(f'{image_path.stem}_level_*.png'))) > 0:
            return
    image = utils.load_image(image_path)[..., :3]
    # print('image:', image.shape)

    results = predictor.tree_generate(
        image=image,
        points_per_side=args.points_per_side,
        points_per_update=args.points_per_update,
        min_mask_region_area=args.min_area,
        max_steps=args.max_steps,
        in_threshold=args.in_threshold,
        in_thre_area=args.in_area_threshold,
        union_threshold=args.union_threshold,
        device=device)
    results.post_process()
    # results.print_tree()

    if args.format == '.tree2d':
        results.save(save_dir.joinpath(image_path.stem + '.tree2d'))
    elif args.format == '.tiff':
        masks = results.masks.cpu().numpy()
        # print(masks.shape, masks.dtype)
        utils.save_image(save_dir.joinpath(image_path.stem + '.tiff'), masks)
    else:
        masks = results.masks.cpu().numpy()
        levels = results.get_levels()
        for i, level_i in enumerate(levels):
            if i == 0:
                continue
            mask = masks[level_i.cpu().numpy() - 1].astype(np.uint8)
            mask = np.max(mask * np.arange(1, mask.shape[0] + 1)[:, None, None], axis=0).astype(np.uint8)
            # plt.imshow(mask)
            # plt.show()
            colors = np.array([[1, 1, 1]] + sns.color_palette(n_colors=len(level_i)))
            mask_i = Image.fromarray(mask, mode='P')
            mask_i.putpalette((colors * 255).astype(np.uint8))
            mask_i.save(save_dir.joinpath(f'{image_path.stem}_level_{i}.png'))


def main():
    args = options()
    print('options:', args)
    save_dir = Path(args.output).expanduser()
    assert save_dir.is_dir(), f"The output directory '{save_dir}' is not exists"

    image_path = Path(args.image).expanduser()
    assert image_path.exists(), f"Image {image_path} not exist"
    predictor, device = load_predictor(args)

    if image_path.is_dir():
        image_paths = sorted(list(image_path.glob('*.*')))
        for img_path in tqdm(image_paths):
            if img_path.suffix in utils.image_extensions:
                try:
                    deal_one_image(args, predictor, device, img_path, save_dir)
                except Exception as e:
                    if args.skip_error:
                        print(str(e))
                    else:
                        raise e
    else:
        assert image_path.suffix in utils.image_extensions, f"File {image_path} not a image"
        deal_one_image(args, predictor, device, image_path, save_dir)
    print('Complete')


if __name__ == '__main__':
    main()
