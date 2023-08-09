import argparse
from tree_segmentation.extension import utils
from pathlib import Path
import numpy as np
import shutil
from time import sleep
from tqdm import tqdm


def options():
    parser = argparse.ArgumentParser('Tree Segmentation for SA-1B dataset')
    parser.add_argument('-o', '--output', default='~/data/SA_1B_test', help='The directory to cache tree2d results')
    parser.add_argument('--data-root', default='/data5/SA-1B', help="The root path of SA-1B dataset")
    parser.add_argument('--seed', default=42, type=int, help='The seed to random choose evaluation images')
    parser.add_argument('-n', '--number', default=1000, type=int, help='The number of images to evaluate')
    parser.add_argument('--eval-part', default=110, type=int, help='The part index of SA-1B to evaluate')
    args = parser.parse_args()
    return args


def main():
    args = options()
    save_root = Path(args.output).expanduser()
    print('Save test to ', save_root)
    # sleep(5)

    data_root = Path(args.data_root).joinpath(f"{args.eval_part:06d}").expanduser()
    images_paths = sorted(list(data_root.glob('*.jpg')))
    print(f'There are {len(images_paths)} images in dir: {data_root}')
    np.random.seed(42)

    eval_image_paths = np.random.choice(images_paths, args.number, replace=False)
    print(f"Try To evaluate {len(eval_image_paths)} image")

    utils.dir_create_empty(save_root)
    for image_path in tqdm(eval_image_paths):
        shutil.copy(image_path, save_root.joinpath(image_path.name))
        shutil.copy(image_path.with_suffix('.json'), save_root.joinpath(image_path.name).with_suffix('.json'))


if __name__ == '__main__':
    main()
