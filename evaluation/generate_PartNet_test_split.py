import random
from pathlib import Path
import json
import os


def get_shapes(root: Path, num_max_per_shape=100, print=print):
    ins_seg_root = root.joinpath('../ins_seg_h5').resolve()
    print('Dataset split root:', ins_seg_root)
    categories = os.listdir(ins_seg_root)
    # categories = [
    #     # 'Bag',
    #     'Bed',
    #     # 'Bottle',
    #     # 'Bowl',
    #     'Chair',
    #     'Clock',
    #     'Dishwasher',
    #     'Display',
    #     'Door',
    #     'Earphone',
    #     'Faucet',
    #     # 'Hat',
    #     # 'Keyboard',
    #     'Knife',
    #     'Lamp',
    #     # 'Laptop',
    #     'Microwave',
    #     # 'Mug',
    #     'Refrigerator',
    #     # 'Scissors',
    #     'StorageFurniture',
    #     'Table',
    #     # 'TrashCan',
    #     # 'Vase',
    # ]
    print('The number of categories:', len(categories))
    anno_ids = []
    for cat in categories:
        test_json = ins_seg_root.joinpath(cat, 'test-00.json')
        with open(test_json, 'r') as f:
            test_info = json.load(f)
        print(f'Category {cat} have {len(test_info)} shapes')
        shapes = [x['anno_id'] for x in test_info]
        random.shuffle(shapes)
        anno_ids.append(shapes[:num_max_per_shape])

    eval_ids = []
    for i in range(num_max_per_shape):
        for j in range(len(categories)):
            if i < len(anno_ids[j]):
                eval_ids.append(anno_ids[j][i])
    print(f"There are {len(eval_ids)} to evaluate")
    return eval_ids


if __name__ == '__main__':
    random.seed(42)
    save_filename = Path('./results/PartNet_test.txt')
    test_ids = get_shapes(Path('~/data/PartNet/data_v0').expanduser())
    with open(save_filename, 'w') as f:
        for index in test_ids:
            f.write(f"{index}\n")