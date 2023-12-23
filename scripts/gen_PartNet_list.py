import math
from pathlib import Path
import random
from collections import defaultdict
import json
from tqdm import tqdm

import numpy as np

data_root = Path('~/data/PartNet/data_v0').expanduser()
obj_names = sorted([path.stem for path in data_root.glob('*')])
print(len(obj_names), obj_names[0])

random.seed(42)

cat_names = defaultdict(list)
for obj_name in tqdm(obj_names):
    with open(data_root / obj_name / 'meta.json', 'r') as f:
        meta = json.load(f)
        cat_names[meta['model_cat']].append(obj_name)
print(f"There are {len(cat_names)} categories")
weights = []
for cat_name, names in cat_names.items():
    # print(cat_name, len(names))
    weights.append(math.sqrt(len(names)))

weights = np.array(weights)
weights = weights / weights.sum() * 1000

sampled_names = []
for i, (cat_name, names) in enumerate(cat_names.items()):
    # num_sampled = int(weights[i])
    num_sampled = 50
    print(f"{cat_name:20s} sample {num_sampled:4d} / {len(names):4d}")
    obj_names_c = random.choices(names, k=num_sampled)
    sampled_names.extend(obj_names_c)
    # print(len(obj_names_c))
    # if i == 0:
    #     print(obj_names_c[:10])

random.shuffle(sampled_names)
with open('results/PartNet_list.txt', 'w') as f:
    for sampled_name in sampled_names:
        f.write(f"{sampled_name}\n")
