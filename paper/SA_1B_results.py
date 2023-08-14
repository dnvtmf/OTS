from pathlib import Path
import torch
from tqdm import tqdm

from tree_segmentation import TreeSegmentMetric, Tree2D

save_root = Path('./results/cache/SA_1B')
save_gt_dir = save_root.joinpath('gt')
f = open('./results/SA_1B_results.csv', 'w')
device = torch.device('cuda')

for suffix in [
        'SAM_origin', 'SAM_L_origin', 'SAM_B_origin', 'SemanticSAM_L_origin', 'SemanticSAM_T_origin', 'SAM_0.5',
        'SAM_L_0.5', 'SAM_B_0.5', 'SemanticSAM_L_0.5', 'SemanticSAM_T_0.5'
]:
    filenames = sorted([p.name for p in save_root.joinpath(suffix).glob('*.tree2d')])[:1000]
    print(f"There are {len(filenames)} test images")
    metric = TreeSegmentMetric()
    num_pd = 0
    for filename in tqdm(filenames):
        gt = Tree2D(device=device)
        gt.load(save_gt_dir.joinpath(filename).with_suffix('.tree2d'))

        pd = Tree2D(device=device)
        pd.load(save_root.joinpath(suffix, filename))
        num_pd += pd.cnt
        metric.update(pd, gt)
        # break
    msg = [f"{suffix:20}"] + [f"{k}={v:.4f}" for k, v in metric.summarize().items()]
    msg.append(f"num={num_pd/len(filenames):5.1f}")
    print(', '.join(msg))
    f.write(', '.join(msg) + '\n')
    # break

metric = TreeSegmentMetric()
num_gt = 0
for filename in tqdm(filenames):
    gt = Tree2D(device=device)
    gt.load(save_gt_dir.joinpath(filename).with_suffix('.tree2d'))
    num_gt += gt.cnt
    metric.update(gt, gt)
msg = [f"{'gt':20}"] + [f"{k}={v:.4f}" for k, v in metric.summarize().items()]
msg.append(f"num={num_gt/len(filenames):5.1f}")
print(', '.join(msg))
f.write(', '.join(msg) + '\n')
f.close()