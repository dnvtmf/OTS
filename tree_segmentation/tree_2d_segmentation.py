from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    build_point_grid,
    coco_encode_rle,
    rle_to_mask,
)


class TreeStructure:

    def __init__(self, N=1, device=None, verbose=0):
        self.device = device
        self.parent = torch.empty(N, dtype=torch.int, device=device)
        self.first = torch.empty(N, dtype=torch.int, device=device)
        self.last = torch.empty(N, dtype=torch.int, device=device)
        self.next = torch.empty(N, dtype=torch.int, device=device)
        self.cnt = 0
        self.verbose = verbose
        self.reset()

    def reset(self):
        if self.verbose > 0:
            print('[Tree] reset')
        self.cnt = 0
        self.parent.fill_(-1)
        self.first.fill_(-1)
        self.last.fill_(-1)
        self.next.fill_(-1)

    def resize(self, N: int):
        M = self.parent.numel()
        if self.verbose > 0:
            print(f'[Tree] Resize tree from {M} to {N}')
        if M == N:
            return N, M
        if M > N:
            assert self.cnt <= N
            self.parent = self.parent[:N]
            self.first = self.first[:N]
            self.last = self.last[:N]
            self.next = self.next[:N]
            return N, M
        size = (N - M,)
        self.parent = torch.cat([self.parent, self.parent.new_full(size, -1)])
        self.first = torch.cat([self.first, self.first.new_full(size, -1)])
        self.last = torch.cat([self.last, self.last.new_full(size, -1)])
        self.next = torch.cat([self.next, self.next.new_full(size, -1)])
        return N, M

    def __len__(self):
        return self.cnt

    def node_new(self):
        assert self.cnt <= self.parent.shape[0]
        self.cnt += 1
        self.first[self.cnt] = -1
        self.last[self.cnt] = -1
        self.next[self.cnt] = -1
        self.parent[self.cnt] = -1
        if self.verbose > 1:
            print(f'[Tree] new node: {self.cnt}')
        return self.cnt

    def node_delete(self, idx: int, move_children=False):
        if self.verbose > 1:
            print(f'[Tree] delete node: {idx}, move_children={move_children}')
        last, next = self.last[idx].item(), self.next[idx].item()
        if last >= 0:
            self.next[last] = next
            if next >= 0:
                self.last[next] = last
        elif next >= 0:
            self.last[next] = last
        self.last[idx] = -1
        self.next[idx] = -1
        pa = self.parent[idx]
        if self.first[pa] == idx:
            self.first[pa] = last if last >= 0 else next
        if move_children:
            for child in self.get_children(idx):
                self.node_insert(child, pa)
            self.first[idx] = -1

    def node_replace(self, idx: int, old: int):
        if self.verbose > 1:
            print(f'[Tree] replace node: {old} by {idx}')
        last, next = self.last[old].item(), self.next[old].item()
        self.last[idx] = last
        if last >= 0:
            self.next[last] = idx
        self.next[idx] = next
        if next >= 0:
            self.last[next] = idx
        self.parent[idx] = self.parent[old]
        if self.first[self.parent[old]].item() == old:
            self.first[self.parent[old]] = idx
        if self.first[old].item() == -1:
            self.first[idx] = -1
            return
        children = self.get_children(old)
        self.first[idx] = children[0]
        for child in children:
            self.parent[child] = idx

    def node_insert(self, idx: int, parent: int):
        if self.verbose > 1:
            print(f'[Tree] insert node: {idx} below {parent}')
        self.parent[idx] = parent
        # self.first[idx] = -1
        next = self.first[parent].item()
        self.first[parent] = idx
        if next == -1:
            self.last[idx] = -1
            self.next[idx] = -1
        else:
            last = self.last[next].item()
            self.last[idx] = last
            self.next[idx] = next
            self.last[next] = idx
            if last >= 0:
                self.next[last] = idx

    def node_move(self, i: int, parent: int):
        if self.verbose > 1:
            print(f'[Tree] move node: {i} to the child of {parent} ')
        self.node_delete(i, move_children=False)
        self.node_insert(i, parent)

    def get_children(self, root: int):
        children = []
        child = self.first[root].item()
        if child == -1:
            return children
        while child != -1:
            children.append(child)
            child = self.next[child].item()
        child = self.last[children[0]]
        while child != -1:
            children.append(child)
            child = self.last[child].item()
        return children

    def get_level(self, root=0, level=1):
        assert level >= 0
        levels = self.get_levels(root, level)
        return levels[level] if len(levels) > level else torch.tensor([], dtype=torch.int, device=self.device)

    def get_levels(self, root=0, depth=-1):
        now_level = [root]
        levels = [torch.tensor(now_level, dtype=torch.int, device=self.device)]
        while len(now_level) > 0 and depth != 0:
            next_level = []
            for x in now_level:
                next_level.extend(self.get_children(x))
            if len(next_level) == 0:
                break
            levels.append(torch.tensor(next_level, dtype=torch.int, device=self.device))
            now_level = next_level
            depth -= 1

        return levels

    def node_rearrange(self, indices=None):
        if self.verbose > 0:
            print(f'[Tree] rerange nodes')
        if indices is None:
            indices = torch.cat(self.get_levels(), dim=0)
        indices = indices.long()
        num = len(indices)
        assert 0 <= indices.min() and indices.max() <= self.cnt
        new_indices = indices.new_full((self.cnt + 2,), -1)
        new_indices[indices + 1] = torch.arange(len(indices), dtype=indices.dtype, device=indices.device)
        self.parent[:num] = new_indices[self.parent[indices].long() + 1]
        self.first[:num] = new_indices[self.first[indices].long() + 1]
        self.last[:num] = new_indices[self.last[indices].long() + 1]
        self.next[:num] = new_indices[self.next[indices].long() + 1]
        self.cnt = len(indices) - 1
        return indices, new_indices

    def print_tree(self):
        from rich.tree import Tree
        import rich
        levels = self.get_levels()
        print_tree = Tree('0: Tree Root')
        nodes = {0: print_tree}
        for i, level in enumerate(levels):
            if i == 0:
                continue
            for j in level:
                j = j.item()
                p = self.parent[j].item()
                nodes[j] = nodes[p].add(f"{j}")
        rich.print(print_tree)

    def to(self, device):
        self.device = device
        self.parent = self.parent.to(device)
        self.next = self.next.to(device)
        self.last = self.last.to(device)
        self.first = self.first.to(device)
        return self


class TreeData(TreeStructure):

    def __init__(
        self,
        mask_data: MaskData = None,
        in_threshold=0.9,
        union_threshold=0.1,
        min_area=100,
        in_thres_area=10,
        device=None,
        verbose=0,
    ) -> None:
        self.data = self._fileter(mask_data)
        self.in_threshold = in_threshold
        self.in_thres_area = in_thres_area
        self.union_threshold = union_threshold
        # self.min_area_rate = min_area_rate
        self.min_area = min_area  # self['masks'][0].numel() * self.min_area_rate
        num = 0 if self.data is None else len(self.data['masks'])
        if self.data is not None:
            self.data['area'] = self.data['masks'].sum(dim=[-1, -2])
        super().__init__(num + 1, device, verbose)
        self.num_samples = torch.zeros(num + 1, dtype=torch.int, device=device)

    def _fileter(self, data: MaskData = None):
        if data is None:
            return data
        else:
            return MaskData(masks=data['masks'], iou_preds=data['iou_preds'])

    def reset(self, mask_data: MaskData = None):
        mask_data = self._fileter(mask_data)
        if self.verbose > 0:
            print('[Tree2D] Reset TreeData')
        if mask_data is not None:  # and 'area' not in mask_data
            mask_data['area'] = mask_data['masks'].sum(dim=[-1, -2])
            self.data = mask_data
        num = 0 if self.data is None else len(self.data['masks'])
        self.parent = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 父节点
        self.first = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 第一个子节点
        self.last = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 同级别前一个节点
        self.next = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 同级别后一个节点
        self.num_samples = torch.zeros((num + 1,), dtype=torch.int, device=self.device)
        self.cnt = 0

    def resize(self, N: int):
        N, M = super().resize(N)
        if M > N:
            self.num_samples = self.num_samples[:N]
        elif M < N:
            self.num_samples = torch.cat([self.num_samples, self.num_samples.new_full((N - M,), 0)])

    def node_new(self):
        index = super().node_new()
        self.num_samples[index] = 0
        return index

    def node_replace(self, idx: int, old: int):
        super().node_replace(idx, old)
        self.num_samples[idx] = self.num_samples[old]

    def node_rearrange(self, indices=None):
        indices, _ = super().node_rearrange(indices)
        self.num_samples[:len(indices)] = self.num_samples[indices]

    def insert(self, i, now=0):
        area_i = self.data['area'][i - 1]
        if area_i < self.min_area:
            if self.verbose > 0:
                print(f'[Tree2D] ingore {i} due to small area {area_i.item()} vs {self.min_area}')
            return
        # print('=' * 10, i, now, f"area={area_i}", '=' * 10)
        mask_i = self.data['masks'][i - 1]
        # bbox_i = self['bbox'][i - 1]
        nodes_in_i = []
        nodes_union = []
        for j in self.get_children(now):
            mask_j = self.data['masks'][j - 1]
            area_j = self.data['area'][j - 1]
            inter = torch.logical_and(mask_i, mask_j).sum()
            union = area_i + area_j - inter
            if inter / area_i >= self.in_threshold or area_i - inter < self.in_thres_area:  # i in j
                if inter / area_j >= self.in_threshold or area_j - inter < self.in_thres_area:  # i == j
                    # print(f"{i} same {j}: area:{area_i}/{area_j},inter={inter}/{inter/area_i:.2%}/{inter/area_j:.2%}")
                    # print(f'inter area: {area_i - inter}/{area_j-inter} < {self.in_thres_area}')
                    if self.data['iou_preds'][i - 1] > self.data['iou_preds'][j - 1]:
                        self.node_replace(i, j)
                    else:
                        if self.verbose > 0:
                            print(f'[Tree2D] {i} is same with {j}, skip')
                    return
                if self.verbose > 0:
                    print(f'[Tree2D] {i} in {j}, {inter.item() / area_i.item():.2%}')
                if self.first[j] < 0:
                    self.node_insert(i, j)
                    self.num_samples[i] = self.num_samples[i] * 0.5
                else:
                    self.insert(i, j)
                return
            elif inter / area_j >= self.in_threshold:  # j in i
                nodes_in_i.append(j)
            elif inter / union > self.union_threshold:
                nodes_union.append((j, (inter / union).item()))
            else:  # no intersect
                pass
        if len(nodes_union) > 0:
            print(f"[Tree2D] {i} union with {nodes_union}")
            return
        # assert len(nodes_union) == 0, f"{i} union with {nodes_union}"
        if self.verbose > 0:
            print(f"[Tree2D] {i} in {now} before {self.first[now].item()}", nodes_in_i)
        self.node_insert(i, now)
        self.num_samples[i] = self.num_samples[i] * 0.5
        if len(nodes_in_i) > 0:
            for j in nodes_in_i:
                self.node_move(j, i)
                if self.verbose > 1:
                    print(f"[Tree2D] move {j} from {now} to {i}")

    def update_tree(self):
        while self.cnt < len(self.data['masks']):
            self.insert(self.node_new())  # self.print_tree()

    def sample_grid(self, points_per_side=32):
        return build_point_grid(points_per_side)

    @torch.no_grad()
    def sample_unfilled(self, num_points=1024, threshold=0.9) -> Tuple[Optional[np.ndarray], Optional[Tensor]]:
        if self.data is None:
            return np.random.random((num_points, 2)), None
        N, H, W = self.data['masks'].shape
        level = 0
        for level, nodes in enumerate(self.get_levels()):
            if level == 0:
                continue
            assert 0 < nodes.min() and nodes.max() <= N
            area_sum = self.data['area'][nodes - 1].sum()
            # par_nodes = set(self.parent[i] for i in nodes)
            par_nodes = torch.unique(self.parent[nodes])
            assert -1 <= par_nodes.min() and par_nodes.max() <= N
            # area_total = sum(self['area'][i - 1] for i in par_nodes if i > 0) if level else H * W
            area_total = self.data['area'][par_nodes - 1].sum() if level > 1 else H * W
            # print('level:', level, nodes, par_nodes, area_sum, area_total)
            if area_sum / area_total >= threshold:
                print(f'[Tree2D] level {level} are fillded, rate: {area_sum / area_total:.2%}')
                continue
            # print(par_nodes)
            # print([self['masks'][i - 1] for i in par_nodes if i > 0])
            # print(torch.stack([self['masks'][i - 1] for i in par_nodes if i > 0], dim=0).shape)
            if level == 1:
                wanted_mask = torch.ones_like(self.data['masks'][0])
            else:
                wanted_mask = self.data['masks'][par_nodes - 1].any(dim=0)
            filled_mask = self.data['masks'][nodes - 1].any(dim=0)
            unfilled_mask = torch.logical_and(wanted_mask, ~filled_mask)
            xy = torch.nonzero(unfilled_mask).flip(-1)
            sample_index = torch.randint(0, xy.shape[0], (num_points,))
            assert 0 <= sample_index.min() and sample_index.max() < xy.shape[0]
            sample_points = xy[sample_index].float()
            sample_points = sample_points + torch.randn_like(sample_points) * 1.0
            sample_points = sample_points / sample_points.new_tensor([W, H])
            return sample_points.cpu().numpy(), unfilled_mask
        if self.verbose > 0:
            print(f'New level {level + 1}')
        unfilled_mask = torch.ones_like(self.data['masks'][0])
        return np.random.random((num_points, 2)), unfilled_mask

    @torch.no_grad()
    def sample_by_counts(self, num_points=1024, sample_limit_fn=None) -> Optional[np.ndarray]:
        """当某一mask的采样的点的数量 < sample_limit_fn(area) 时, 从该mask中采样点"""
        if self.first[0] <= 0:
            return np.random.random((num_points, 2))

        if sample_limit_fn is None:  # 默认采样点数限制函数: 面积的平方根
            sample_limit_fn = lambda area: np.sqrt(area.item())

        indices = self.get_sample_indices(sample_limit_fn, 0)
        if len(indices) == 0:
            return None
        sample_mask = torch.zeros_like(self.data['masks'][0], dtype=torch.int)
        for index in indices:
            sample_mask[self.data['masks'][index - 1]] = index
        H, W = sample_mask.shape
        xy = torch.nonzero(sample_mask).flip(-1)
        sample_index = torch.randint(0, xy.shape[0], (num_points,))
        sample_points = xy[sample_index]
        indices = sample_mask[sample_points[:, 1], sample_points[:, 0]].long().to(self.num_samples.device)
        indices, counts = indices.unique(return_counts=True)
        self.num_samples[indices] += counts
        sample_points = sample_points.float()
        sample_points = sample_points + torch.randn_like(sample_points) * 1.0  # 随机噪声
        sample_points = sample_points / sample_points.new_tensor([W, H])
        return sample_points.cpu().numpy()

    def get_sample_indices(self, sample_limit_fn, root=0):
        sample_indices = []
        for child in self.get_children(root):
            if self.num_samples[child] > sample_limit_fn(self.data['area'][child - 1]):
                sample_indices.extend(self.get_sample_indices(sample_limit_fn, child))
            else:
                sample_indices.append(child)
        return sample_indices

    def cat(self, mask_data: MaskData) -> None:
        mask_data = self._fileter(mask_data)
        if self.verbose > 0:
            print('[Tree2D] cat', (None if self.data is None else self.data['masks'].shape), mask_data['masks'].shape)
        if self.data is None:
            self.reset(mask_data)
            return
        mask_data['area'] = mask_data['masks'].sum(dim=[-1, -2])
        self.data.cat(mask_data)
        self.resize(self.data['masks'].shape[0] + 1)

    # def apply_nms(self, box_nms_thresh=0.7):
    #     keep_by_nms = batched_nms(
    #         self.data["boxes"].float(),
    #         self.data["iou_preds"],
    #         torch.zeros_like(self.data["boxes"][:, 0]),
    #         # categories
    #         iou_threshold=box_nms_thresh,
    #     )
    #     assert keep_by_nms.max() < len(self.data["boxes"])
    #     self.data.filter(keep_by_nms)
    #     self.reset(self.data)
    #     return keep_by_nms

    def filter(self, keep) -> None:
        self.data.filter(keep)
        self.reset()

    def remove_not_in_tree(self):
        """remove duplicate after merge new results"""
        if self.verbose > 0:
            print('[Tree2D] remove_not_in_tree')
        if self.cnt == 0:  # empty
            return
        # if self.cnt == self.data['masks'].shape[0]:
        #     return
        keep = torch.cat(self.get_levels(), dim=0)[1:].long()
        assert 0 < keep.min() and keep.max() <= len(self.data['masks'])
        self.data.filter(keep - 1)
        self.node_rearrange()
        self.resize(self.cnt + 1)  # self.print_tree()

    def dillate(self):
        """腐蚀边界, 直至无空隙"""
        pass

    def output(self, output_mode='binary_mask'):
        assert output_mode in ["binary_mask", "uncompressed_rle", "coco_rle"], f"Unknown output_mode {output_mode}."
        # Compress to RLE
        # data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        # data["rles"] = mask_to_rle_pytorch(data["masks"])
        # del data["masks"]
        # Encode masks
        if output_mode == "coco_rle":
            self.data["segmentations"] = [coco_encode_rle(rle) for rle in self.data["rles"]]
        elif output_mode == "binary_mask":
            self.data["segmentations"] = [rle_to_mask(rle) for rle in self.data["rles"]]
        else:
            self.data["segmentations"] = self.data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(self.data["segmentations"])):
            ann = {
                "segmentation": self.data["segmentations"][idx],
                "area": area_from_rle(self.data["rles"][idx]),
                # "bbox": box_xyxy_to_xywh(self.data["boxes"][idx]).tolist(),
                "predicted_iou": self.data["iou_preds"][idx].item(),
                # "point_coords": [self.data["points"][idx].tolist()],
                # "stability_score": self.data["stability_score"][idx].item(),
                # "crop_box": box_xyxy_to_xywh(self.data["crop_boxes"][idx]).tolist(),
                # TODO: include tree results
            }
            curr_anns.append(ann)

        return curr_anns

    def save(self, filename, **kwargs):
        if self.data is None:
            data = {}
        else:
            self.remove_not_in_tree()
            masks = []
            for level, nodes in enumerate(self.get_levels()):
                if level == 0:
                    continue
                mask = torch.zeros_like(self.data['masks'][0], dtype=torch.int)
                for i in nodes:
                    mask[self.data['masks'][i - 1]] = i
                masks.append(mask)
            masks = torch.stack(masks, dim=0)
            print(masks.shape)
            data = {k: masks if k == 'masks' else v for k, v in self.data.items()}
        data.update({
            'parent': self.parent,
            'first': self.first,
            'next': self.next,
            'last': self.last,
            'cnt': self.cnt,
            'num_samples': self.num_samples,
            'extra': kwargs
        })
        if filename is not None:
            torch.save(data, filename)
            if self.verbose > 0:
                print(f"[Tree2D]save now Tree2D to {filename}")
        return data

    def load(self, filename, **data):
        if filename is not None:
            data.update(**torch.load(filename, map_location=self.device))
            if self.verbose > 0:
                print('[Tree2D] load Tree2D from:', filename)
        self.parent = data.pop('parent')
        self.first = data.pop('first')
        self.next = data.pop('next')
        self.last = data.pop('last')
        self.cnt = data.pop('cnt')
        self.num_samples = data.pop('num_samples')
        extra = data.pop('extra')
        if len(data) > 0:
            self.data = MaskData(**data)
        else:
            self.data = None
        if self.data is not None:
            masks = self.data['masks']
            self.data['masks'] = torch.zeros((self.cnt, *masks.shape[1:]), dtype=torch.bool, device=self.device)
            for level, nodes in enumerate(self.get_levels()):
                if level == 0:
                    continue
                for i in nodes:
                    self.data['masks'][i - 1] = masks[level - 1] == i
        return extra

    def to(self, device):
        super().to(device)
        for k, v in self.data.items():
            if isinstance(v, Tensor):
                self.data[k] = v.to(device)
        return self
