from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from segment_anything.utils.amg import (MaskData, build_point_grid, remove_small_regions)
from tree_segmentation.tree_structure import TreeStructure
from tree_segmentation.extension import utils, Masks


class Tree2D(TreeStructure):
    _masks: Optional[Masks]

    def __init__(
        self,
        masks: Union[MaskData, Tensor, 'Tree2D', Masks] = None,
        scores: Tensor = None,
        in_threshold=0.8,
        union_threshold=0.1,
        min_area=100,
        in_thres_area=10,
        device=None,
        verbose=0,
        format=Masks.ENCODED,
    ) -> None:
        self._is_compressed = False
        self.in_threshold = in_threshold
        self.in_thres_area = in_thres_area
        self.union_threshold = union_threshold
        # self.min_area_rate = min_area_rate
        self.min_area = min_area  # self['masks'][0].numel() * self.min_area_rate
        self.format = format

        if isinstance(masks, MaskData):
            self._masks, scores = Masks(masks['masks'], format=format), masks['iou_preds']
        elif isinstance(masks, Tensor):
            self._masks = Masks(masks['masks'], format=format)
        elif isinstance(masks, Tree2D):
            self._masks, scores = masks._masks, masks.scores
        elif isinstance(masks, Masks):
            self._masks = masks
        else:
            self._masks, scores = None, torch.ones(0, device=device)
        self.scores = torch.ones(self._masks.shape[0], device=device) if scores is None else scores

        if device is not None:
            if self._masks is not None:
                self._masks = self._masks.to(device)
            self.scores = self.scores.to(device)
        if self._masks is not None:
            self._masks.format = format

        num = self.scores.shape[0]
        self.num_samples = torch.zeros(num + 1, dtype=torch.int, device=device)
        super().__init__(num + 1, device, verbose)

    @property
    def num_masks(self) -> int:
        return self.scores.shape[0]

    @property
    def is_compressed(self):
        return self._is_compressed

    def clear(self, masks: Union[MaskData, Tensor, 'Tree2D', Masks] = None, scores: Tensor = None):
        if self.verbose > 0:
            print('[Tree2D] clear TreeData')
        if isinstance(masks, MaskData):
            self._masks, scores = Masks(masks['masks'], format=self.format), masks['iou_preds']
        elif isinstance(masks, Tensor):
            self._masks = Masks(masks['masks'], format=self.format)
        elif isinstance(masks, Tree2D):
            self._masks, scores = masks._masks, masks.scores
        elif isinstance(masks, Masks):
            self._masks = masks
        else:
            self._masks = None

        self.scores = torch.ones(self._masks.shape[0], device=self.device) if scores is None else scores
        if self._masks is not None:
            self._masks = self._masks.to(self.device)
        self._masks.format = self.format
        self.scores = self.scores.to(self.device)
        num = self.scores.shape[0]
        self.parent = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 父节点
        self.first = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 第一个子节点
        self.last = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 同级别前一个节点
        self.next = torch.full((num + 1,), -1, dtype=torch.int, device=self.device)  # 同级别后一个节点
        self.num_samples = torch.zeros((num + 1,), dtype=torch.int, device=self.device)
        self.reset()

    def resize(self, N: int):
        self.uncompress()
        N, M = super().resize(N)
        if self._masks is not None:
            M_ = len(self._masks) + 1
            if M_ > N:
                self._masks = self._masks[:N - 1]
            else:
                self._masks = self._masks.pad(0, N - M_)
        if M > N:
            self.scores = self.scores[:N - 1]
            self.num_samples = self.num_samples[:N]
        elif M < N:
            self.scores = F.pad(self.scores, [0, N - M], value=0)
            self.num_samples = F.pad(self.num_samples, [0, N - M], value=0)
        assert len(self.parent) == N and len(self.num_samples) == N, f"{N}!={len(self.parent)}/{len(self.num_samples)}"
        if self._masks is None:
            assert M == 1 and len(self.scores) == N - 1
        else:
            assert len(self.scores) == N - 1 and len(self._masks) == N - 1, f"{len(self.scores)} != {N - 1}"
        assert self.cnt <= N, f"cnt={self.cnt} > N={N}"
        return N, M

    def new_node(self, mask: Masks, score=1):
        self.resize(self.num_masks + 2)
        if self._masks is None:
            self._masks = mask[None]
        else:
            self._masks[-1] = mask
        self.scores[-1] = score
        index = self.node_new()
        self.num_samples[index] = 0
        return index

    def node_rearrange(self, indices=None):
        if indices is not None:
            assert 0 <= indices.min() and indices.max() <= self.cnt
        self.uncompress()
        indices, new_indices = super().node_rearrange(indices)
        self.num_samples[:len(indices)] = self.num_samples[indices]
        idx_ = indices[indices != 0] - 1
        if len(idx_) > 0:
            assert 0 <= idx_.min() and idx_.max() < len(self._masks)
            assert len(self._masks) == len(self.scores)
            num = len(idx_)
            self._masks[:num] = self._masks[idx_]
            self.scores[:num] = self.scores[idx_]
        return indices, new_indices

    def node_replace(self, i, j):
        super().node_replace(i, j)
        self.num_samples[i] = self.num_samples[j]

    def insert(self, mask: Union[Masks, Tensor], score=1, i: int = None) -> int:
        self.uncompress()
        if isinstance(mask, Tensor):
            mask = Masks(mask, format=self.format)
        mask = mask.to(self.device)
        if isinstance(score, Tensor):
            score = score.item()
        area = mask.area
        if area < self.min_area:
            if self.verbose > 1:
                print(f'[Tree2D] ingore insert due to small area {area.item()} vs {self.min_area}')
            return -1
        return self._insert(mask, score, i, 0)

    def _insert(self, mask: Masks, score=1, i: int = None, now=0) -> int:
        # print('=' * 10, i, now, f"area={area}", '=' * 10)
        # bbox_i = self['bbox'][i - 1]
        nodes_in_i = []
        nodes_union = []
        for j in self.get_children(now):
            mask_j = self._masks[j - 1]
            inter = mask & mask_j
            union = mask.area + mask_j.area - inter
            if inter / mask.area >= self.in_threshold or mask.area - inter < self.in_thres_area:  # i in j
                if inter / mask_j.area >= self.in_threshold or mask_j.area - inter < self.in_thres_area:  # i == j
                    if score > self.scores[j - 1]:
                        if i is None:
                            self.scores[j - 1] = score
                            self._masks[j - 1] = mask
                        else:
                            self.node_replace(i, j)
                        if self.verbose > 1:
                            print(f'[Tree2D] insert {i} replace {j}')
                    else:
                        if self.verbose > 1:
                            print(f'[Tree2D] insert {i} is same with {j}, skip')
                    return j
                if self.verbose > 1:
                    print(f'[Tree2D] insert in {j}, {inter.item() / mask.area.item():.2%}')
                if self.first[j] < 0:
                    if i is None:
                        i = self.new_node(mask, score)
                    self.node_insert(i, j)
                    # self.num_samples[i] = self.num_samples[i] * 0.5
                    return i
                else:
                    return self._insert(mask, score, i, j)
            elif inter / mask_j.area >= self.in_threshold:  # j in i
                nodes_in_i.append(j)
            elif inter / union > self.union_threshold:
                nodes_union.append((j, (inter / union).item()))
            else:  # no intersect
                pass
        if len(nodes_union) > 0:
            if self.verbose > 0:
                print(f"[Tree2D] {i} union with {nodes_union}")
            return -1
        if i is None:
            i = self.new_node(mask, score)
        # assert len(nodes_union) == 0, f"{i} union with {nodes_union}"
        if self.verbose > 1:
            print(f"[Tree2D] insert {i} in {now} before {self.first[now].item()}", nodes_in_i)
        self.node_insert(i, now)
        # self.num_samples[i] = self.num_samples[i] * 0.5
        if len(nodes_in_i) > 0:
            for j in nodes_in_i:
                self.node_move(j, i)
                if self.verbose > 2:
                    print(f"[Tree2D] move {j} from {now} to {i}")
        return i

    def insert_batch(self, masks: Union[MaskData, 'Tree2D', Tensor, Masks], scores=None):
        self.uncompress()
        if isinstance(masks, MaskData):
            masks, scores = Masks(masks['masks'], format=self.format), masks['iou_preds']
        elif isinstance(masks, Tree2D):
            masks, scores = masks.masks, masks.scores
        elif isinstance(masks, Tensor):
            masks = Masks(masks, format=self.format)
        elif isinstance(masks, np.ndarray):
            masks = Masks(torch.from_numpy(masks).to(self.device), format=self.format)
        if scores is None:
            scores = torch.ones(masks.shape[0], device=masks.device)
        num_ignored = 0
        for i in range(len(masks)):
            if self.insert(masks[i], scores[i]) < 0:
                num_ignored += 1
        if self.verbose > 0:
            print(f'[Tree2D] Try to insert {len(masks)} masks, {num_ignored} are ignored')
        return num_ignored

    def cat(self, masks: Union[MaskData, 'Tree2D', Tensor, Masks], scores=None):
        self.uncompress()
        if isinstance(masks, MaskData):
            masks, scores = Masks(masks['masks'], format=self.format), masks['iou_preds']
        elif isinstance(masks, Tree2D):
            masks, scores = masks.masks, masks.scores
        elif isinstance(masks, Tensor):
            masks = Masks(masks, format=self.format)
        if scores is None:
            scores = torch.ones(masks.shape[0], device=masks.device)
        assert masks.ndim == 3
        N = len(masks)
        if self._masks is not None:
            masks = self.masks + masks
        self._masks = masks
        self.resize(len(self.scores) + N + 1)
        masks, scores = masks.to(self.device), scores.to(self.device)
        self.scores[-N:] = scores
        if self.verbose > 0:
            print(f'[Tree2D] cat {len(masks)} masks')
        return self

    def update_tree(self):
        """insert undealed masks"""
        if self._masks is None:
            return
        self.uncompress()
        num_ignored = 0
        while self.cnt < len(self._masks):
            i = self.node_new()
            if self.insert(self._masks[i - 1], self.scores[i - 1], i) < 0:
                num_ignored += 1
        return num_ignored

    def sample_grid(self, points_per_side=32):
        return build_point_grid(points_per_side)

    @torch.no_grad()
    def sample_by_counts(self, num_points=1024, sample_limit_fn=None, noise=1.0, ratio=0.5) -> Optional[np.ndarray]:
        """当某一mask的采样的点的数量 < sample_limit_fn(area) 时, 从该mask中采样点"""
        if self.first[0] <= 0:
            self.num_samples[0] += num_points
            return np.random.random((num_points, 2))
        self.uncompress()

        if sample_limit_fn is None:  # 默认采样点数限制函数: 面积的平方根
            sample_limit_fn = lambda area: np.sqrt(area)
        indices = self.get_sample_indices(sample_limit_fn, 0)
        if len(indices) == 0:
            return None
        _, H, W = self._masks.shape
        sample_mask = torch.full((H, W), -1, dtype=torch.int, device=self.device)
        unfill_mask = torch.full((H, W), -1, dtype=torch.int, device=self.device)
        for index in indices:
            if index == 0:
                mask = torch.ones((H, W), device=self.device, dtype=torch.bool)
            else:
                mask = self._masks[index - 1].binary().data
            sample_mask[mask] = index
            for c in self.get_children(index):
                mask = mask & torch.logical_not(self._masks[c - 1].binary().data)
            unfill_mask[mask] = index
        H, W = sample_mask.shape
        xy1 = torch.nonzero(sample_mask >= 0).flip(-1)
        xy2 = torch.nonzero(unfill_mask >= 0).flip(-1)
        num2 = min(xy2.shape[0], int(num_points * ratio))
        num1 = min(xy1.shape[0], num_points - num2)
        if num1 > 0:
            p1 = xy1[torch.randint(0, xy1.shape[0], (num_points - int(num_points * ratio),))]
            indices = sample_mask[p1[:, 1], p1[:, 0]].long().to(self.num_samples.device)
            indices, counts = indices.unique(return_counts=True)
            self.num_samples[indices] += counts
        else:
            p1 = None
        if num2 > 0:
            p2 = xy2[torch.randint(0, xy2.shape[0], (int(num_points * ratio),))]
            indices = unfill_mask[p2[:, 1], p2[:, 0]].long().to(self.num_samples.device)
            indices, counts = indices.unique(return_counts=True)
            self.num_samples[indices] += counts
        else:
            p2 = None
        if p1 is None:
            if p2 is None:
                return None
            else:
                sample_points = p2.float()
        else:
            if p2 is None:
                sample_points = p1.float()
            else:
                sample_points = torch.cat([p1, p2], dim=0).float()
        if len(sample_points) == 0:
            return None
        sample_points = sample_points + torch.randn_like(sample_points) * noise  # 随机噪声
        sample_points = sample_points / sample_points.new_tensor([W, H])
        return sample_points.clamp(0, 1).cpu().numpy()

    def get_sample_indices(self, sample_limit_fn, root=0):
        """find the all non intersecion masks from root to leaf which sample number is not exceed limie"""
        _, H, W = self._masks.shape
        if root == 0 and self.num_samples[0] <= sample_limit_fn(H * W):
            return [0]
        sample_indices = []
        for child in self.get_children(root):
            if self.num_samples[child] > sample_limit_fn(self._masks.area[child - 1].item()):
                sample_indices.extend(self.get_sample_indices(sample_limit_fn, child))
            else:
                sample_indices.append(child)
        return sample_indices

    @torch.no_grad()
    def sample_each_mask(self, num_point_per_mask=1, noise=1.0) -> np.ndarray:
        self.uncompress()
        self.remove_not_in_tree()
        points = []
        _, H, W = self._masks.shape
        for i in range(self.num_masks):
            mask = self._masks[i].binary().data
            xy = torch.nonzero(mask).flip(-1)
            indices = torch.randint(0, len(xy), (num_point_per_mask,), device=xy.device)
            xy = xy[indices]
            xy += torch.randn_like(xy) * noise
            points.append(xy)
        return np.clip(torch.cat(points, dim=0).cpu().numpy() / np.array([W, H]), 0, 1)

    def remove_not_in_tree(self):
        """remove duplicate after merge new results"""
        if self.verbose > 0:
            print('[Tree2D] remove_not_in_tree')
        self.uncompress()
        if self.cnt == 0:  # empty
            return
        self.node_rearrange()
        self.resize(self.cnt + 1)

    def save(self, filename, compress=True, **kwargs):
        masks: Masks = self._compress() if compress else self._uncompress()
        data = {
            'parent': self.parent,
            'first': self.first,
            'next': self.next,
            'last': self.last,
            'cnt': self.cnt,
            'num_samples': self.num_samples,
            'masks': (masks.data, masks.start, masks.shape, masks.format),
            'scores': self.scores,
            'format': self.format,
            'extra': kwargs
        }
        if filename is not None:
            torch.save(data, filename)
            if self.verbose > 0:
                print(f"[Tree2D]save now Tree2D to {filename}")
        return data

    def load(self, filename: Union[str, Path, None], **data):
        if filename is not None:
            data.update(**torch.load(filename, map_location=self.device))
        data = utils.tensor_to(data, device=self.device)
        self.parent = data.pop('parent')
        self.first = data.pop('first')
        self.next = data.pop('next')
        self.last = data.pop('last')
        self.cnt = data.pop('cnt')
        self._masks = Masks(*data.pop('masks'))
        self.scores = data.pop('scores')
        self.num_samples = data.pop('num_samples')
        self.format = data.pop('format')
        extra = data.pop('extra')
        self._is_compressed = False  # (len(self._masks) != len(self.scores))  # or (self._masks.dtype != torch.bool)
        self.to(self.device)
        if self.verbose > 0:
            print(f'[Tree2D] loaded from file {filename} or input data')
        return extra

    def load_from_png(self, mask_dir: Path, filename: str):
        mask_paths = sorted(list(mask_dir.glob(f"{filename}_level_*.png")))
        self.reset()
        for i, mask_path in enumerate(mask_paths):
            level = int(mask_path.stem.split('_')[-1])
            assert i + 1 == level, f"Except load level {i + 1}, but got {level} "
            mask = utils.load_image(mask_path)
            mask_ids = np.unique(mask)
            if mask_ids[0] == 0:
                mask_ids = mask_ids[1:]
            if len(mask_ids) == 0:
                continue
            masks = mask == mask_ids[:, None, None]
            self.insert_batch(masks)
        self.update_tree()
        self.node_rearrange()
        return self

    def to(self, device):
        super().to(device)
        self._masks = self._masks.to(device) if self._masks is not None else None
        self.scores = self.scores.to(device)
        return self

    def post_process(self, min_area=100, compress=True):
        """
        1. remove disconnected regions and holes in masks with area smaller than min_area
        2. remove overlapped area for the masks in same level
        """
        if self.verbose > 0:
            print(f"[Tree2D] post process min_area={min_area}")
        torch.cuda.synchronize()
        self.uncompress()
        masks = self.masks
        unchanged = True
        for level, nodes in enumerate(self.get_levels()):
            if level == 0:
                assert len(nodes) == 1 and nodes.item() == 0
                continue
            assert nodes.min() > 0
            order = torch.argsort(self.scores[nodes - 1], descending=False)
            # 1. remove disconnected regions and holes in masks with area smaller than min_area
            for i in range(len(nodes)):
                mask = masks[nodes[order[i]] - 1].cpu().numpy()
                mask, changed = remove_small_regions(mask, min_area, mode="holes")
                unchanged = unchanged and not changed
                mask, changed = remove_small_regions(mask, min_area, mode="islands")
                unchanged = unchanged and not changed
                masks[nodes[order[i]].item() - 1] = torch.from_numpy(mask).to(masks)
            # 2. remove overlapped area for the masks in same level
            mask_level = torch.zeros_like(masks[0], dtype=torch.int)
            for i in range(len(nodes)):
                idx = nodes[order[i]]
                mask_i = masks[idx - 1]
                mask_f = masks[self.parent[idx] - 1] if self.parent[idx] > 0 else torch.ones_like(mask_i)
                mask_level[mask_i & mask_f] = idx
            for i in range(len(nodes)):
                masks[nodes[i] - 1] = mask_level == nodes[i]
        self._masks = Masks(masks, format=self._masks.format)
        # re-build segmentation tree
        self.cnt = 0
        self.first[0] = -1
        self.update_tree()
        if compress:
            self.compress()
            self.uncompress()
            self.reset()
            self.update_tree()
            self.remove_not_in_tree()
            self.compress()
        if self.verbose > 0:
            print(f"[Tree2D] complete post process")
        return self

    def _compress(self):
        if self.is_compressed:
            return self._masks
        return self._masks
        # _masks = self._masks.binary().data
        # masks = []
        # for level, nodes in enumerate(self.get_levels()):
        #     if level == 0:
        #         continue
        #     mask = torch.zeros_like(_masks[0], dtype=torch.int)
        #     for i in nodes:
        #         mask[_masks[i - 1]] = i
        #     masks.append(mask)
        # return torch.stack(masks, dim=0) if len(masks) > 0 else None

    def compress(self):
        """save masks use multi-level tensor, all mask in same level do not intersection"""
        if self.is_compressed:
            return self
        if self._masks is None:
            return self
        if self.verbose > 0:
            print(f"[Tree2D] Compress masks")
        self.remove_not_in_tree()
        self._masks = self._compress()
        self._is_compressed = True
        return self

    def _uncompress(self) -> Optional[Tensor]:
        # print('is_compressed:', self.is_compressed)
        if not self.is_compressed:
            return self._masks
        return self._masks
        # masks = self._masks
        # assert masks.max() <= self.num_masks
        # masks_u = torch.zeros((self.num_masks, *masks._shape[1:]), dtype=torch.bool, device=self.device)
        # for level, nodes in enumerate(self.get_levels()):
        #     if level == 0:
        #         continue
        #     assert 1 <= nodes.min() and nodes.max() <= self.num_masks and 1 <= level <= len(masks)
        #     for i in nodes:
        #         masks_u[i - 1] = masks[level - 1] == i
        # return masks_u

    def uncompress(self):
        if not self.is_compressed:
            return self
        if self._masks is None:
            return self
        if self.verbose > 0:
            print(f"[Tree2D] uncompress masks")
        self._masks = self._uncompress()
        self._is_compressed = False
        return self

    @property
    def masks(self) -> Optional[Tensor]:
        if self._masks is None:
            return None
        else:
            self.uncompress()
            return self._masks.binary().data

    # @property
    # def scores(self):
    #     if self.data is None:
    #         return None
    #     return self._scores

    def remove_background(self, background: Tensor, threshold=0.5):
        """The masks belong to background"""
        self.uncompress()
        keep = torch.ones(self.num_masks + 1, device=self.device, dtype=torch.bool)
        for i in range(self.num_masks):
            mask = self._masks[i].binary().data
            area = mask.sum()
            bg_area = (mask & background).sum()
            if (area - bg_area) <= self.min_area or bg_area / area >= threshold:
                keep[i + 1] = False
        if keep.all():
            return
        self.node_rearrange(torch.nonzero(keep)[:, 0])
        self.resize(self.cnt + 1)
        self.reset()
        self.update_tree()
        self.remove_not_in_tree()
        if self.verbose > 0:
            print('[Tree2D] remove backbground')
