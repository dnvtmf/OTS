from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tree_segmentation import TreeData, Tree3D, Tree3Dv2


class TreeSegmentMetric:
    """The Metric to comaper two tree segmentation results"""

    def __init__(self, iou_threshold=0.5):
        self.cnt = 0
        self.sum = 0
        self.iou_threshold = iou_threshold
        self.eps = 1e-7
        self.gt_iou_sum = 0

    @torch.no_grad()
    def update(self, prediction: Union[Tree3D, TreeData, Tree3Dv2], gt: Union[Tree3D, TreeData, Tree3Dv2]):
        if type(gt).__name__.startswith('Tree3D'):
            match, match_iou, matched, matched_iou = self.match_3d(prediction, gt)
        elif isinstance(gt, TreeData):
            match, match_iou, matched, matched_iou = self.match_2d(prediction, gt)
        else:
            raise NotImplementedError(f"prediction: {prediction.__class__.__name__}, gt: {gt.__class__.__name__}")
        num_gt = len(matched)
        TP = (match > 0).sum()
        FP = match.shape[0] - TP
        FN = num_gt - TP
        iou = match_iou.sum()
        PQ = iou / (TP + FP * 0.5 + FN * 0.5)
        self.sum += PQ
        self.cnt += 1
        self.gt_iou_sum += matched_iou.mean()

    def match_3d(self, prediction: Union[Tree3D, Tree3Dv2], gt: Union[Tree3D, Tree3Dv2]):
        # assert type(prediction) == type(gt)
        prediction.node_rearrange()
        N = prediction.cnt
        if isinstance(prediction, Tree3D):
            pred_masks = torch.zeros((N, prediction.num_faces), device=prediction.device, dtype=torch.bool)
            temp = prediction.face_parent[1:].clone()
            for i in range(N, 0, -1):
                pred_masks[i - 1] = temp == i
                temp[pred_masks[i - 1]] = prediction.parent[i].to(temp.dtype)
        else:
            pred_masks = prediction.masks[:N, 1:]
        pred_masks = pred_masks.float()
        pred_area = torch.mv(pred_masks, prediction.area)
        # print('predictions:', pred_masks.shape, pred_masks.dtype, pred_area.shape)

        gt.node_rearrange()
        M = gt.cnt
        if isinstance(gt, Tree3D):
            gt_masks = torch.zeros((M, gt.num_faces), device=gt.device, dtype=torch.bool)
            temp = gt.face_parent[1:].clone()
            for i in range(M, 0, -1):
                gt_masks[i - 1] = temp == i
                temp[gt_masks[i - 1]] = gt.parent[i].to(temp.dtype)
        else:
            gt_masks = gt.masks[:M, 1:]
        gt_masks = gt_masks.float()
        gt_area = torch.mv(gt_masks, gt.area)
        # print('ground_truth:', gt_masks.shape, gt_masks.dtype)
        inter = F.linear(pred_masks, gt_masks * gt.area)
        iou_matrix = inter / (pred_area[:, None] + gt_area[None, :] - inter).clamp_min(1e-7)
        iou_matrix = iou_matrix * (iou_matrix >= self.iou_threshold)
        iou_matrix = iou_matrix.detach().cpu().numpy()
        # get GT match
        pred_idx, gt_idx = linear_sum_assignment(1 - iou_matrix)
        mask = iou_matrix[pred_idx, gt_idx] > 0
        pred_idx, gt_idx = pred_idx[mask], gt_idx[mask]

        match = np.full(N, -1, dtype=np.int32)
        matched = np.full(M, -1, dtype=np.int32)
        match_iou = np.zeros(N, dtype=np.float32)
        matched_iou = np.zeros(M, dtype=np.float32)
        match[pred_idx] = gt_idx
        matched[gt_idx] = pred_idx
        match_iou[pred_idx] = matched_iou[gt_idx] = iou_matrix[pred_idx, gt_idx]
        print(match, match_iou, matched, matched_iou, sep='\n')
        return match, match_iou, matched, matched_iou

    def match_2d(self, prediction: TreeData, gt: TreeData):
        prediction.remove_not_in_tree()
        gt.remove_not_in_tree()
        N = prediction.cnt
        M = gt.cnt
        # print('N:', N, 'M:', M)
        iou_matrix = torch.zeros((N, M), dtype=torch.float, device=gt.device)
        for i in range(N, 0, -1):
            mask_i = prediction.data['masks'][i - 1]
            area_i = mask_i[1:].sum().item()
            for j in range(M, 0, -1):
                mask_j = gt.data['masks'][j - 1]
                area_j = mask_j[1:].sum().item()
                inter = (mask_i & mask_j).sum().item()
                iou = inter / max(area_i + area_j - inter, self.eps)
                iou_matrix[i - 1][j - 1] = iou
        iou_matrix = iou_matrix * (iou_matrix >= self.iou_threshold)
        iou_matrix = iou_matrix.detach().cpu().numpy()
        pred_idx, gt_idx = linear_sum_assignment(1 - iou_matrix)
        mask = iou_matrix[pred_idx, gt_idx] > 0
        pred_idx, gt_idx = pred_idx[mask], gt_idx[mask]

        match = np.full(N, -1, dtype=np.int32)
        matched = np.full(M, -1, dtype=np.int32)
        match_iou = np.zeros(N, dtype=np.float32)
        matched_iou = np.zeros(M, dtype=np.float32)
        match[pred_idx] = gt_idx
        matched[gt_idx] = pred_idx
        match_iou[pred_idx] = matched_iou[gt_idx] = iou_matrix[pred_idx, gt_idx]
        # print(match, match_iou, matched, matched_iou, sep='\n')
        return match, match_iou, matched, matched_iou

    def summarize(self):
        return self.sum / self.cnt
