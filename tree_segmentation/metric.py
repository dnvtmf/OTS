import math
from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tree_segmentation import TreeData, Tree3D, Tree3Dv2


class TreeSegmentMetric:
    """The Metric to compare two tree segmentation results"""

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.eps = 1e-7
        self.cnt = 0
        self.PQ_sum = 0  # panoptic quality
        self.SQ_sum = 0  # segmentation quality
        self.RQ_sum = 0  # recognition quality
        self.TQ_sum = 0  # tree quality
        self.gt_iou_sum = 0

    @torch.no_grad()
    def update(self, prediction: Union[Tree3D, TreeData, Tree3Dv2], gt: Union[Tree3D, TreeData, Tree3Dv2]):
        if type(gt).__name__.startswith('Tree3D'):
            IoU, indices_pd, indices_gt = self.calc_IoU_3d(prediction, gt)
        elif isinstance(gt, TreeData):
            IoU, indices_pd, indices_gt = self.calc_IoU_2d(prediction, gt)
        else:
            raise NotImplementedError(f"prediction: {prediction.__class__.__name__}, gt: {gt.__class__.__name__}")
        N, M = IoU.shape
        # get TQ
        matched_iou, matched = IoU.max(dim=1)
        iou_ = torch.zeros_like(IoU)
        iou_[torch.arange(N, device=iou_.device), matched] = matched_iou * (matched_iou >= self.iou_threshold)
        iou_ = torch.cummax(iou_, dim=0)[0]
        TP = (iou_ > 0).sum(dim=1)
        FP = torch.arange(N, device=TP.device, dtype=TP.dtype) + 1 - TP
        FN = M - TP
        TQ = iou_.sum(dim=1) / (TP + FP * 0.5 + FN * 0.5)
        self.TQ_sum += TQ.mean().item()

        # get PQ
        IoU = IoU * (IoU >= self.iou_threshold)
        IoU = IoU.detach().cpu().numpy()
        pred_idx, gt_idx = linear_sum_assignment(1 - IoU)
        mask = IoU[pred_idx, gt_idx] >= self.iou_threshold
        pred_idx, gt_idx = pred_idx[mask], gt_idx[mask]
        TP = len(mask)
        FP = N - TP
        FN = M - TP
        iou = IoU[pred_idx, gt_idx].sum()
        self.SQ_sum += iou / TP
        self.RQ_sum += TP / (TP + FP * 0.5 + FN * 0.5)
        self.PQ_sum += iou / (TP + FP * 0.5 + FN * 0.5)
        self.cnt += 1
        self.gt_iou_sum += matched_iou.mean()

        # output match result
        assert 0 <= pred_idx.min() and pred_idx.max() < len(indices_pd)
        assert 0 <= gt_idx.min() and gt_idx.max() < len(indices_gt)
        pred_idx = indices_pd.cpu().numpy()[pred_idx]
        gt_idx = indices_gt.cpu().numpy()[gt_idx]
        match = np.full(prediction.cnt + 1, -1, dtype=np.int32)
        matched = np.full(gt.cnt + 1, -1, dtype=np.int32)
        match_iou = np.zeros(prediction.cnt + 1, dtype=np.float32)
        matched_iou = np.zeros(gt.cnt + 1, dtype=np.float32)
        match[pred_idx] = gt_idx
        matched[gt_idx] = pred_idx
        match_iou[pred_idx] = matched_iou[gt_idx] = IoU[pred_idx, gt_idx]
        # print(match, match_iou, matched, matched_iou, sep='\n')
        return match, match_iou, matched, matched_iou

    def calc_IoU_3d(self, prediction, gt):
        # type: (Union[Tree3D, Tree3Dv2],  Union[Tree3D, Tree3Dv2])-> Tuple[Tensor, Tensor, Tensor]
        # get prediction masks
        indices_pd = torch.cat(prediction.get_levels(), dim=0)[1:] - 1  # do not care root
        N = len(indices_pd)
        if isinstance(prediction, Tree3D):
            masks_pd = torch.zeros((N, prediction.num_faces), device=prediction.device, dtype=torch.bool)
            temp = prediction.face_parent[1:].clone()
            for i in range(N, 0, -1):
                idx = indices_pd[i] + 1
                masks_pd[i - 1] = temp == idx
                temp[masks_pd[i - 1]] = prediction.parent[idx].to(temp.dtype)
        else:
            masks_pd = prediction.masks[indices_pd, 1:]
        order = torch.argsort(prediction.scores[indices_pd], descending=True)
        indices_pd = indices_pd[order]
        masks_pd = masks_pd[order].float()

        # get ground-truth masks
        indices_gt = torch.cat(gt.get_levels(), dim=0)[1:] - 1  # do not care root
        M = len(indices_gt)
        if isinstance(gt, Tree3D):
            masks_gt = torch.zeros((M, gt.num_faces), device=gt.device, dtype=torch.bool)
            temp = gt.face_parent[1:].clone()
            for i in range(M, 0, -1):
                idx = indices_gt[i] + 1
                masks_gt[i - 1] = temp == idx
                temp[masks_gt[i - 1]] = gt.parent[idx].to(temp.dtype)
        else:
            masks_gt = gt.masks[indices_gt, 1:]
        masks_gt = masks_gt.float()

        # calc IoU matrix
        area_pd = torch.mv(masks_pd, gt.area)
        area_gt = torch.mv(masks_gt, gt.area)
        inter = F.linear(masks_pd, masks_gt * gt.area)
        IoU = inter / (area_pd[:, None] + area_gt[None, :] - inter).clamp_min(self.eps)
        return IoU, indices_pd, indices_gt

    def calc_IoU_2d(self, prediction: TreeData, gt: TreeData) -> Tuple[Tensor, Tensor, Tensor]:
        indices_pd = torch.cat(prediction.get_levels(), dim=0)[1:] - 1  # do not care root
        order = torch.argsort(prediction.data['iou_preds'][indices_pd], descending=True)  # sorted by score
        indices_pd = indices_pd[order]
        masks_pd = prediction.data['masks'][indices_pd].flatten(1, )

        indices_gt = torch.cat(gt.get_levels(), dim=0)[1:] - 1  # do not care root
        masks_gt = gt.data['masks'][indices_gt].flatten(1, )
        # get IoU
        area_pd = masks_pd.sum(dim=1)
        area_gt = masks_gt.sum(dim=1)
        interscection = F.linear(masks_pd.float(), masks_gt.float())
        IoU = interscection / (area_pd[:, None] + area_gt[None, :] - interscection).clamp_min(self.eps)
        return IoU, indices_pd, indices_gt

    @property
    def PQ(self):
        return self.PQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def SQ(self):
        return self.SQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def RQ(self):
        return self.RQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def TQ(self):
        return self.TQ_sum / self.cnt if self.cnt > 0 else math.nan
