import math
from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tree_segmentation import Tree2D, Tree3D, Tree3Dv2, TreeStructure


class TreeSegmentMetric:
    """The Metric to compare two tree segmentation results"""

    def __init__(self, iou_threshold=0.5, is_resize_2d_as_gt=True):
        self.iou_threshold = iou_threshold
        self.eps = 1e-7
        self.cnt = 0
        self.PQ_sum = 0  # panoptic quality
        self.SQ_sum = 0  # segmentation quality
        self.RQ_sum = 0  # recognition quality
        self.TQ_sum = 0  # tree quality
        self.TS_sum = 0  # tree structure quality
        self.mPQ_sum = 0  # panoptic quality
        self.mSQ_sum = 0  # segmentation quality
        self.mRQ_sum = 0  # recognition quality
        self.mTQ_sum = 0  # tree quality
        self.mTS_sum = 0  # tree structure quality
        self.maxIoU_sum = 0  # for each gt, choose the best IoU in prediction
        self.is_resize_2d_as_gt = is_resize_2d_as_gt

    @torch.no_grad()
    def update(self,
               prediction: Union[Tree3D, Tree2D, Tree3Dv2],
               gt: Union[Tree3D, Tree2D, Tree3Dv2],
               return_match=False):
        if type(gt).__name__.startswith('Tree3D'):
            IoU, indices_pd, indices_gt = self.calc_IoU_3d(prediction, gt)
        elif isinstance(gt, Tree2D):
            IoU, indices_pd, indices_gt = self.calc_IoU_2d(prediction, gt)
        else:
            raise NotImplementedError(f"prediction: {prediction.__class__.__name__}, gt: {gt.__class__.__name__}")
        N, M = IoU.shape
        TS = self.calc_tree_structure_score(prediction, indices_pd)
        # get TQ
        matched_iou, matched = IoU.max(dim=1)
        iou_ = torch.zeros_like(IoU)
        iou_[torch.arange(N, device=iou_.device), matched] = matched_iou * (matched_iou >= self.iou_threshold)
        iou_ = torch.cummax(iou_, dim=0)[0]
        # print(TS.shape, iou_.shape)
        TP = (iou_ > 0).sum(dim=1)
        FP = torch.arange(N, device=TP.device, dtype=TP.dtype) + 1 - TP
        FN = M - TP
        iou_ = iou_.sum(dim=1)
        SQ = iou_ / TP.clamp_min(self.eps)  # to avoid divided zero
        RQ = TP / (TP + FP * 0.5 + FN * 0.5).clamp_min(self.eps)
        PQ = iou_ / (TP + FP * 0.5 + FN * 0.5).clamp_min(self.eps)
        TQ = PQ * TS
        self.mSQ_sum += SQ.mean().item()
        self.mRQ_sum += RQ.mean().item()
        self.mPQ_sum += PQ.mean().item()
        self.mTS_sum += TS.mean().item()
        self.mTQ_sum += TQ.mean().item()
        self.TS_sum += TS[-1].item()
        self.TQ_sum += TQ[-1].item()
        # self.SQ_sum += SQ[-1].item()
        # self.RQ_sum += RQ[-1].item()
        # self.PQ_sum += PQ[-1].item()

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
        self.SQ_sum += iou / max(TP, self.eps)
        self.RQ_sum += TP / max(TP + FP * 0.5 + FN * 0.5, self.eps)
        self.PQ_sum += iou / max(TP + FP * 0.5 + FN * 0.5, self.eps)
        self.cnt += 1
        self.maxIoU_sum += matched_iou.mean().item()

        # output match result
        if not return_match:
            return
        assert 0 <= pred_idx.min() and pred_idx.max() < len(indices_pd)
        assert 0 <= gt_idx.min() and gt_idx.max() < len(indices_gt)
        iou = IoU[pred_idx, gt_idx]
        pred_idx = indices_pd.cpu().numpy()[pred_idx]
        gt_idx = indices_gt.cpu().numpy()[gt_idx]
        match = np.full(prediction.cnt + 1, -1, dtype=np.int32)
        matched = np.full(gt.cnt + 1, -1, dtype=np.int32)
        match_iou = np.zeros(prediction.cnt + 1, dtype=np.float32)
        matched_iou = np.zeros(gt.cnt + 1, dtype=np.float32)
        match[pred_idx] = gt_idx
        matched[gt_idx] = pred_idx
        match_iou[pred_idx] = matched_iou[gt_idx] = iou
        # print(match, match_iou, matched, matched_iou, sep='\n')
        return match, match_iou, matched, matched_iou

    def calc_IoU_3d(self, prediction, gt):
        # type: (Union[Tree3D, Tree3Dv2],  Union[Tree3D, Tree3Dv2])-> Tuple[Tensor, Tensor, Tensor]
        # get prediction masks
        assert prediction.num_faces == gt.num_faces, f"Only can compare the results of same mesh"
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

    def calc_IoU_2d(self, prediction: Tree2D, gt: Tree2D) -> Tuple[Tensor, Tensor, Tensor]:
        indices_pd = torch.cat(prediction.get_levels(), dim=0)[1:] - 1  # do not care root
        order = torch.argsort(prediction.scores[indices_pd], descending=True)  # sorted by score
        indices_pd = indices_pd[order]
        masks_pd = prediction.masks[indices_pd]

        indices_gt = torch.cat(gt.get_levels(), dim=0)[1:] - 1  # do not care root
        masks_gt = gt.masks[indices_gt]
        if masks_pd.shape[1:3] != masks_gt.shape[1:3]:
            if self.is_resize_2d_as_gt:
                masks_pd = F.interpolate(masks_pd[None].float(), masks_gt.shape[1:3], mode='nearest')[0]
            else:
                masks_gt = F.interpolate(masks_gt[None].float(), masks_pd.shape[1:3], mode='nearest')[0]
        masks_gt, masks_pd = masks_gt.flatten(1,), masks_pd.flatten(1,)
        # get IoU
        area_pd = masks_pd.sum(dim=1)
        area_gt = masks_gt.sum(dim=1)
        interscection = F.linear(masks_pd.float(), masks_gt.float())
        IoU = interscection / (area_pd[:, None] + area_gt[None, :] - interscection).clamp_min(self.eps)
        return IoU, indices_pd, indices_gt

    def calc_tree_structure_score(self, p: Union[Tree2D, Tree3D, Tree3Dv2], indices: Tensor = None):
        if indices is None:
            indices = torch.cat(p.get_levels(), dim=0)[1:] - 1  # do not care root
            order = torch.argsort(p.scores[indices], descending=True)  # sorted by score
            indices = indices[order]
        M = len(indices)
        if isinstance(p, Tree2D):
            masks = p.masks[indices].flatten(1).float()
            areas = masks.sum(dim=1)
            inter = F.linear(masks, masks)
        else:
            masks = p.masks[indices, 1:].float()
            areas = torch.mv(masks, p.area)
            inter = F.linear(masks, masks * p.area)
        IoU = inter / (areas[:, None] + areas[None, :] - inter).clamp_min(self.eps)
        In = inter / (areas[:, None]).clamp_min(self.eps)

        # x, y = torch.triu_indices(M, M, 1, device=IoU.device)
        # score_1 = torch.maximum(1 - IoU[x, y], torch.maximum(inter[x, y] / areas[x], inter[x, y] / areas[y]))
        # for t in range(M):
        #     score_1 = None

        tree = TreeStructure(p.cnt + 1, device=p.device)
        tree.parent = p.parent.clone()
        tree.first = p.first.clone()
        tree.last = p.last.clone()
        tree.next = p.next.clone()
        tree.cnt = p.cnt
        tree.node_rearrange(torch.cat([indices.new_zeros(1), indices + 1], dim=0))
        # tree.print_tree()

        # TODO: remove check
        cmp = [None] * M

        def check(u=0):
            s = 0
            mask_u = p.masks[indices[u - 1]] if u > 0 else torch.ones_like(p.masks[0])
            for c in tree.get_children(u):
                mask_c = p.masks[indices[c - 1]]
                s += (mask_u * mask_c).sum() / mask_c.sum().clamp_min(self.eps)
                # cmp[c - 1] = (mask_u * mask_c).sum() / mask_c.sum().clamp_min(self.eps)
                total = 0
                cnt = 0
                for v in tree.get_children(u):
                    if v == c:
                        continue
                    mask_v = p.masks[indices[v - 1]]
                    inter_ = (mask_v * mask_c).sum()
                    iou = inter_ / (mask_c.sum() + mask_v.sum() - inter_).clamp_min(self.eps)
                    assert (IoU[c - 1, v - 1] - iou).abs() < 1e-5, f"{IoU[c-1, v-1]} vs. {iou}"
                    total += 1 - iou
                    cnt += 1
                s += 1 if cnt == 0 else total / cnt
                cmp[c - 1] = 1 if cnt == 0 else (total / cnt).item()
                s += check(c)
            return s

        temp = torch.zeros(M, device=p.device)
        scores = torch.zeros(M, device=p.device)
        now = 0
        for i in range(M):
            pi = tree.parent[i + 1]
            now += 1 if pi <= 0 else In[i, pi - 1]
            disjoint = [1 - IoU[i, c - 1] for c in tree.get_children(pi) if c - 1 != i]
            temp[i] = 1 if len(disjoint) == 0 else sum(disjoint) / len(disjoint)
            now += temp[i]
        # print(now, check())
        # print([x - y for x, y in zip(cmp, temp.tolist()) if abs(x - y) > 1e-6])
        # assert abs(now.item() - check().item()) < 1e-4, f"{abs(now.item() - check().item())}"
        scores[M - 1] = now / M
        for t in range(M - 1, 0, -1):
            pi = tree.parent[t + 1]
            children = torch.tensor(tree.get_children(t + 1), dtype=torch.long, device=tree.device) - 1
            now += (In[children, pi - 1].sum() if pi > 0 else len(children)) - In[children, t].sum()
            tree.node_delete(t + 1, move_children=True)
            children = torch.tensor(tree.get_children(pi), dtype=torch.long, device=tree.device) - 1
            now -= 1 if pi <= 0 else In[t, pi - 1]
            now -= temp[t]
            if len(children) == 1:
                now -= temp[children].sum()
                temp[children] = 1
                now += 1
            elif len(children) > 1:
                now -= temp[children].sum()
                temp[children] = (1 - IoU[children, :][:, children]).sum(dim=1) / (len(children) - 1)
                now += temp[children].sum()
            scores[t - 1] = now / t
            # check_score = check()
            # print([x - y for x, y in zip(cmp, temp.tolist()) if abs(x - y) > 1e-6])
            # assert abs(now.item() - check_score.item()) < 1e-4, f"{abs(now.item() - check_score.item())}"
        return scores * 0.5

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
    def TS(self):
        return self.TS_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def TQ(self):
        return self.TQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def mPQ(self):
        return self.mPQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def mSQ(self):
        return self.mSQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def mRQ(self):
        return self.mRQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def mTS(self):
        return self.mTS_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def mTQ(self):
        return self.mTQ_sum / self.cnt if self.cnt > 0 else math.nan

    @property
    def mIoU(self):
        return self.maxIoU_sum / self.cnt if self.cnt > 0 else math.nan

    def summarize(self):
        return {
            'SQ': self.SQ,
            'RQ': self.RQ,
            'PQ': self.PQ,
            'TQ': self.TQ,
            'TS': self.TS,
            'mSQ': self.mSQ,
            'mRQ': self.mRQ,
            'mPQ': self.mPQ,
            'mTQ': self.mTQ,
            'mTS': self.mTS,
            'mIoU': self.mIoU,
        }