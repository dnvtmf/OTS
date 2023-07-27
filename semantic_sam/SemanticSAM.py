"""
paper: Semantic-SAM: Segment and Recognize Anything at Any Granularity
code: https://github.com/UX-Decoder/Semantic-SAM
"""
import random
from typing import Tuple, Any
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from kornia.contrib import distance_transform

from extension.utils import retry_if_cuda_oom, convert_pth, make_divisible
from .mask_dino import IMaskDINOHead
from .base import BackboneBase

ImageList = None

__all__ = ['SemanticSAM', 'semantic_sam_l', 'semantic_sam_t']


# noinspection PyDefaultArgument
class SemanticSAM(nn.Module):
    def __init__(
        self,
        backbone: BackboneBase,
        dim_proj=512,
        num_classes=1,
        # criterion_switch: dict={'mo1': criterion_mo1, 'm2m': criterion_m2m},
        num_queries: int = 0,
        object_mask_threshold: float = 0.25,
        overlap_threshold: float = 0.8,
        # metadata,
        size_divisibility: int = 32,
        sem_seg_postprocess_before_inference: bool = True,
        pixel_mean: Tuple[float] = (123.675, 116.280, 103.530),
        pixel_std: Tuple[float] = (58.395, 57.120, 57.375),
        # inference
        semantic_on: bool = True,
        panoptic_on: bool = True,
        instance_on: bool = True,
        test_topk_per_image: int = 100,
        # data_loader: str=None,
        pano_temp: float = 0.06,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        # train_dataset_name: str,
        coco_mask_on=True,
        sam_on: bool = True,
        regenerate_point: bool = False,
        num_mask_tokens: int = 3,
        max_num_instance: int = 100,
        classification_on: bool = False,
        many_to_one: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = IMaskDINOHead(channels=backbone.channels, dim_proj=dim_proj, num_classes=num_classes)
        self.pano_temp = pano_temp

        self.criterion = None
        self.criterion_switch = self.get_criterion_switch()
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        # self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility  # use backbone size_divisibility if not set
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        # self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        self.train_class_names = dict()
        # self.train_dataset_name = train_dataset_name
        self.coco_mask_on = coco_mask_on
        self.classification_on = classification_on
        self.task_switch = {'sam': sam_on}

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.max_num_instance = max_num_instance
        self.num_mask_tokens = num_mask_tokens
        self.regenerate_point = regenerate_point
        self.many_to_one = many_to_one

    def get_criterion_switch(self):
        criterion_mo1 = None
        # SetCriterion(enc_cfg['NUM_CLASSES'],
        #     matcher=matcher,
        #     weight_dict=weight_dict,
        #     eos_coef=no_object_weight,
        #     losses=losses,
        #     num_points=dec_cfg['TRAIN_NUM_POINTS'],
        #     oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
        #     importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
        #     dn=dec_cfg['DN'],
        #     dn_losses=dn_losses,
        #     panoptic_on=dec_cfg['PANO_BOX_LOSS'],
        #     semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and
        #                      not dec_cfg['TEST']['PANOPTIC_ON'],
        #     num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3))
        # this is for many-to-many match, we use it in training sa-1b
        criterion_m2m = None
        # SetCriterionOsPartWholeM2M(enc_cfg['NUM_CLASSES'],
        # matcher=matcher,
        # weight_dict=weight_dict,
        # eos_coef=no_object_weight,
        # losses=losses,
        # num_points=dec_cfg['TRAIN_NUM_POINTS'],
        # oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
        # importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
        # dn=dec_cfg['DN'],
        # dn_losses=dn_losses,
        # panoptic_on=dec_cfg['PANO_BOX_LOSS'],
        # semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and
        #                  dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST']['PANOPTIC_ON'],
        # num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3))

        criterion_switch = {'mo1': criterion_mo1, 'm2m': criterion_m2m}
        return criterion_switch

    @property
    def device(self):
        return self.pixel_mean.device

    def evaluate_demo(
        self,
        batched_inputs,
        all_whole,
        all_parts,
        mask_features=None,
        multi_scale_features=None,
        return_features=False
    ):
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        targets = batched_inputs[0]['targets']
        height = images[0].shape[1]
        width = images[0].shape[2]
        padded_h = images.tensor.shape[-2]  # divisible to 32
        padded_w = images.tensor.shape[-1]
        # import pdb;pdb.set_trace()
        targets[0]['points'] = targets[0]['points'] * torch.as_tensor(
            [width, height, width, height], dtype=torch.float, device=self.device) / torch.as_tensor(
            [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)

        if mask_features is None or multi_scale_features is None:
            features = self.backbone(images.tensor)
            mask_features, transformer_encoder_features, multi_scale_features = \
                self.sem_seg_head.pixel_decoder.forward_features(features, None)
        outputs, mask_dict = self.sem_seg_head.predictor(multi_scale_features, mask_features, None, targets=targets,
            target_queries=None, target_vlp=None, task='demo', extra=prediction_switch)
        # mask_box_results = outputs["pred_boxes"]
        pred_ious = None
        if 'pred_ious' in outputs.keys():
            pred_ious = outputs["pred_ious"]

        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        pred_masks = mask_pred_results[0]

        image_size = images.image_sizes[0]

        # height = input_per_image.get("height", image_size[0])
        height = image_size[0]
        # width = input_per_image.get("width", image_size[1])
        width = image_size[1]
        # new_size = (images.tensor.shape[-2], images.tensor.shape[-1])  # padded size (divisible to 32)
        if self.sem_seg_postprocess_before_inference:
            pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(pred_masks, image_size, height, width)
        if return_features:
            return pred_masks, pred_ious, mask_features, multi_scale_features
        else:
            return pred_masks, pred_ious

    def forward(self, batched_inputs, inference_task='seg'):
        """
        forward for all data, including sa-1b, generic seg, part seg data
        currently only support interactive segmentation on sa-1b, stay tuned for more code!
        """
        if self.training:
            losses = {}
            if self.task_switch['sam']:
                prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
                if self.many_to_one:
                    # this is to reproduce sam training
                    self.criterion = self.criterion_switch['mo1']
                else:
                    # this is using our own many2many training for better performance
                    self.criterion = self.criterion_switch['m2m']
                self.criterion.num_classes = 1
                data = batched_inputs if type(batched_inputs) == list else batched_inputs['sam']
                losses_sam = self.forward_seg(data, task='sam', prediction_switch=prediction_switch)
                new_losses_sam = {}
                for key, value in losses_sam.items():
                    new_losses_sam['sam.' + str(key)] = losses_sam[key]
                losses.update(new_losses_sam)
            return losses
        else:
            prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
            if inference_task == 'interactive':
                processed_results = self.evaluate_interactive(batched_inputs,
                    task=inference_task,
                    prediction_switch=prediction_switch)
            elif inference_task == 'multi_granularity':
                processed_results = self.evaluate_interactive_granularity(batched_inputs,
                    task=inference_task,
                    prediction_switch=prediction_switch)
            else:
                raise NotImplementedError
            return processed_results

    def forward_seg(
        self,
        batched_inputs,
        task='seg',
        prediction_switch={
            'part': True,
            'whole': True,
            'seg': True,
            'det': True
        }
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets_interactive(gt_instances, images, prediction_switch=prediction_switch)
            else:
                targets = None
                print("empty targets", targets, task)

            outputs, mask_dict = self.sem_seg_head(features, targets=targets, task=task, extra=prediction_switch)
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, mask_dict, task=task, extra=prediction_switch)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses

    def prepare_targets_interactive(self, targets, images, prediction_switch, task='seg'):
        """
        prepare targets for interactive segmentation, mainly includes:
            box:
            mask:
            labels: part / instance
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        box_start = random.randint(int((self.max_num_instance - 1) / 2),
            self.max_num_instance - 1)  # box based interactive after this number; about 1/4
        for targets_per_image in targets:
            gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(
                targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
            # pad gt
            h, w = targets_per_image.image_size
            if not self.training:
                h_pad, w_pad = h, w

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_masks = targets_per_image.gt_masks if torch.is_tensor(
                targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor
            if not self.training:
                max_num_instance_ori = self.max_num_instance
                self.max_num_instance = len(gt_masks)
                box_start = self.max_num_instance  # FIXME all points evaluation
            if len(gt_masks) == 0:
                new_targets.append({
                    'boxes': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'points': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'boxes_dn': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    "pb": torch.cat([torch.zeros(self.max_num_instance - box_start),
                                     torch.ones(box_start)], 0),
                    'box_start': box_start
                })
                if not self.training:
                    self.max_num_instance = max_num_instance_ori
                continue
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            num_mask = targets_per_image.gt_classes.shape[0]

            index = torch.randperm(num_mask)
            if num_mask == 0:
                print("wrong empty image! argets_per_image.gt_classes.shape[0] ", targets_per_image.gt_classes.shape[0],
                    "targets_per_image", targets_per_image)
            if self.max_num_instance > num_mask:
                rep = 0 if num_mask == 0 else int(self.max_num_instance / num_mask) + 1
                index = index.repeat(rep)
            index = index[:self.max_num_instance]
            box_start = self.max_num_instance
            level_target_inds = []
            # randomly sample one point as the user input
            if self.regenerate_point and box_start > 0:
                point_coords = []
                for i in range(box_start):
                    mask = gt_masks[index[i]].clone()
                    center_point = True  # for evaluation sample the center as clicks
                    if not self.training and center_point:
                        mask = mask[None, None, :]
                        n, _, h, w = mask.shape
                        mask_dt = (distance_transform(
                            (~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:, :, 1:-1, 1:-1])
                        # selected_index = torch.stack([torch.arange(n*_), mask_dt.max(dim=-1)[1].cpu()]).tolist()
                        selected_point = torch.tensor([mask_dt.argmax() / w,
                                                       mask_dt.argmax() % w]).long().cuda().flip(0)
                    else:
                        candidate_indices = mask.nonzero()
                        if len(candidate_indices) == 0:
                            print('wrong')
                            selected_point = torch.tensor([0, 0]).cuda()
                        else:
                            selected_index = random.randint(0, len(candidate_indices) - 1)
                            selected_point = candidate_indices[selected_index].flip(0)
                        # only build level targets for sam data
                        if not prediction_switch['whole'] and not prediction_switch['part']:
                            level_target_ind = []
                            for ind, m in enumerate(gt_masks):
                                if m[tuple(selected_point.flip(0))]:
                                    level_target_ind.append(ind)
                            assert len(level_target_ind) > 0, "each point must have at least one target"
                            # randomly sample some target index if targets exceeds the maximum tokens
                            # FIXME another way is to filter small objects when too many level targets
                            if len(level_target_ind) > self.num_mask_tokens:
                                random.shuffle(level_target_ind)
                                level_target_ind = level_target_ind[:self.num_mask_tokens]
                            level_target_inds.append(level_target_ind)
                    selected_point = torch.cat([selected_point - 3, selected_point + 3], 0)
                    point_coords.append(selected_point)
                point_coords = torch.stack(point_coords).to('cuda')
            else:
                point_coords = targets_per_image.gt_boxes.tensor[index[:box_start]]
            max_num_tgt_per_click = -1
            if len(level_target_inds) > 0:
                num_tgt = [len(l) for l in level_target_inds]
                max_num_tgt_per_click = max(num_tgt)
                if max_num_tgt_per_click > 5:
                    print("max number of levels ", max(num_tgt))
            new_target = {
                "ori_mask_num":
                    len(targets_per_image.gt_classes),
                "level_target_inds":
                    level_target_inds,
                "max_num_tgt_per_click":
                    max_num_tgt_per_click,
                "labels":
                    targets_per_image.gt_classes[index] if prediction_switch['whole'] else None,
                "masks":
                    padded_masks[index],
                "ori_masks":
                    padded_masks,
                "boxes":
                    box_xyxy_to_cxcywh(gt_boxes[index]) / image_size_xyxy,
                "ori_boxes":
                    box_xyxy_to_cxcywh(gt_boxes) / image_size_xyxy,
                "points":
                    box_xyxy_to_cxcywh(point_coords) / image_size_xyxy,
                "pb":
                    torch.cat([torch.zeros(self.max_num_instance - box_start),
                               torch.ones(box_start)], 0),
                "gt_whole_classes":
                    targets_per_image.gt_whole_classes[index]
                    if targets_per_image.has('gt_whole_classes') and prediction_switch['whole'] else None,
                "gt_part_classes":
                    targets_per_image.gt_part_classes[index]
                    if targets_per_image.has('gt_part_classes') and prediction_switch['part'] else None,
            }
            # handle coco data format
            if prediction_switch['whole'] and not prediction_switch['part']:
                new_target['gt_whole_classes'] = targets_per_image.gt_classes[index]

            if not self.training:
                # transform targets for inference due to padding
                self.max_num_instance = max_num_instance_ori
                new_target["pb"] = torch.zeros_like(new_target["pb"])
                height = images[0].shape[1]
                width = images[0].shape[2]
                padded_h = images.tensor.shape[-2]  # divisable to 32
                padded_w = images.tensor.shape[-1]
                new_target["boxes_dn_ori"] = torch.cat(
                    [new_target["points"].clone(), new_target["boxes"][box_start:].clone()], 0)
                new_target['points'] = new_target['points'] * torch.as_tensor(
                    [width, height, width, height], dtype=torch.float, device=self.device) / torch.as_tensor(
                    [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
                new_target['boxes'] = new_target['boxes'] * torch.as_tensor(
                    [width, height, width, height], dtype=torch.float, device=self.device) / torch.as_tensor(
                    [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
            new_target["boxes_dn"] = torch.cat([new_target["points"], new_target["boxes"][box_start:]], 0)
            new_target['box_start'] = box_start
            new_targets.append(new_target)

        return new_targets

    def evaluate_interactive(
        self,
        batched_inputs,
        task='seg',
        prediction_switch={
            'part': True,
            'whole': True,
            'seg': True,
            'det': True
        },
        oracle=True
    ):
        """
        evaluate interactive segmentation on other datasets (i.e, coco)
        """
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets_ = self.prepare_targets_interactive(gt_instances, images, prediction_switch=prediction_switch)
        outputs_, mask_dict = self.sem_seg_head(features, targets=targets_, task=task, extra=prediction_switch)
        outputs = {}
        targets = {}
        outputs['point'] = outputs_
        targets['point'] = targets_
        processed_results_all = {}
        for key in outputs.keys():
            num_tokens = self.num_mask_tokens
            all_batch_shape_iou = []
            if 'pred_ious' in outputs[key].keys():
                pred_ious = outputs[key]["pred_ious"]

            mask_pred_results = outputs[key]["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            gt_masks = targets[key][0]["masks"]
            pred_masks = mask_pred_results[0]

            image_size = images.image_sizes[0]
            height = image_size[0]
            width = image_size[1]

            if self.sem_seg_postprocess_before_inference:
                pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(pred_masks, image_size, height, width)

            scores = pred_ious.view(-1, num_tokens)
            score, index = scores.max(1)
            pred_masks_max = torch.gather(
                pred_masks.view(-1, num_tokens, pred_masks.shape[-2], pred_masks.shape[-1]), 1,
                index[:, None, None, None].repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1])).squeeze(1)
            pred_masks_max = pred_masks_max > 0
            all_batch_shape_iou += [get_iou(gt_masks, pred_masks_max)]
            all_batch_shape_iou = torch.stack(all_batch_shape_iou)
            processed_results = [{"mask_iou": all_batch_shape_iou[:, i]} for i in range(len(all_batch_shape_iou[0]))]
            # match with the best prediction for oracle evaluation
            all_batch_shape_iou = []
            gt_masks_repeat = gt_masks.repeat_interleave(num_tokens, 0)
            iou_all = get_iou(gt_masks_repeat, pred_masks > 0)
            selected_ious, index = iou_all.view(-1, num_tokens).max(1)
            all_batch_shape_iou += [selected_ious]
            all_batch_shape_iou = torch.stack(all_batch_shape_iou)
            processed_results = {
                'oracle': [{
                    "mask_iou": all_batch_shape_iou[:, i]
                } for i in range(len(all_batch_shape_iou[0]))],
                'max': processed_results
            }
            processed_results_all = processed_results

        return processed_results_all

    def evaluate_interactive_granularity(
        self,
        batched_inputs,
        task='seg',
        prediction_switch={
            'part': True,
            'whole': True,
            'seg': True,
            'det': True
        },
        oracle=True
    ):
        """
        evaluate multiple granualrity output on a subset of sam
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        ###
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets_ = self.prepare_targets_sam_eval(gt_instances, images, prediction_switch=prediction_switch)
        outputs_, mask_dict = self.sem_seg_head(features, targets=targets_, task=task, extra=prediction_switch)
        outputs = {}
        targets = {}
        outputs['point'] = outputs_
        targets['point'] = targets_
        processed_results_all = {}
        for key in outputs.keys():
            all_batch_shape_iou = []
            mask_pred_results = outputs[key]["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = mask_pred_results[0]
            image_size = images.image_sizes[0]

            height = image_size[0]
            width = image_size[1]

            new_size = (images.tensor.shape[-2], images.tensor.shape[-1])  # padded size (divisible to 32)
            if self.sem_seg_postprocess_before_inference:
                pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(pred_masks, image_size, height, width)

            outputs_without_aux = {k: v for k, v in outputs[key].items() if k != "aux_outputs"}
            match_cost = ["cls", "box", "mask"]
            matcher = self.criterion_switch['m2m'].matcher
            criterion = self.criterion_switch['m2m']
            indices = matcher(outputs_without_aux, targets[key], match_cost, extra=prediction_switch)
            src_idx = criterion._get_src_permutation_idx(indices)
            tgt_idx = criterion._get_tgt_permutation_idx(indices)
            src_masks = pred_masks.unsqueeze(0)
            src_masks = src_masks[src_idx]

            level_target_inds = targets[key][0]['level_target_inds']
            ori_masks = targets[key][0]["ori_masks"].to(src_masks)
            ori_gt_masks = [torch.stack([ori_masks[ind] for inds in level_target_inds for ind in inds])]

            target_masks = ori_gt_masks[0].unsqueeze(0)[tgt_idx]

            all_batch_shape_iou += [get_iou(target_masks > 0, src_masks > 0)]

            all_batch_shape_iou = torch.stack(all_batch_shape_iou)

            processed_results = [{"mask_iou": all_batch_shape_iou[:, i]} for i in range(len(all_batch_shape_iou[0]))]

            processed_results_all = processed_results

        return processed_results_all

    def preprocess(self, x: Tensor) -> Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        if self.size_divisibility > 1:
            padh = make_divisible(h, self.size_divisibility) - h
            padw = make_divisible(w, self.size_divisibility) - w
            x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(self, masks: Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]) -> Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Args:
          masks (Tensor): semantic segmentation prediction logits. A tensor of shape (B, C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
          input_size (tuple(int, int)): image size that segmentor is taking as input, in (H, W) format.
            Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
            semantic segmentation prediction (Tensor): A tensor of the shape (B, C, H, W)
                that contains per-pixel soft predictions, where (H, W) is given by original_size.
        """
        if self.size_divisibility > 1:
            H = make_divisible(input_size[0], self.size_divisibility)
            W = make_divisible(input_size[1], self.size_divisibility)
            masks = F.interpolate(masks, (H, W), mode="bilinear", align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]  # .expand(1, -1, -1, -1)
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bicubic", align_corners=False, antialias=True
    )[0]
    return result


def get_iou(gt_masks, pred_masks, ignore_label=-1):
    rev_ignore_mask = ~(gt_masks == ignore_label)
    gt_masks = gt_masks.bool()
    n, h, w = gt_masks.shape
    intersection = ((gt_masks & pred_masks) & rev_ignore_mask).reshape(n, h * w).sum(dim=-1)
    union = ((gt_masks | pred_masks) & rev_ignore_mask).reshape(n, h * w).sum(dim=-1)
    ious = (intersection / union)
    return ious


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def semantic_sam_l(office_model: Any = ''):
    from .SwinTransformer import SwinL
    backbone = SwinL(variant='base384_22K', drop_path_rate=0.3)
    model = SemanticSAM(backbone)
    if office_model:
        if isinstance(office_model, (str, Path)):
            pth = torch.load(office_model, map_location='cpu')
        else:
            pth = office_model
        pth = convert_pth(pth, replace={
            r'model\.backbone\.norm0\.(.*)': "backbone.norm_4.{0}",
            r'model\.backbone\.norm1\.(.*)': "backbone.norm_8.{0}",
            r'model\.backbone\.norm2\.(.*)': "backbone.norm_16.{0}",
            r'model\.backbone\.norm3\.(.*)': "backbone.norm_32.{0}",
            r'model\.criterion\..*': '',
            r"model\.(.*)": '{0}',
        })
        model.load_state_dict(pth)

    return model


def semantic_sam_t(office_model: Any = ''):
    from .SwinTransformer import SwinT
    backbone = SwinT(drop_path_rate=0.3)
    model = SemanticSAM(backbone)
    if office_model:
        if isinstance(office_model, (str, Path)):
            pth = torch.load(office_model, map_location='cpu')
        else:
            pth = office_model
        pth = convert_pth(pth, replace={
            r'model\.backbone\.norm0\.(.*)': "backbone.norm_4.{0}",
            r'model\.backbone\.norm1\.(.*)': "backbone.norm_8.{0}",
            r'model\.backbone\.norm2\.(.*)': "backbone.norm_16.{0}",
            r'model\.backbone\.norm3\.(.*)': "backbone.norm_32.{0}",
            r'model\.criterion\..*': '',
            r"model\.(.*)": '{0}',
        })
        model.load_state_dict(pth)

    return model


def test():
    m = semantic_sam_l(office_model=Path('~/models/segmentation/Semantic-SAM/swinl_only_sam_many2many.pth'))
    print(m)
