from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import batched_nms

from semantic_sam import SemanticSAM
from segment_anything.modeling import Sam
from segment_anything.utils.amg import (
    build_point_grid,
    MaskData,
    batch_iterator,
    calculate_stability_score,
    batched_mask_to_box,
    rle_to_mask,
    remove_small_regions,
    mask_to_rle_pytorch,
)
from segment_anything.utils.transforms import ResizeLongestSide
from tree_segmentation.tree_2d_segmentation import TreeData


# noinspection PyAttributeOutsideInit
class TreePredictor:

    def __init__(
        self,
        model: Union[Sam, SemanticSAM],
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        image_size=1024,
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters low quality and duplicate masks.
        The default settings are chosen for SAM with a ViT-H backbone.

        Args:
          model (Sam): The SAM model to use for mask prediction.
          points_per_batch (int): Sets the number of points run simultaneously by the model.
            Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
        """

        self.model = model
        if isinstance(model, Sam):
            self.model_type = "SAM"
            self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        elif isinstance(model, SemanticSAM):
            self.model_type = 'SemanticSAM'
            self.transform = None if image_size < 0 else ResizeLongestSide(image_size)
            self.model.image_format = 'RGB'
            self.model.mask_threshold = 0.0
        else:
            raise ValueError(f"The model must be SAM or SemanticSAM, not {type(model)}")

        self.reset_image()

        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh

    @torch.no_grad()
    def generate(
        self,
        image: np.ndarray,
        points_per_side: Optional[int] = 32,
        points_per_update: int = 256,
        min_mask_region_area: int = 50,
        max_iters=100,
        in_threshold=0.90,
        in_thre_area=10,
        union_threshold=0.10,
        filled_threshold=0.9,
        sample_limit_fn=None,
        device=None,
    ) -> TreeData:
        assert image.shape[-1] == 3
        self.set_image(image)
        init_points = build_point_grid(points_per_side)
        data = self.process_points(init_points)

        tree_data = TreeData(
            data,
            in_threshold=in_threshold,
            union_threshold=union_threshold,
            min_area=min_mask_region_area,
            in_thres_area=in_thre_area,
            device=device
        )
        tree_data.update_tree()
        print('complete init segmentation')

        for step in range(max_iters):
            # points, unfilled_mask = tree_data.sample_unfilled(points_per_update, filled_threshold)
            points = tree_data.sample_by_counts(points_per_update, sample_limit_fn)
            if points is None:
                break
            new_mask_data = self.process_points(points)

            tree_data.cat(new_mask_data)
            tree_data.update_tree()
            tree_data.remove_not_in_tree()
            tree_data.update_tree()

            print(f'complete iter {step} update segmentation')
        self.reset_image()
        # tree_data.to_numpy()

        # Filter small disconnected regions and holes in masks
        # if self.min_mask_region_area > 0:
        #     data = self.postprocess_small_regions(
        #         data,
        #         self.min_mask_region_area,
        #         max(self.box_nms_thresh, self.crop_nms_thresh),
        #     )
        return tree_data

    @torch.no_grad()
    def process_points(self, points: np.ndarray, normalized=True) -> MaskData:
        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points):
            batch_data = self._process_batch(points, self.original_size, normalized)
            data.cat(batch_data)
            del batch_data
        # self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        # data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        # data["points"] = uncrop_points(data["points"], crop_box)
        # data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(self, points: Union[np.ndarray, Tensor], im_size: Tuple[int, ...], normalized=True) -> MaskData:
        # Run model on this batch
        if isinstance(points, Tensor):
            if normalized:
                points = points * points.new_tensor([im_size[1], im_size[0]])
            in_points = self.transform.apply_coords_torch(points, im_size)
        else:
            if normalized:
                points = points * np.array([im_size[1], im_size[0]], dtype=points.dtype)
            transformed_points = self.transform.apply_coords(points, im_size)
            in_points = torch.as_tensor(transformed_points, dtype=torch.float, device=self.device)

        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predict_torch(
            in_points[:, None, :], in_labels[:, None], multimask_output=True, return_logits=True
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        # keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        # if not torch.all(keep_mask):
        #     data.filter(keep_mask)

        return data

    @staticmethod
    def postprocess_small_regions(mask_data: MaskData, min_area: int, nms_thresh: float) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """
        Calculates the image embeddings for the provided image,allowing masks to be predicted with the 'predict' method.

        Args::
            image (np.ndarray): The image for calculating masks. Expects an image in HWC uint8 format,
                with pixel values in [0, 255].
            image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        if self.transform is not None:  # Transform the image to the form expected by the model
            input_image = self.transform.apply_image(image)
        else:
            input_image = image
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(self, transformed_image: torch.Tensor, original_image_size: Tuple[int, ...]) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape 1x3xHxW,
            which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image before transformation, in (H, W) format.
        """
        if self.model_type == 'SAM':
            assert (
                len(transformed_image.shape) == 4 and transformed_image.shape[1] == 3 and
                max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
            ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        else:
            assert transformed_image.ndim == 4 and transformed_image.shape[1] == 3
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        if self.model_type == 'SAM':
            self.features = self.model.image_encoder(input_image)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features = self.model.backbone(input_image)
                mask_features, transformer_encoder_features, multi_scale_features = \
                    self.model.sem_seg_head.pixel_decoder.forward_features(features, None)
                self.features = (multi_scale_features, mask_features)
                self.point_pos_scale = 1. / torch.as_tensor(
                    [input_image.shape[-2], input_image.shape[-1]], device=self.model.device, dtype=torch.float)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (point_labels is not None), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the model.
            Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the point prompts.
            1 indicates a foreground point and 0 indicates a background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the number of masks,
            and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if self.model_type == 'SemanticSAM':
            assert point_coords is not None and point_coords.shape[1] == 1 and torch.all(point_labels)
            assert boxes is None and mask_input is None
            points = point_coords[:, 0, :] * self.point_pos_scale
            points = torch.cat([points, points.new_tensor([[0.005, 0.005]]).repeat(len(points), 1)], dim=-1)
            targets = [{'points': points, 'pb': points.new_tensor([0.] * len(points))}]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, mask_dict = self.model.sem_seg_head.predictor(
                    self.features[0],
                    self.features[1],
                    None,
                    targets=targets,
                    target_queries=None,
                    target_vlp=None,
                    task='demo',
                    extra={
                        'part': False,
                        'whole': False,
                        'seg': True,
                        'det': True
                    }
                )
            iou_predictions = outputs["pred_ious"][0]
            low_res_masks = outputs["pred_masks"][0]
            low_res_masks = low_res_masks.view(*iou_predictions.shape, *low_res_masks.shape[-2:])
            # print('iou_predictions, low_res_masks:', utils.show_shape(iou_predictions, low_res_masks))
        else:  # model_type == "SAM"
            if point_coords is not None:
                points = (point_coords, point_labels)
            else:
                points = None

            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points, boxes=boxes, masks=mask_input
            )

            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            # print('iou_predictions, low_res_masks:', utils.show_shape(iou_predictions, low_res_masks))

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) to generate an embedding.")
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
        self.point_pos_scale = None
