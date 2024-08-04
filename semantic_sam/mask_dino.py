"""
paper: Semantic-SAM: Segment and Recognize Anything at Any Granularity
code: https://github.com/UX-Decoder/Semantic-SAM
"""
from typing import Dict, Optional, Union, Callable, Sequence

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn.functional as F
# from detectron2.layers import Conv2d, get_norm
from torch import nn, autocast

from .utils import trunc_normal_
from .dino_decoder import (
    inverse_sigmoid,
    MLP,
    TransformerDecoder,
    DeformableTransformerDecoderLayer,
    PositionEmbeddingSine,
    MSDeformAttnTransformerEncoderOnly,
)


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": torch.nn.BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            # "naiveSyncBN": NaiveSyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            # "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
        }[norm]
    return norm(out_channels)


# noinspection PyUnusedLocal,PyDefaultArgument
class IMaskDINOHead(nn.Module):

    def __init__(
        self,
        channels: Dict[int, int],
        *,
        num_classes: int = 1,
        dim_proj: int = 512,
        # pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        enc_cfg: dict = None,
        dec_cfg: dict = None
        # transformer_predictor: nn.Module,
    ):
        super().__init__()
        if enc_cfg is None:
            enc_cfg = {}
        if dec_cfg is None:
            dec_cfg = {}
        # channels = sorted(channels.items(), key=lambda x: x[1].stride)
        # self.in_features = [str(stride) for stride, v in channels]
        # self.ignore_value = ignore_value
        # self.common_stride = 4
        # self.loss_weight = loss_weight

        self.pixel_decoder = MaskDINOEncoder(
            channels=channels,
            transformer_in_strides=[8, 16, 32],
            transformer_dropout=0,
            transformer_dim_feedforward=2048,
            **enc_cfg
        )

        self.predictor = IMaskDINODecoder(
            lang_encoder=None,
            in_channels=self.pixel_decoder.conv_dim,
            num_classes=num_classes,
            dim_proj=dim_proj,
            mask_dim=self.pixel_decoder.mask_dim,
            # cfg, transformer_predictor_in_channels, lang_encoder, extra=extra
            mask_classification=True,
            **dec_cfg
        )

        self.num_classes = num_classes
        # store processed features
        self.processed_features = None

    def forward_encoder(
        self,
        features,
        mask=None,
        targets=None,
        target_queries=None,
        target_vlp=None,
        prediction_switch=None,
        task='seg',
        extra={}
    ):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features, mask
        )
        self.processed_features = (mask_features, transformer_encoder_features, multi_scale_features)

    def forward_decoder(
        self,
        features,
        mask=None,
        targets=None,
        target_queries=None,
        target_vlp=None,
        prediction_switch=None,
        task='seg',
        extra={}
    ):
        assert self.processed_features is not None, "need to precess features first"
        mask_features, transformer_encoder_features, multi_scale_features = self.processed_features
        if task == 'teacher':
            predictions = self.predictor.forward_teacher(
                multi_scale_features,
                mask_features,
                mask,
                targets=targets,
                target_queries=target_queries,
                target_vlp=target_vlp,
                task=task,
                extra=extra
            )
        else:
            predictions = self.predictor(
                multi_scale_features,
                mask_features,
                mask,
                targets=targets,
                target_queries=target_queries,
                target_vlp=target_vlp,
                task=task,
                extra=extra
            )
        return predictions

    def forward(self, features, mask=None, targets=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        return self.layers(
            features,
            mask,
            targets=targets,
            target_queries=target_queries,
            target_vlp=target_vlp,
            task=task,
            extra=extra
        )

    def layers(
        self,
        features,
        mask=None,
        targets=None,
        target_queries=None,
        target_vlp=None,
        prediction_switch=None,
        task='seg',
        extra={}
    ):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features, mask
        )
        predictions = self.predictor(
            multi_scale_features,
            mask_features,
            mask,
            targets=targets,
            target_queries=target_queries,
            target_vlp=target_vlp,
            task=task,
            extra=extra
        )
        return predictions


class MaskDINOEncoder(nn.Module):
    """
    This is the multi-scale encoder in detection models, also named as pixel decoder in segmentation models.
    """

    def __init__(
        self,
        channels: Dict[int, int],
        *,
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: Optional[Union[str, Callable]] = 'GN',
        # deformable transformer encoder args
        transformer_in_strides: Sequence[int] = (8, 16, 32),
        common_stride: int = 4,
        num_feature_levels: int = 3,
        total_num_feature_levels: int = 4,
        feature_order: str = 'low2high',
        use_ckpt=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            channels: shapes {stride:channels} of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
            num_feature_levels: feature scales used
            total_num_feature_levels: total feautre scales used (include the downsampled features)
            feature_order: 'low2high' or 'high2low', i.e., 'low2high' means low-resolution features are put in the first.
        """
        super().__init__()
        # TODO: 修改为Neck
        self.conv_dim = conv_dim
        self.use_ckpt = use_ckpt
        transformer_input_shape = {k: v for k, v in channels.items() if k in transformer_in_strides}
        # this is the input shape of pixel decoder
        input_shape = sorted(channels.items(), key=lambda x: x[0])
        self.in_features = [str(k) for k, v in input_shape]  # starting from "4" to "32"
        self.feature_strides = [k for k, v in input_shape]
        self.feature_channels = [v for k, v in input_shape]
        self.feature_order = feature_order

        if feature_order == "low2high":
            transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: -x[0])
        else:
            transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[0])
        self.transformer_in_features = [str(k) for k, v in transformer_input_shape]  # starting from "4" to "32"
        transformer_in_channels = [v for k, v in transformer_input_shape]
        self.transformer_feature_strides = [k for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.maskdino_num_feature_levels = num_feature_levels  # always use 3 scales
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.low_resolution_index = transformer_in_channels.index(max(transformer_in_channels))
        self.high_resolution_index = 0 if self.feature_order == 'low2high' else -1
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                )
            # input projection for downsample
            in_channels = max(transformer_in_channels)
            for _ in range(self.total_num_feature_levels - self.transformer_num_feature_levels):  # exclude the res2
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                )
                in_channels = conv_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )
            ])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.total_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = max(int(np.log2(stride) - np.log2(self.common_stride)), 1)

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @autocast(device_type='cuda', enabled=False)
    def forward_features(self, features, masks):
        """
        Args:
            features: multi-scale features from the backbone
            masks: image mask
        Return:
            enhanced multi-scale features and mask feature (1/4 resolution) for the decoder to produce binary mask
        """
        # backbone features
        srcs = []
        pos = []
        # additional downsampled features
        srcsl = []
        posl = []
        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            smallest_feat = features[self.transformer_in_features[self.low_resolution_index]].float()
            _len_srcs = self.transformer_num_feature_levels
            for i in range(_len_srcs, self.total_num_feature_levels):
                if i == _len_srcs:
                    src = self.input_proj[i](smallest_feat)
                else:
                    src = self.input_proj[i](srcsl[-1])
                srcsl.append(src)
                posl.append(self.pe_layer(src))
        srcsl = srcsl[::-1]
        # Reverse feature maps
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        srcs.extend(srcsl) if self.feature_order == 'low2high' else srcsl.extend(srcs)
        pos.extend(posl) if self.feature_order == 'low2high' else posl.extend(pos)
        if self.feature_order != 'low2high':
            srcs = srcsl
            pos = posl
        y, spatial_shapes, level_start_index = self.transformer(srcs, masks, pos, use_ckpt=self.use_ckpt)
        bs = y.shape[0]

        split_size_or_sections = [0] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(
                out[self.high_resolution_index], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False
            )
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.total_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features


# noinspection PyDefaultArgument
class IMaskDINODecoder(nn.Module):

    def __init__(
        self,
        lang_encoder: Optional[nn.Module],
        in_channels,
        num_classes: int,
        mask_dim: int,
        dim_proj: int,
        *,
        mask_classification=True,
        hidden_dim: int = 256,
        num_queries: int = 0,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,  # 9 decoder layers, add one for the loss on learnable query
        enforce_input_project: bool = False,
        two_stage: bool = False,
        dn: str = 'seg',
        noise_scale: float = 0.4,
        dn_num: int = 100,
        initialize_box_type: bool = 'no',
        initial_pred: bool = False,
        learn_tgt: bool = False,
        total_num_feature_levels: int = 4,
        dropout: float = 0.0,
        activation: str = 'relu',
        nhead: int = 8,
        dec_n_points: int = 4,
        return_intermediate_dec: bool = True,
        query_dim: int = 4,
        dec_layer_share: bool = False,
        semantic_ce_loss: bool = False,
        num_mask_tokens: int = 6,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn = dn
        self.learn_tgt = learn_tgt
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries

        self.semantic_ce_loss = semantic_ce_loss
        interactive_only = True
        # learnable query features
        if num_queries > 0 and not interactive_only:
            if not two_stage or self.learn_tgt:
                self.query_feat = nn.Embedding(num_queries, hidden_dim)
            if not two_stage and initialize_box_type == 'no':
                self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes = num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        # self.label_enc=nn.Embedding(505, hidden_dim)  # this is a hack for o365+coco (365+133=498)
        self.dim_proj = dim_proj
        self.lang_encoder = lang_encoder
        # if lang_encoder is not None:
        self.lang_mapper = nn.Parameter(torch.empty(dim_proj, hidden_dim))
        trunc_normal_(self.lang_mapper, std=.02)

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, activation, self.num_feature_levels, nhead, dec_n_points
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=hidden_dim,
            query_dim=query_dim,
            num_feature_levels=self.num_feature_levels,
            dec_layer_share=dec_layer_share,
        )

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

        # whole category classification
        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)
        # part category classification
        self.class_embed_part = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed_part, std=.02)

        # FIXME iou head; iou prediction: 1. iou token to predict 3 score. 2. predict each iou score from query tokens
        # FIXME seems we only need to stack these tokens in batch dimension to reduce self attention burden.
        self.num_mask_tokens = num_mask_tokens  # sam uses 4 to handle multi prompts
        self.iou_token = 0  # FIXME hack to remove iou token
        self.num_all_tokens = self.num_mask_tokens + self.iou_token  # sam uses 4 to handle multi prompts
        self.iou_prediction_head = MLP(hidden_dim, hidden_dim, 1, 3)
        # self.iou_token = nn.Embedding(self.iou_token, hidden_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_dim)
        self.pb_embedding = nn.Embedding(2, hidden_dim)
        self.label_enc = nn.Embedding(2, hidden_dim)

        self.prediction_switch = None

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num) > 0:
                scalar = scalar // (int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            # use languge as denosing content queries.
            # if task == 'det':
            #     labels = labels  # o365 start from 133 class
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                    diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            # import ipdb; ipdb.set_trace()
            input_label_embed = torch.gather(
                self.lang_encoder.default_text_embeddings, 0, m[:, None].repeat(1, self.dim_proj)
            ) @ self.lang_mapper

            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = input_label_embed.new_zeros(pad_size, self.hidden_dim)
            padding_bbox = input_bbox_embed.new_zeros(pad_size, 4)

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label = padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = input_label_embed.new_tensor([])
            if len(known_num):
                map_known_indice = torch.cat([input_label_embed.new_tensor(range(num)) for num in known_num]
                )  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = input_label_embed.new_ones(tgt_size, tgt_size) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label = None
                input_query_bbox = None
            attn_mask = None
            mask_dict = None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_for_dn_mo(self, targets, tgt, refpoint_emb, batch_size):
        """
        Train SA-1B data with point input.
        This training can be regarded as a multi-granularity denoising process
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        scalar, noise_scale = self.dn_num, self.noise_scale

        pb_labels = torch.stack([t['pb'] for t in targets])
        # FIXME this is for future content-based interaction; pool content features as label embedding
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.stack([t['boxes_dn'] for t in targets])
        box_start = [t['box_start'] for t in targets]

        known_labels = labels
        known_pb_labels = pb_labels

        known_bboxs = boxes
        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        if noise_scale > 0 and self.training:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :, :2] = known_bbox_expand[:, :, 2:] / 2
            diff[:, :, 2:] = known_bbox_expand[:, :, 2:]
            # add very small noise to input points
            sc = 0.01
            for i, st in enumerate(box_start):
                diff[i, :st] = diff[i, :st] * sc
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0), diff).cuda() * noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to('cuda')
        m_pb = known_pb_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m) + self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(
            self.num_all_tokens, 1
        ) + self.mask_tokens.weight.unsqueeze(0).repeat(input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_all_tokens, 1)

        single_pad = self.num_all_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_all_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            # 'know_idx': know_idx,
            'pad_size': pad_size,
            'scalar': scalar,
        }

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_for_dn_mo_infer(self, targets, tgt, refpoint_emb, batch_size):
        known = [(torch.ones_like(t['points'])).cuda() for t in targets]
        known_num = [k.sum() for k in known]

        assert max(known_num) > 0

        pb_labels = torch.stack([t['pb'] for t in targets])
        # FIXME this is for future content-based interaction; pool content features as label embedding
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.stack([t['points'] for t in targets])

        known_labels = labels
        known_pb_labels = pb_labels

        known_bboxs = boxes
        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        m = known_labels_expaned.long().to('cuda')
        m_pb = known_pb_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m) + self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(
            self.num_all_tokens, 1
        ) + self.mask_tokens.weight.unsqueeze(0).repeat(input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_all_tokens, 1)

        scalar = int(input_label_embed.shape[1] / self.num_all_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed
        else:
            raise RuntimeError()

        attn_mask = None
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            # 'know_idx': know_idx,
            'pad_size': pad_size,
            'scalar': scalar,
        }

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            # layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            new_layer_ref_sig = layer_ref_sig.view(
                layer_ref_sig.shape[0], -1, self.num_all_tokens, layer_ref_sig.shape[-1]
            )
            new_layer_ref_sig = new_layer_ref_sig[:, :, :self.num_mask_tokens].reshape(
                new_layer_ref_sig.shape[0], -1, new_layer_ref_sig.shape[-1]
            )
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(new_layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(
        self, x, mask_features, masks, targets=None, target_queries=None, target_vlp=None, task='seg', extra={}
    ):
        """
        task: seg/det TODO add sam
        """
        prediction_switch = extra
        self.prediction_switch = prediction_switch
        assert len(x) == self.num_feature_levels
        do_seg = True  # if task is det, not do segmentation training (for O365)
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x
            ]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_class_part = []
        predictions_mask = []
        predictions_iou_score = []

        tgt_mask = None
        mask_dict = None
        if self.dn != "no":
            assert targets is not None
            if task == 'demo':
                input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                    self.prepare_for_dn_mo_infer(targets, None, None, x[0].shape[0])
            else:
                input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                    self.prepare_for_dn_mo(targets, None, None, x[0].shape[0])
            tgt = input_query_label
            refpoint_embed = input_query_bbox
            if tgt is None:
                tgt = torch.zeros(bs, self.num_queries, self.hidden_dim).cuda()
                refpoint_embed = torch.zeros(bs, self.num_queries, 4).cuda()
        else:
            raise RuntimeError()

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        new_hs = []
        for i, output in enumerate(hs):
            outputs_class, outputs_mask, iou_score, decoder_output_mask = self.interactive_forward_prediction_heads(
                output.transpose(0, 1), mask_features, (self.training or (i == len(hs) - 1)) and do_seg
            )
            outputs_class_whole, outputs_class_part = outputs_class
            predictions_class.append(outputs_class_whole)
            predictions_class_part.append(outputs_class_part)
            predictions_mask.append(outputs_mask)
            if iou_score is not None:
                predictions_iou_score.append(iou_score)
                new_hs.append(decoder_output_mask)
        if new_hs is not None:
            hs = new_hs
        # iteratively box prediction
        out_boxes = self.pred_box(references, hs)
        out_boxes[-1] = out_boxes[-1] + 0.0 * (
            self.label_enc.weight.sum() + self.pb_embedding.weight.sum() + self.mask_tokens.weight.sum() +
            self.lang_mapper.sum()
        )
        if mask_dict is not None:
            if predictions_mask is None:
                predictions_class[-1] = predictions_class[-1]
                for i in range(self.mask_embed.num_layers):
                    predictions_class[-1] = predictions_class[-1] + 0.0 * (
                        self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[0]
                    )  # avoid no mask loss
                predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss

            if do_seg:
                predictions_mask = list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            for i in range(self.mask_embed.num_layers):
                predictions_class[-1] = predictions_class[-1] + 0.0 * (
                    self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[0]
                )  # avoid no mask loss
            predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss

        out = {
            'pred_logits': predictions_class[-1],
            'pred_logits_part': predictions_class_part[-1],
            'pred_masks': None if not do_seg else predictions_mask[-1],
            'pred_boxes': out_boxes[-1],
            'pred_ious': predictions_iou_score[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, out_boxes,
                predictions_iou_score, predictions_class_part
            )
        }

        return out, mask_dict

    def interactive_forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        decoder_output = decoder_output + 0.0 * (self.class_embed_part.sum() + self.class_embed.sum())

        out = decoder_output.view(decoder_output.shape[0], -1, self.num_all_tokens, decoder_output.shape[-1])
        decoder_output_mask = out[:, :, :self.num_mask_tokens].reshape(
            decoder_output.shape[0], -1, decoder_output.shape[-1]
        )
        # decoder_output_iou = out[:, :, -1].view(decoder_output.shape[0], -1, decoder_output.shape[-1])
        decoder_output_iou = decoder_output_mask

        outputs_mask = outputs_class_whole = outputs_class_part = None
        if self.prediction_switch['whole']:
            class_embed_whole = decoder_output @ self.class_embed
            outputs_class_whole = self.lang_encoder.compute_similarity(class_embed_whole, name='whole')
        if self.prediction_switch['part']:
            class_embed_part = decoder_output @ self.class_embed_part
            outputs_class_part = self.lang_encoder.compute_similarity(class_embed_part, name='part')

        outputs_class = (outputs_class_whole, outputs_class_part)
        if self.prediction_switch['seg']:
            mask_embed = self.mask_embed(decoder_output_mask)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        iou_score = self.iou_prediction_head(decoder_output_iou
        ).squeeze(-1).view(decoder_output.shape[0], -1, self.num_mask_tokens)
        # outputs_mask = outputs_mask + 0.0 * iou_score.sum()  # TODO add iou prediction head

        return outputs_class, outputs_mask, iou_score, decoder_output_mask

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class=None,
        outputs_seg_masks=None,
        out_boxes=None,
        predictions_iou_score=None,
        predictions_class_part=None
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        elif outputs_seg_masks is None:
            return [{"pred_logits": a, "pred_boxes": c} for a, c in zip(outputs_class[:-1], out_boxes[:-1])]
        elif predictions_iou_score is None:
            return [{
                "pred_logits": a,
                "pred_masks": b,
                "pred_boxes": c
            } for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])]
        else:
            return [{
                "pred_logits": a,
                "pred_masks": b,
                "pred_boxes": c,
                "pred_ious": d,
                "pred_logits_part": e
            } for a, b, c, d, e in zip(
                outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_iou_score[:-1],
                predictions_class_part[:-1]
            )]
