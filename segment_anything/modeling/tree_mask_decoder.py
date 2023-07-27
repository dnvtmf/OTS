from typing import Tuple, List, Type
import torch
from torch import nn, Tensor

from .common import LayerNorm2d
from .mask_decoder import MLP
from .transformer import TwoWayTransformer
from .prompt_encoder import PositionEmbeddingRandom


class TreeMaskDecoder(nn.Module):

    def __init__(
        self,
        *,
        image_embedding_size: Tuple[int, int],
        mask_in_chans=16,
        transformer_dim: int = 256,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          num_multimask_outputs (int): the number of masks to predict when disambiguating masks
          activation (nn.Module): the type of activation to use when upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP used to predict mask quality
        """
        super().__init__()
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        self.prompt_embed = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim, 3, 2, 1),  # BXCx32x32
            LayerNorm2d(transformer_dim),
            activation(),
            nn.Conv2d(transformer_dim, transformer_dim, 3, 2, 1),  # BXCx16x16
            LayerNorm2d(transformer_dim),
            activation(),
            nn.Conv2d(transformer_dim, transformer_dim, 3, 2, 1),  # BXCx8x8
            LayerNorm2d(transformer_dim),
            activation(),
            nn.Conv2d(transformer_dim, transformer_dim, 1, 1, 0),  # BXCx8x8
        )

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, transformer_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, transformer_dim)

        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=transformer_dim, mlp_dim=2048, num_heads=8)

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 1 + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(self, image_embeddings: Tensor, goal_mask: Tensor, parent_embeddings: Tensor = None):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          goal_mask (torch.Tensor): the embeddings of the points and boxes
          parent_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        B, C, H, W = image_embeddings.shape
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        sparse_prompt_embeddings = self.mask_downscaling(goal_mask)  # BxCx64x64
        sparse_prompt_embeddings = self.prompt_embed(sparse_prompt_embeddings).flatten(2).permute(0, 2, 1)  # Bx64xF
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if parent_embeddings is None:
            src = image_embeddings + self.no_mask_embed.weight.reshape(1, -1, 1, 1)
        else:
            src = image_embeddings + self.mask_downscaling(parent_embeddings)
        image_pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        pos_src = torch.repeat_interleave(image_pe, B, dim=0)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(B, C, H, W)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        B, C, H, W = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(B, C, H * W)).view(B, -1, H, W)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        masks = masks[:, 0, :, :]
        iou_pred = iou_pred[:, 0]

        # Prepare output
        return masks, iou_pred


def test():
    from tree_segmentation.extension.utils import show_shape
    m = TreeMaskDecoder(image_embedding_size=(64, 64))
    print(m)
    mask = torch.zeros((2, 1, 256, 256))
    mask[:, :, 64:128, 64:128] = 1.
    image_embeddings = torch.randn((2, 256, 64, 64))
    print('inputs:', show_shape(image_embeddings, mask))
    outputs = m(image_embeddings, mask)
    print('outputs:', show_shape(outputs))
