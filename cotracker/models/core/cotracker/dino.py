from typing import List, Literal
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoImageProcessor, AutoModel

from cotracker.models.core.cotracker.blocks import SpatioTemporalTransformer
from cotracker.models.core.cotracker.dpt import DPTHead


class DINOFeatureExtractor(nn.Module):
    def __init__(
        self,
        dino: str,
        input_height: int = 384,
        input_width: int = 512,
        resize_height: int = 336,
        resize_width: int = 448,
        target_height: int = 96,
        target_width: int = 128,
        freeze_dino: bool = False,
        dino_dim: int = 1024,
        dino_patch_size: int = 14,
        feature_dim: int = 128,
        refine_dino: bool = False,
        dino_intermediate_features_idx: List[int] = [-1],
        upsampling_type: Literal["bilinear", "dpt"] = "bilinear"
    ):
        super(DINOFeatureExtractor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(dino)
        self.model = AutoModel.from_pretrained(dino)
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.target_height = target_height
        self.target_width = target_width
        if freeze_dino:
            for param in self.model.parameters():
                param.requires_grad = False
        self.freeze_dino = freeze_dino
        self.dino_intermediate_features_idx = dino_intermediate_features_idx
        self.refine_dino = refine_dino
        self.upsampling_type = upsampling_type
        if refine_dino:
            num_refiners = len(dino_intermediate_features_idx)
            self.projs = nn.ModuleList([
                nn.Linear(dino_dim, feature_dim)
                for _ in range(num_refiners)
            ])
            self.refiners = nn.ModuleList([
                SpatioTemporalTransformer(
                    hidden_size=feature_dim,
                    space_depth=3,
                    time_depth=3
                ) for _ in range(num_refiners)
            ])
        if dino_intermediate_features_idx != [-1]:
          down_ratio = input_height // target_height
          self.dpt = DPTHead(
              dim_in=dino_dim,
              patch_size=dino_patch_size,
              features=feature_dim,
              output_dim=feature_dim,
              feature_only=True,
              down_ratio=down_ratio,
              out_size=(
                 input_height // down_ratio,
                 input_width // down_ratio,
              ),
              pos_embed=False,
              intermediate_layer_idx=list(range(len(dino_intermediate_features_idx))),
              activation="linear"
          )
        self.refine_dino = refine_dino

    def state_dict(self, *args, **kwargs):
        full_state_dict = super().state_dict(*args, **kwargs)
        trainable_param_names = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        return {
            name: param
            for name, param in full_state_dict.items()
            if name in trainable_param_names or not isinstance(param, torch.Tensor)
        }

    def forward(self, raw_video: torch.Tensor):
        b, t, *_ = raw_video.shape
        processed_video = self.processor(
            rearrange(
                raw_video.to(torch.uint8),
                "b t h w c -> (b t) h w c"
            ),
            return_tensors="pt",
            size=(self.resize_height, self.resize_width),
            do_center_crop=False
        )
        processed_video = {
            k: v.to(self.model.device).to(raw_video.dtype)
            for k, v in processed_video.items()
        }
        _, c, h, w = processed_video["pixel_values"].shape
        with torch.inference_mode(self.freeze_dino):
            features_idx = self.dino_intermediate_features_idx
            if features_idx != [-1]:
                outputs = self.model(
                    **processed_video,
                    output_hidden_states=True,
                )
                hs = []
                for idx in features_idx:
                  hs.append(
                      rearrange(
                        outputs.hidden_states[idx],
                        "(b t) n d -> b t n d",
                        b=b
                      )
                  )
            else:
                outputs = self.model(**processed_video)
                hs = [
                    outputs.last_hidden_state
                ]

        if self.refine_dino:
          xs = []
          for (h, proj, refine) in zip(hs, self.projs, self.refiners):
            h = h.clone()
            h = proj(h)
            h = refine(h)
            xs.append(h)
        else:
          xs = [
            h.clone()
            for h in hs
          ]
        if self.upsampling_type == "bilinear":
          [x] = xs
          x = x.clone()
          x = rearrange(
              x,
              "b (h w) d -> b d h w",
              h=h // 14,
              w=w // 14
          )
          x = F.interpolate(
              x,
              size=(self.target_height, self.target_width),
              mode='bilinear',
              align_corners=False
          )
          x = rearrange(
              x,
              "(b t) d h w -> b t h w d",
              b=b,
              t=t
          )
          x = self.proj(x)
        else:
          x = self.dpt(
              aggregated_tokens_list=xs,
              images=rearrange(
                processed_video["pixel_values"],
                "(b t) c h w -> b t c h w",
                b=b
              ),
              patch_start_idx=1 # index 0 is CLS token
          )
        return x