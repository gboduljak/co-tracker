from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoVideoProcessor, AutoModel
from cotracker.models.core.cotracker.dpt import DPTHead

class VJEPAFeatureExtractor(nn.Module):
    def __init__(self, vjepa: str, height: int = 384, width: int = 512):
        super(VJEPAFeatureExtractor, self).__init__()
        self.processor = AutoVideoProcessor.from_pretrained(vjepa)
        self.model = AutoModel.from_pretrained(
            vjepa,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )
        self.height = height
        self.width = width

    def forward(self, raw_video: torch.Tensor):
        processed_video = self.processor(
            raw_video,
            return_tensors="pt",
            size=(self.height, self.width),
            do_center_crop=False
        )
        processed_video = {
            k: v.to(self.model.device)
            for k, v in processed_video.items()
        }
        b, t, c, h, w = processed_video["pixel_values_videos"].shape
        with torch.inference_mode():
            outputs = self.model(**processed_video)
        x = outputs.last_hidden_state
        x = rearrange(
            x,
            "b (t h w) d -> b d t h w",
            t=t // 2,
            h=h // 16,
            w=w // 16
        )
        x = F.interpolate(
            x,
            size=(t, h, w),
            mode='trilinear',
            align_corners=False
        )
        x = rearrange(x, "b d t h w -> b t h w d")
        x = x.to(torch.float32)
        return x


class VJEPADPTExtractor(nn.Module):
    def __init__(
        self,
        vjepa: str,
        height: int = 384,
        width: int = 512,
        vjepa_intermediate_layers: List[int] = [10, 20, 30, 40],
        vjepa_dim: int = 1408,
        latent_dim: int = 128,
        vjepa_patch_size: int = 16,
        down_ratio: int = 4,
        freeze_vjepa: bool = False,
    ):
        super(VJEPADPTExtractor, self).__init__()
        self.processor = AutoVideoProcessor.from_pretrained(vjepa)
        self.model = AutoModel.from_pretrained(
            vjepa,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )
        if freeze_vjepa:
            for param in self.model.parameters():
                param.requires_grad = False

        self.height = height
        self.width = width
        self.vjepa_intermediate_layers = vjepa_intermediate_layers
        self.vjepa_patch_size = vjepa_patch_size
        self.dpt = DPTHead(
            dim_in=vjepa_dim,
            patch_size=vjepa_patch_size,
            features=latent_dim,
            output_dim=latent_dim,
            feature_only=True,
            down_ratio=down_ratio,
            pos_embed=False,
            intermediate_layer_idx=vjepa_intermediate_layers,
            activation="linear"
        )
        self.freeze_vjepa = freeze_vjepa


    def forward(self, raw_video: torch.Tensor):
        processed_video = self.processor(
            raw_video,
            return_tensors="pt",
            size=(self.height, self.width),
            do_center_crop=False
        )
        processed_video = {
            k: v.to(self.model.device)
            for k, v in processed_video.items()
        }
        b, t, c, h, w = processed_video["pixel_values_videos"].shape
        patch_t = t // 2
        patch_h = h // self.vjepa_patch_size
        patch_w = w // self.vjepa_patch_size
    
        with torch.inference_mode(self.freeze_vjepa):
            outputs = self.model(
                **processed_video,
                output_hidden_states=True,
                skip_predictor=True
            )

        hs: List[torch.Tensor] = []
        for h in outputs.hidden_states:
            h = rearrange(
                h.to(torch.float32),
                "b (t h w) d -> b d t h w",
                t=patch_t,
                h=patch_h,
                w=patch_w
            )
            h = F.interpolate(
                h,
                scale_factor=(2, 1, 1),
                mode='trilinear',
                align_corners=False
            )
            h = rearrange(
                h,
                "b d t h w -> b t (h w) d"
            )
            hs.append(h)

        x = self.dpt(
            aggregated_tokens_list=hs,
            images=raw_video / 255.0,
            patch_start_idx=0
        )

        return x

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