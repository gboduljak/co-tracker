import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoVideoProcessor, AutoModel

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