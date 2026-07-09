import torch
import torch.nn as nn
from typing import Dict, Optional
from torch import Tensor

from src.model.backbone import FastViTBackbone
from src.model.depth_head import DPTHead

class ODANetFast(nn.Module):
    """
    Obstacle Avoidance Network (ODANetFast).
    Stripped down architecture:
    1. FastViT backbone extracts visual features.
    2. DPT Head predicts the relative depth map.
    
    NOTE: 
    - No internal Voxelization (Projection module removed).
    - Point Cloud generation is deferred to the downstream pipeline for Sparse implementation.
    """
    def __init__(
        self,
        backbone_name: str = "fastvit_t12",
        pretrained: bool = True
    ) -> None:
        super().__init__()
        # 1. Visual Backbone
        self.backbone = FastViTBackbone(model_name=backbone_name, pretrained=pretrained)
        channels = self.backbone.channels
        reductions = self.backbone.reductions
        reductions = self.backbone.reductions

        # 2. DPT Dense Depth Decoder Head (predicts per-frame raw relative depth)
        self.depth_head = DPTHead(
            in_channels=channels,
            reductions=reductions,
            output_dim=1,
            activation="softplus",
            features=128,
            out_channels=[128, 128, 128, 128]
        )

    def forward(
        self,
        images: Tensor,
        camera_pitch_deg: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        """
        Fast Forward pass.
        Returns only Depth.
        """
        # Determine sequence shape properties
        is_sequential = images.ndim == 5
        if is_sequential:
            B, T, C, H, W = images.shape
            images_flat = images.reshape(B * T, C, H, W)
        else:
            B = images.shape[0]
            T = 1
            H, W = images.shape[2], images.shape[3]
            images_flat = images

        # 1. Visual Feature Extraction
        features = self.backbone(images_flat)
        stage0, stage1, stage2, stage3 = features

        # 2. Dense Per-Frame Prediction (Depth Head)
        depth_raw = self.depth_head(features, img_size=(H, W))

        # 3. Metric-Scale Calibration
        depth_metric = depth_raw

        # 4. Reconstruct output sequence structure
        if is_sequential:
            depth_out = depth_metric.reshape(B, T, 1, H, W)
        else:
            depth_out = depth_metric

        return {
            "depth": depth_out,
        }
