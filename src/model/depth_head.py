import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
from torch import Tensor

# =====================================================================
# 1. DPT Intermediate Blocks and Helpers (Self-Contained)
# =====================================================================

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""
    def __init__(self, features: int, activation: nn.Module, bn: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block to merge and upsample multi-scale representations."""
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Optional[Tuple[int, int]] = None,
        has_residual: bool = True,
    ):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size: Optional[Tuple[int, int]] = None) -> Tensor:
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if size is None and self.size is None:
            modifier = {"scale_factor": 2.0}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features: int, has_residual: bool = True) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        has_residual=has_residual,
    )


def _make_scratch(in_shape: List[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch

# =====================================================================
# 2. Main DPT Depth Head Model
# =====================================================================

class DPTHead(nn.Module):
    """
    Dense Prediction Transformer (DPT) Head designed for UAV 2D Depth + Confidence estimation.
    Fuses multi-scale feature maps from hierarchical backbones (like FastViT).
    """
    def __init__(
        self,
        in_channels: List[int] = [64, 128, 256, 512],
        reductions: List[int] = [4, 8, 16, 32],
        output_dim: int = 1,
        activation: str = "inv_log",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
    ) -> None:
        """
        Args:
            in_channels (List[int]): Channels of the 4 backbone/GCA stages.
            reductions (List[int]): Resolution reduction factor for each stage.
            output_dim (int): Number of output channels (1: for depth map).
            activation (str): Activation type for depth map ('inv_log', 'exp', 'relu', 'sigmoid', 'linear').
            features (int): Intermediate channels for fusion block layers.
            out_channels (List[int]): Output channels of projection layers for intermediate features.
        """
        super().__init__()
        self.in_channels = in_channels
        self.reductions = reductions
        self.output_dim = output_dim
        self.activation = activation

        # Projection layers matching multi-scale feature channels to DPT channels
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch, out_channels=oc, kernel_size=1, stride=1, padding=0) 
            for in_ch, oc in zip(in_channels, out_channels)
        ])

        # Resize layers. Since FastViT features are already multi-resolution, 
        # we keep them as identities and handle resolution shifts inside FeatureFusionBlock.
        self.resize_layers = nn.ModuleList([nn.Identity() for _ in range(4)])

        self.scratch = _make_scratch(out_channels, features)

        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features: List[Tensor], img_size: Tuple[int, int] = (224, 224)) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            out_features (List[Tensor]): List of 4 feature maps from the backbone/GCA blocks.
                                         Can be 4D shape (B, C, H, W) or sequential 4D/5D shapes.
            img_size (Tuple[int, int]): Target resolution to upscale the depth and confidence maps.
            
        Returns:
            Tensor: Depth Map of shape (B, 1, H_img, W_img) or (B, T, 1, H_img, W_img)
        """
        # Determine sequence shape properties
        first_feat = out_features[0]
        B, T = 1, 1
        
        if first_feat.ndim == 5:  # (B, T, C, H, W)
            B, T = first_feat.shape[0], first_feat.shape[1]
        elif first_feat.ndim == 4:
            if first_feat.shape[1] == self.in_channels[0]:  # (B, C, H, W)
                B = first_feat.shape[0]
                T = 1
            else:  # (B, T, N, C)
                B, T = first_feat.shape[0], first_feat.shape[1]
        else:  # (B, N, C)
            B = first_feat.shape[0]
            T = 1

        # Normalize and align multi-scale features to (B * T, C, H_stage, W_stage)
        normalized_features = []
        for i, feat in enumerate(out_features):
            H_stage = img_size[0] // self.reductions[i]
            W_stage = img_size[1] // self.reductions[i]
            N_spatial = H_stage * W_stage
            
            if feat.ndim == 5:  # (B, T, C, H, W)
                feat_norm = feat.reshape(B * T, feat.shape[2], feat.shape[3], feat.shape[4])
            elif feat.ndim == 4:
                if feat.shape[1] == self.in_channels[i]:  # (B, C, H, W)
                    feat_norm = feat.reshape(B * T, self.in_channels[i], H_stage, W_stage)
                else:  # (B, T, N, C)
                    # Exclude prepended special/IMU tokens from sequence if present
                    spatial_tokens = feat[:, :, -N_spatial:]  # (B, T, N_spatial, C)
                    feat_norm = spatial_tokens.permute(0, 1, 3, 2).reshape(B * T, self.in_channels[i], H_stage, W_stage)
            else:  # (B, N, C)
                spatial_tokens = feat[:, -N_spatial:]  # (B, N_spatial, C)
                feat_norm = spatial_tokens.permute(0, 2, 1).reshape(B * T, self.in_channels[i], H_stage, W_stage)
                
            normalized_features.append(feat_norm)

        # Apply projections
        out_projections = []
        for i, x in enumerate(normalized_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out_projections.append(x)

        # Merge features from multi-scale stages via DPT RefineNets
        layer_1, layer_2, layer_3, layer_4 = out_projections
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # Upsample merged features to target image size
        out_fused = self.scratch.output_conv1(path_1)
        out_fused = F.interpolate(out_fused, size=img_size, mode="bilinear", align_corners=True)
        out_final = self.scratch.output_conv2(out_fused)  # (B * T, output_dim, H, W)

        # Depth raw map
        depth_raw = out_final

        # 1. Depth activation
        if self.activation == "inv_log":
            depth = torch.sign(depth_raw) * (torch.expm1(torch.abs(depth_raw)))
        elif self.activation == "exp":
            depth = torch.exp(depth_raw)
        elif self.activation == "relu":
            depth = F.relu(depth_raw)
        elif self.activation == "softplus":
            depth = F.softplus(depth_raw)
        elif self.activation == "sigmoid":
            depth = torch.sigmoid(depth_raw)
        elif self.activation == "linear":
            depth = depth_raw
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Reshape back to sequence format if necessary
        if T > 1:
            depth = depth.reshape(B, T, 1, img_size[0], img_size[1])

        return depth

# =====================================================================
# 3. Auxiliary Distance Regressor Head
# =====================================================================

class DistanceRegressor(nn.Module):
    """
    Auxiliary regression head that takes visual/spatial-temporal features 
    and outputs a scalar prediction of the distance to the nearest obstacle.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        """
        Args:
            in_channels (int): Channels of the input feature map (e.g. Stage 3 features).
            hidden_dim (int): Hidden layer dimensionality in the regressor MLP.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input feature tensor. Can be 4D shape (B, C, H, W) or 5D shape (B, T, C, H, W).
            
        Returns:
            Tensor: Scalar distance predictions of shape (B, 1) or (B, T, 1).
        """
        orig_shape = x.shape
        if x.ndim == 5:
            B, T, C, H, W = orig_shape
            x = x.reshape(B * T, C, H, W)
        else:
            B, C, H, W = orig_shape
            T = 1

        # Global average pool spatial dimensions
        x_pooled = self.pool(x).flatten(1)  # (B * T, C)
        out = self.mlp(x_pooled)  # (B * T, 1)
        
        # Constrain output to a physically meaningful range [0.0, 10.0] meters using sigmoid
        out = 10.0 * torch.sigmoid(out)

        if T > 1:
            out = out.reshape(B, T, 1)
        return out
