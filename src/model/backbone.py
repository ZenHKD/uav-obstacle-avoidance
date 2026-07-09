import torch
import torch.nn as nn
import timm

class FastViTBackbone(nn.Module):
    def __init__(self, model_name='fastvit_t8', pretrained=True):
        """
        FastViT backbone wrapper for autonomous UAV perception.
        
        Args:
            model_name (str): Name of the FastViT variant (e.g., 'fastvit_t8', 'fastvit_t12', 'fastvit_sa12').
            pretrained (bool): If True, loads pre-trained ImageNet weights from timm.
        """
        super().__init__()
        self.model_name = model_name
        
        # Create the model using timm with features_only=True to get stage outputs
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Expose stage feature dimensions for the depth head / downstream decoders
        self.channels = [info['num_chs'] for info in self.model.feature_info]
        self.reductions = [info['reduction'] for info in self.model.feature_info]
        
    def forward(self, x):
        """
        Forward pass to extract multi-scale stage features.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            
        Returns:
            list of torch.Tensor: List of 4 feature maps from stages 0 to 3.
                                 For an input of 224x224, their spatial resolutions
                                 are typically 56x56, 28x28, 14x14, and 7x7.
        """
        return self.model(x)
