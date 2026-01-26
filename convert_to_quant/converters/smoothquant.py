import torch
from torch import Tensor
from typing import Optional

class SmoothQuantPreprocessor:
    """
    SmoothQuant preprocessing to migrate quantization difficulty from activations to weights.
    
    Reference: https://arxiv.org/abs/2211.10438
    """
    def __init__(self, alpha: float = 0.5):
        """
        Initialize SmoothQuant preprocessor.
        
        Args:
            alpha: Migration strength. 0.0 = all difficulty to weights, 
                   1.0 = all difficulty to activations. Default 0.5.
        """
        self.alpha = alpha
    
    def compute_smoothing_factors(self, weight: Tensor, activation_scales: Tensor) -> Tensor:
        """
        Compute per-channel smoothing factors.
        
        Args:
            weight: Weight tensor of shape (out_features, in_features)
            activation_scales: Per-channel max from calibration data, shape (in_features,)
            
        Returns:
            Smoothing factors of shape (in_features,)
        """
        # weight_scales: per-channel max of weight columns (in_features)
        # weight is (M, N), we want max over M for each N
        weight_scales = weight.abs().amax(dim=0)
        
        # Smoothing factor per channel: s = act^alpha / weight^(1-alpha)
        # We use float() to ensure precision during power operations
        act_pow = activation_scales.float().pow(self.alpha)
        weight_pow = weight_scales.float().pow(1.0 - self.alpha)
        
        s = act_pow / weight_pow
        s = s.clamp(min=1e-5)  # Avoid division by zero or extremely small values
        
        # If any value is NaN or Inf (e.g. from 0/0), default to 1.0
        s[torch.isnan(s) | torch.isinf(s)] = 1.0
        
        return s.to(weight.dtype)
    
    def apply_to_weight(self, weight: Tensor, s: Tensor) -> Tensor:
        """
        Apply smoothing factors to weights.
        W_smooth = W * diag(s)
        
        Args:
            weight: Weight tensor of shape (out_features, in_features)
            s: Smoothing factors of shape (in_features,)
            
        Returns:
            Smoothed weight tensor
        """
        # weight is (M, N), s is (N,)
        # We want to multiply each column j by s[j]
        return weight * s.unsqueeze(0)
    
    def apply_to_activation(self, x: Tensor, s: Tensor) -> Tensor:
        """
        Apply smoothing factors to activations (for inference/validation).
        X_smooth = X / diag(s)
        
        Args:
            x: Activation tensor
            s: Smoothing factors
            
        Returns:
            Smoothed activation tensor
        """
        return x / s
