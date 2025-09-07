"""
Shared utilities for CLIP-PPO implementation across different environments.
"""
import torch
import torch.nn as nn
import clip
from enum import Enum
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


class AblationMode(Enum):
    """Ablation study modes for CLIP-PPO."""
    NONE = "NONE"
    FROZEN_CLIP = "FROZEN_CLIP" 
    RANDOM_ENCODER = "RANDOM_ENCODER"


# CLIP ImageNet normalization constants
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

CLIP_LOSS_FREQUENCY = 1

def compute_cosine_embedding_loss(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine embedding loss between PPO latent representations (z) and CLIP text embeddings (c).
    
    Formula: L_CLIP = 1 - cos(z/||z||, c_text/||c_text||)
    
    Args:
        z: PPO latent vectors [batch_size, latent_dim]
        c: CLIP text embeddings [batch_size, clip_dim]
    
    Returns:
        Cosine embedding loss scalar
    """
    # Check for dimensional mismatch - should not happen with current architecture
    if z.shape[-1] != c.shape[-1]:
        raise ValueError(f"Dimension mismatch: PPO latents ({z.shape[-1]}) vs CLIP embeddings ({c.shape[-1]}). "
                        f"Both should be 512-dim for ViT-B/32. Check agent architecture.")
    
    # L2 Normalize: z/||z|| and c_text/||c_text||
    z_norm = torch.nn.functional.normalize(z, dim=-1)
    c_norm = torch.nn.functional.normalize(c, dim=-1)
    
    # Compute cosine similarity: cos(z/||z||, c_text/||c_text||)
    cosine_sim = torch.sum(z_norm * c_norm, dim=-1)  # [batch_size]
    
    # L_CLIP = 1 - cos(z/||z||, c_text/||c_text||)
    loss = torch.mean(1 - cosine_sim)
    
    return loss


def load_clip_model(model_name: str = "ViT-B/32", device: str = "cuda") -> torch.nn.Module:
    """
    Load and prepare CLIP model for inference.
    
    Args:
        model_name: CLIP model variant
        device: Device to load model on
        
    Returns:
        CLIP model ready for inference (frozen)
    """
    model, _ = clip.load(model_name, device=device)
    model.eval()  # Set to eval mode for inference
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def generate_clip_embeddings(
    ablation_mode: AblationMode,
    clip_model: torch.nn.Module,
    modality: str,
    batch_size: int,
    device: str,
    descriptions: Optional[List[str]] = None,
    images: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Generate embeddings based on ablation mode and modality.
    
    Args:
        ablation_mode: Ablation mode to determine embedding type
        clip_model: CLIP model for encoding
        modality: "text" or "image"
        batch_size: Batch size for random embeddings
        device: Device for tensor operations
        descriptions: Text descriptions for CLIP text encoding (required for text modality)
        images: Images for CLIP image encoding (required for image modality)
        clip_mean: ImageNet normalization mean (required for image modality)
        clip_std: ImageNet normalization std (required for image modality)
        
    Returns:
        Generated embeddings based on ablation mode
    """
    if ablation_mode == AblationMode.RANDOM_ENCODER:
        # Use random embeddings instead of CLIP embeddings
        clip_embeddings = torch.randn(batch_size, 512, device=device)
        clip_embeddings = torch.nn.functional.normalize(clip_embeddings, dim=-1)
        return clip_embeddings
    
    elif modality == "text":
        if descriptions is None:
            raise ValueError("descriptions required for text modality")
        # Compute CLIP text embeddings
        text_tokens = clip.tokenize(descriptions).to(device)
        with torch.no_grad():
            clip_embeddings = clip_model.encode_text(text_tokens).float()
        return torch.nn.functional.normalize(clip_embeddings, dim=-1)
        
    elif modality == "image":
        if images is None:
            raise ValueError("images required for image modality")
            
        # Create normalization tensors on the correct device
        clip_mean = _CLIP_MEAN.to(device).view(1, 3, 1, 1)
        clip_std = _CLIP_STD.to(device).view(1, 3, 1, 1)
        
        # Preprocess images for CLIP
        import torch.nn.functional as F
        clip_images = F.interpolate(
            images.float() / 255.0,  # [B, C, H, W], convert to [0,1]
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False,
            antialias=True,
        )
        # Apply ImageNet normalization
        clip_images = (clip_images - clip_mean) / clip_std
        
        # Use CLIP image embeddings
        with torch.no_grad():
            clip_embeddings = clip_model.encode_image(clip_images).float()
        return torch.nn.functional.normalize(clip_embeddings, dim=-1)
    
    else:
        raise ValueError(f"Invalid modality: {modality}. Must be 'image' or 'text'")


def should_compute_clip_loss(ablation_mode: AblationMode, clip_lambda: float) -> bool:
    """
    Determine whether CLIP alignment loss should be computed based on ablation mode.
    
    Args:
        ablation_mode: Current ablation mode
        clip_lambda: CLIP loss coefficient
        
    Returns:
        True if CLIP loss should be computed
    """
    # No loss computation if lambda is 0 or in FROZEN_CLIP mode
    return clip_lambda > 0.0 and ablation_mode != AblationMode.FROZEN_CLIP


def get_frozen_clip_features(
    x: torch.Tensor, 
    clip_model: clip.model.VisionTransformer | clip.model.CLIP
) -> torch.Tensor:
    """
    Get features from frozen CLIP encoder with standard preprocessing.
    
    Args:
        x: Input observations (batch of images), assumed to be normalized
        clip_model: CLIP visual encoder (frozen)
        
    Returns:
        CLIP visual features in float32
    """
    import torch.nn.functional as F
    
    # Create normalization tensors on the correct device
    clip_mean = _CLIP_MEAN.to(x.device).view(1, 3, 1, 1)
    clip_std = _CLIP_STD.to(x.device).view(1, 3, 1, 1)
    
    # Resize to 224x224 for CLIP and apply ImageNet normalization
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
    # Apply ImageNet normalization
    x = (x - clip_mean) / clip_std
    # Convert to half precision to match CLIP model
    x = x.half()
    with torch.no_grad():
        if type(clip_model) == clip.model.VisionTransformer:
            features = clip_model(x)
        else:
            features = clip_model.encode_image(x)
    # Convert back to float32 for consistency with rest of pipeline
    return features.float()


@dataclass
class ClipPPOConfig:
    """Shared configuration for CLIP-PPO specific parameters."""
    
    # CLIP specific arguments
    clip_lambda: float = 0.00001
    """coefficient for CLIP alignment loss"""
    clip_model: str = "ViT-B/32"
    """CLIP model variant to use"""
    clip_modality: str = "text"
    """CLIP modality to use for alignment: 'image' or 'text'"""
    
    # Ablation study arguments
    ablation_mode: AblationMode = AblationMode.NONE
    """ablation mode for controlled experiments"""
    
    # Visual disturbance arguments
    apply_disturbances: bool = False
    """whether to apply visual disturbances during training"""
    disturbance_severity: str = "MODERATE"
    """disturbance severity level: MILD, MODERATE, HARD, SEVERE"""
