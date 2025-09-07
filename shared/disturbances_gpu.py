"""
GPU-accelerated visual disturbance wrapper for robustness testing using PyTorch.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import numpy as np
from typing import Optional, Union
from shared.disturbance_types import DisturbanceSeverity, SEVERITY_CONFIGS


class DisturbanceWrapperGPU:
    """
    GPU-accelerated visual disturbance wrapper using PyTorch.
    Significantly faster than CPU version, especially for batched operations.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        seed: Optional[int] = None,
        severity: Optional[DisturbanceSeverity] = DisturbanceSeverity.MILD,
        gaussian_noise_sigma: Optional[float] = None,
        gaussian_blur_sigma: Optional[float] = None,
        contrast_range: Optional[tuple] = None,
        cutout_ratio: Optional[float] = None
    ):
        """
        Initialize the GPU disturbance wrapper.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or torch.device)
            seed: Random seed for reproducible disturbances
            severity: Severity level (MILD, MODERATE, HARD, SEVERE)
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Set up parameters from severity config or custom values
        severity_config = None
        if severity is not None:
            severity_config = SEVERITY_CONFIGS[severity]
        elif None in (gaussian_noise_sigma, gaussian_blur_sigma, contrast_range, cutout_ratio):
            raise ValueError('All custom parameters must not be None if not setting a severity.')

        using_severity_config = severity_config is not None
        self.gaussian_noise_sigma = severity_config['gaussian_noise_sigma'] if using_severity_config else gaussian_noise_sigma
        self.gaussian_blur_sigma = severity_config['gaussian_blur_sigma'] if using_severity_config else gaussian_blur_sigma
        self.contrast_range = severity_config['contrast_range'] if using_severity_config else contrast_range
        self.cutout_ratio = severity_config['cutout_ratio'] if using_severity_config else cutout_ratio

        # Set up random number generator
        if seed is not None:
            torch.manual_seed(seed)
        
        # Set up torchvision transforms
        kernel_size = max(3, int(2 * self.gaussian_blur_sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.blur_transform = T.GaussianBlur(kernel_size, self.gaussian_blur_sigma)
        self.noise_transform = T2.GaussianNoise(mean=0.0, sigma=self.gaussian_noise_sigma)
        self.contrast_transform = T.ColorJitter(contrast=self.contrast_range)
    
    
    def apply_disturbances(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply all disturbances to tensor observations (fast path for training)."""
        # Assume obs is already in [B, C, H, W] format and float [0,1] range
        disturbed_obs = self.apply_gaussian_noise(obs)
        disturbed_obs = self.apply_contrast_jitter(disturbed_obs)
        disturbed_obs = self.apply_gaussian_blur(disturbed_obs)
        disturbed_obs = self.apply_cutout(disturbed_obs)
        return disturbed_obs
    
    def apply_disturbances_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Apply all disturbances to numpy observations (for test scripts)."""
        was_single = len(obs.shape) == 3
        obs_tensor = torch.from_numpy(obs).to(self.device).float() / 255.0
        
        # Convert to [B, C, H, W] format
        if was_single:  # [H, W, C] -> [1, C, H, W]
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
        else:  # [B, H, W, C] -> [B, C, H, W]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        
        # Apply disturbances
        disturbed = self.apply_disturbances(obs_tensor)
        
        # Convert back to numpy format
        if was_single:
            disturbed = disturbed.squeeze(0).permute(1, 2, 0)  # [1, C, H, W] -> [H, W, C]
        else:
            disturbed = disturbed.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        
        return (disturbed * 255.0).byte().cpu().numpy()
    
    def apply_gaussian_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to tensor observations."""
        return self.noise_transform(obs)
    
    def apply_gaussian_noise_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise to numpy observations."""
        was_single = len(obs.shape) == 3
        obs = torch.from_numpy(obs).to(self.device).float() / 255.0
        
        if was_single:  # [H, W, C] -> [1, C, H, W]
            obs = obs.permute(2, 0, 1).unsqueeze(0)
            noisy = self.noise_transform(obs)
            noisy = noisy.squeeze(0).permute(1, 2, 0)
        else:  # [B, H, W, C] -> [B, C, H, W]
            obs = obs.permute(0, 3, 1, 2)
            noisy = self.noise_transform(obs)
            noisy = noisy.permute(0, 2, 3, 1)
        
        return (noisy.clamp(0, 1) * 255.0).byte().cpu().numpy()
    
    def apply_contrast_jitter(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply contrast jitter to tensor observations."""
        return self.contrast_transform(obs)
    
    def apply_contrast_jitter_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Apply contrast jitter to numpy observations."""
        was_single = len(obs.shape) == 3
        obs_tensor = torch.from_numpy(obs).to(self.device).float() / 255.0
        
        if was_single:  # [H, W, C] -> [1, C, H, W]
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
            jittered = self.contrast_transform(obs_tensor)
            jittered = jittered.squeeze(0).permute(1, 2, 0)
        else:  # [B, H, W, C] -> [B, C, H, W]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            jittered = self.contrast_transform(obs_tensor)
            jittered = jittered.permute(0, 2, 3, 1)
        
        return (jittered * 255.0).byte().cpu().numpy()
    
    def apply_gaussian_blur(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to tensor observations."""
        return self.blur_transform(obs)
    
    def apply_gaussian_blur_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to numpy observations."""
        was_single = len(obs.shape) == 3
        obs_tensor = torch.from_numpy(obs).to(self.device).float() / 255.0
        
        if was_single:  # [H, W, C] -> [1, C, H, W]
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
            blurred = self.blur_transform(obs_tensor)
            blurred = blurred.squeeze(0).permute(1, 2, 0)
        else:  # [B, H, W, C] -> [B, C, H, W]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            blurred = self.blur_transform(obs_tensor)
            blurred = blurred.permute(0, 2, 3, 1)
        
        return (blurred * 255.0).byte().cpu().numpy()
    
    def apply_cutout(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply cutout to tensor observations."""
        batch_size, channels, height, width = obs.shape
        cutout_obs = obs.clone()
        
        # Calculate patch dimensions
        patch_area = int(height * width * self.cutout_ratio)
        patch_h = int(np.sqrt(patch_area))
        patch_w = patch_area // patch_h
        
        # Apply same cutout patch to entire batch
        start_h = torch.randint(0, max(1, height - patch_h + 1), (1,)).item()
        start_w = torch.randint(0, max(1, width - patch_w + 1), (1,)).item()
        
        cutout_obs[:, :, start_h:start_h+patch_h, start_w:start_w+patch_w] = 0.0
        return cutout_obs
    
    def apply_cutout_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Apply cutout to numpy observations."""
        was_single = len(obs.shape) == 3
        obs_tensor = torch.from_numpy(obs).to(self.device).float() / 255.0
        
        # Convert to [B, C, H, W] format
        if was_single:  # [H, W, C] -> [1, C, H, W]
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
        else:  # [B, H, W, C] -> [B, C, H, W]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        
        # Apply cutout
        cutout_obs = self.apply_cutout(obs_tensor)
        
        # Convert back to numpy format
        if was_single:
            cutout_obs = cutout_obs.squeeze(0).permute(1, 2, 0)  # [1, C, H, W] -> [H, W, C]
        else:
            cutout_obs = cutout_obs.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        
        return (cutout_obs * 255.0).byte().cpu().numpy()


# Convenience function for backward compatibility
def create_disturbance_wrapper(use_gpu=True, **kwargs):
    """
    Create appropriate disturbance wrapper based on GPU availability.
    
    Args:
        use_gpu: Whether to use GPU acceleration (default: True)
        **kwargs: Arguments passed to wrapper constructor
        
    Returns:
        DisturbanceWrapper instance (GPU or CPU version)
    """
    if use_gpu and torch.cuda.is_available():
        return DisturbanceWrapperGPU(device="cuda", **kwargs)
    else:
        # Fallback to CPU version
        from disturbances import DisturbanceWrapper
        return DisturbanceWrapper(**kwargs)