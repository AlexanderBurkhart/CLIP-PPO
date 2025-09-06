"""
Visual disturbance wrapper for robustness testing.
"""

import numpy as np
import cv2
from typing import Optional
from enum import Enum


class DisturbanceSeverity(Enum):
    """Disturbance severity levels."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    HARD = "hard"
    SEVERE = "severe"


# Predefined severity configurations
SEVERITY_CONFIGS = {
    DisturbanceSeverity.MILD: {
        'gaussian_noise_sigma': 0.08,
        'gaussian_blur_sigma': 1.0,
        'contrast_range': (0.75, 1.25),
        'cutout_ratio': 0.10
    },
    DisturbanceSeverity.MODERATE: {
        'gaussian_noise_sigma': 0.12,
        'gaussian_blur_sigma': 2.0,
        'contrast_range': (0.7, 1.3),
        'cutout_ratio': 0.17
    },
    DisturbanceSeverity.HARD: {
        'gaussian_noise_sigma': 0.13,
        'gaussian_blur_sigma': 2.1,
        'contrast_range': (0.69, 1.31),
        'cutout_ratio': 0.18
    },
    DisturbanceSeverity.SEVERE: {
        'gaussian_noise_sigma': 0.26,
        'gaussian_blur_sigma': 3.0,
        'contrast_range': (0.6, 1.4),
        'cutout_ratio': 0.25
    }
}


class DisturbanceWrapper:
    """
    Applies visual disturbances to image observations for robustness testing.
    """
    
    def __init__(
        self, seed: Optional[int] = None, severity: Optional[DisturbanceSeverity] = DisturbanceSeverity.MILD,
        gaussian_noise_sigma: Optional[float] = None,
        gaussian_blur_sigma: Optional[float] = None,
        contrast_range: Optional[tuple] = None,
        cutout_ratio: Optional[float] = None
    ):
        """
        Initialize the disturbance wrapper.
        
        Args:
            seed: Random seed for reproducible disturbances
            severity: Severity level (MILD, MODERATE, SEVERE)
        """
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


        self.rng = np.random.RandomState(seed)
    
    def apply_disturbances(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply all disturbances to the observation using instance parameters.
        
        Args:
            obs: Input observation (RGB image as uint8 or float)
            
        Returns:
            Disturbed observation as uint8
        """
        if obs.dtype != np.uint8:
            # Convert to uint8 if needed
            obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)
        
        disturbed_obs = obs.copy()
        
        # Apply all disturbances using instance parameters
        disturbed_obs = self.apply_gaussian_noise(disturbed_obs)
        disturbed_obs = self.apply_contrast_jitter(disturbed_obs)
        disturbed_obs = self.apply_gaussian_blur(disturbed_obs)
        disturbed_obs = self.apply_cutout(disturbed_obs)
        
        return disturbed_obs
    
    def apply_gaussian_noise(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to observation.
        
        Args:
            obs: Input observation (RGB image as uint8)
            
        Returns:
            Noisy observation as uint8
        """
        noise = self.rng.normal(0, self.gaussian_noise_sigma * 255, obs.shape)
        noisy_obs = obs.astype(np.float32) + noise
        return np.clip(noisy_obs, 0, 255).astype(np.uint8)
    

    def apply_contrast_jitter(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply contrast jitter to observation.
        
        Args:
            obs: Input observation (RGB image as uint8)
            
        Returns:
            Contrast adjusted observation as uint8
        """
        contrast_factor = self.rng.uniform(self.contrast_range[0], self.contrast_range[1])
        adjusted = obs.astype(np.float32) * contrast_factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def apply_gaussian_blur(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to observation.
        
        Args:
            obs: Input observation (RGB image as uint8)
            
        Returns:
            Blurred observation as uint8
        """
        kernel_size = max(3, int(2 * self.gaussian_blur_sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply blur to each channel if multi-channel
        if len(obs.shape) == 3:
            blurred = np.zeros_like(obs)
            for c in range(obs.shape[2]):
                blurred[:, :, c] = cv2.GaussianBlur(obs[:, :, c], (kernel_size, kernel_size), self.gaussian_blur_sigma)
            return blurred
        else:
            return cv2.GaussianBlur(obs, (kernel_size, kernel_size), self.gaussian_blur_sigma)
    
    def apply_cutout(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply cutout/occlusion patches to observation.
        
        Args:
            obs: Input observation (RGB image as uint8)
            
        Returns:
            Observation with cutout patches as uint8
        """
        h, w = obs.shape[:2]
        
        # Calculate patch dimensions
        patch_area = int(h * w * self.cutout_ratio)
        patch_h = int(np.sqrt(patch_area))
        patch_w = patch_area // patch_h
        
        # Random patch position
        start_h = self.rng.randint(0, max(1, h - patch_h))
        start_w = self.rng.randint(0, max(1, w - patch_w))
        
        # Apply cutout (fill with black)
        cutout_obs = obs.copy()
        cutout_obs[start_h:start_h+patch_h, start_w:start_w+patch_w] = 0
        return cutout_obs
