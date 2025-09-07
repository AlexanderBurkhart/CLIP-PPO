"""
Shared disturbance types and configurations.
"""

from enum import Enum


class DisturbanceSeverity(Enum):
    """Disturbance severity levels."""
    NONE = "NONE"
    MILD = "MILD"
    MODERATE = "MODERATE"
    HARD = "HARD"
    SEVERE = "SEVERE"


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