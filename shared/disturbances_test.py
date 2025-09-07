#!/usr/bin/env python3
"""
Test script for visual disturbances on Lenna image.
"""

import numpy as np
import cv2
import tyro
from dataclasses import dataclass
from typing import Optional
import os
from disturbances_gpu import create_disturbance_wrapper
from disturbances import DisturbanceSeverity


@dataclass
class DisturbanceTestArgs:
    """Configuration for disturbance testing."""
    
    # Image configuration
    image_path: str = "lenna.png"
    """Path to the test image"""
    
    # Severity configuration (overrides individual parameters if set)
    severity: Optional[DisturbanceSeverity] = DisturbanceSeverity.SEVERE
    """Severity level (MILD, MODERATE, HARD, SEVERE) - overrides individual parameters"""
    
    # Disturbance toggles
    apply_gaussian_noise: bool = True
    """Apply Gaussian noise disturbance"""
    
    apply_gaussian_blur: bool = True
    """Apply Gaussian blur disturbance"""
    
    apply_contrast_jitter: bool = True
    """Apply contrast jitter disturbance"""
    
    apply_cutout: bool = True
    """Apply cutout disturbance"""
    
    # Disturbance parameters (ignored if severity is set)
    gaussian_noise_sigma: float = 0.15
    """Gaussian noise standard deviation (0.08-0.26 range) - ignored if severity is set"""
    
    gaussian_blur_sigma: float = 2.0
    """Gaussian blur standard deviation (1-3 range) - ignored if severity is set"""
    
    contrast_range_min: float = 0.7
    """Minimum contrast factor (0.6-1.4 range) - ignored if severity is set"""
    
    contrast_range_max: float = 1.3
    """Maximum contrast factor (0.6-1.4 range) - ignored if severity is set"""
    
    cutout_ratio: float = 0.15
    """Fraction of image area to occlude (0.1-0.25 range) - ignored if severity is set"""
    
    # Display configuration
    display_time: int = 0
    """Display time in milliseconds (0 = wait for key press)"""
    
    seed: Optional[int] = 42
    """Random seed for reproducible disturbances"""


def load_test_image(image_path: str) -> np.ndarray:
    """
    Load the test image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as uint8 numpy array
    """
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return None
    
    print(f"Loaded image: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min()}, {image.max()}]")
    
    return image


def show_side_by_side(original: np.ndarray, disturbed: np.ndarray, window_name: str):
    """
    Display original and disturbed images side by side.
    
    Args:
        original: Original image
        disturbed: Disturbed image  
        window_name: Window title
    """
    # Concatenate images horizontally
    if hasattr(disturbed, 'cpu'):
        disturbed = disturbed.cpu().numpy()
    combined = np.hstack([original, disturbed])
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)
    thickness = 2
    
    # Add "Original" label
    cv2.putText(combined, "Original", (10, 30), font, font_scale, color, thickness)
    
    # Add "Disturbed" label
    cv2.putText(combined, "Disturbed", (original.shape[1] + 10, 30), font, font_scale, color, thickness)
    
    cv2.imshow(window_name, combined)


def apply_and_show_disturbance(image: np.ndarray, disturbance_name: str, 
                             apply_func, args: DisturbanceTestArgs):
    """
    Apply a disturbance and show the result.
    
    Args:
        image: Original image
        disturbance_name: Name for display
        apply_func: Disturbance function to call
        args: Configuration arguments
    """
    print(f"Applying {disturbance_name}...")
    disturbed_image = apply_func(image.copy())
    show_side_by_side(image, disturbed_image, disturbance_name)
    cv2.waitKey(args.display_time)


def main():
    """Main function for disturbance testing."""
    args = tyro.cli(DisturbanceTestArgs)
    
    print("=== Lenna Image Disturbance Test ===")
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    # Load test image
    image = load_test_image(args.image_path)
    if image is None:
        print("Failed to load test image. Exiting.")
        return
    
    # Create disturbance wrapper with severity or custom parameters
    disturber = create_disturbance_wrapper(
        use_gpu=True,
        seed=args.seed, 
        severity=args.severity,
        gaussian_noise_sigma=args.gaussian_noise_sigma,
        gaussian_blur_sigma=args.gaussian_blur_sigma,
        contrast_range=(args.contrast_range_min, args.contrast_range_max),
        cutout_ratio=args.cutout_ratio
    )
    
    # Apply and display individual disturbances
    if args.apply_gaussian_noise:
        apply_and_show_disturbance(
            image,
            "Gaussian Noise",
            disturber.apply_gaussian_noise_numpy,
            args
        )
    
    if args.apply_contrast_jitter:
        apply_and_show_disturbance(
            image,
            "Contrast Jitter",
            disturber.apply_contrast_jitter_numpy,
            args
        )
    
    if args.apply_gaussian_blur:
        apply_and_show_disturbance(
            image,
            "Gaussian Blur",
            disturber.apply_gaussian_blur_numpy,
            args
        )
    
    if args.apply_cutout:
        apply_and_show_disturbance(
            image,
            "Cutout",
            disturber.apply_cutout_numpy,
            args
        )
    
    # Apply and display combined disturbances
    print("\nApplying combined disturbances...")
    combined_image = disturber.apply_disturbances_numpy(image)
    show_side_by_side(image, combined_image, "Combined Disturbances")
    
    print("\nPress any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
