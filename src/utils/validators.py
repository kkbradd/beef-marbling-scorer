"""
Input validation utilities.
"""
import os
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np


class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_image_path(image_path: str) -> str:
    """
    Validate and sanitize image path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Sanitized absolute path
    
    Raises:
        ValidationError: If path is invalid
    """
    if not image_path:
        raise ValidationError("Image path cannot be empty")
    
    path = Path(image_path)
    
    if ".." in str(path) or str(path).startswith("/"):
        if not path.exists():
            raise ValidationError(f"Invalid path: {image_path}")
    
    abs_path = path.resolve()
    
    if not abs_path.exists():
        raise ValidationError(f"Image file not found: {abs_path}")
    
    if not abs_path.is_file():
        raise ValidationError(f"Path is not a file: {abs_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if abs_path.suffix.lower() not in valid_extensions:
        raise ValidationError(f"Invalid image format: {abs_path.suffix}")
    
    return str(abs_path)


def validate_image_size(
    img: np.ndarray,
    min_size: Tuple[int, int] = (100, 100),
    max_size: Tuple[int, int] = (10000, 10000)
) -> bool:
    """
    Validate image dimensions.
    
    Args:
        img: Image array
        min_size: Minimum (width, height)
        max_size: Maximum (width, height)
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If image size is invalid
    """
    if img is None or len(img.shape) < 2:
        raise ValidationError("Invalid image: image is None or has invalid shape")
    
    height, width = img.shape[:2]
    
    if width < min_size[0] or height < min_size[1]:
        raise ValidationError(
            f"Image too small: {width}x{height}, minimum: {min_size[0]}x{min_size[1]}"
        )
    
    if width > max_size[0] or height > max_size[1]:
        raise ValidationError(
            f"Image too large: {width}x{height}, maximum: {max_size[0]}x{max_size[1]}"
        )
    
    return True


def validate_segmentation_mask(mask: np.ndarray, min_meat_ratio: float = 0.05) -> bool:
    """
    Validate segmentation mask quality.
    
    Args:
        mask: Binary mask (0 = background, 255 = meat)
        min_meat_ratio: Minimum ratio of meat pixels
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If segmentation quality is poor
    """
    if mask is None:
        raise ValidationError("Segmentation mask is None")
    
    total_pixels = mask.size
    meat_pixels = np.sum(mask > 0)
    meat_ratio = meat_pixels / total_pixels if total_pixels > 0 else 0
    
    if meat_ratio < min_meat_ratio:
        raise ValidationError(
            f"Segmentation failed: meat region too small ({meat_ratio:.2%}), "
            f"minimum required: {min_meat_ratio:.2%}"
        )
    
    if meat_ratio > 0.95:
        raise ValidationError(
            f"Segmentation suspicious: almost entire image marked as meat ({meat_ratio:.2%})"
        )
    
    return True


def safe_read_image(image_path: str) -> Optional[np.ndarray]:
    """
    Safely read image with validation.
    
    Args:
        image_path: Path to image
    
    Returns:
        Image array or None if failed
    
    Raises:
        ValidationError: If image cannot be read or is invalid
    """
    validated_path = validate_image_path(image_path)
    
    img = cv2.imread(validated_path)
    if img is None:
        raise ValidationError(f"Could not read image file: {validated_path}")
    
    validate_image_size(img)
    
    return img

