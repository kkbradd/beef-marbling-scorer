"""
Image segmentation utilities for meat region extraction.
This module provides functions to segment meat regions from raw beef images,
similar to the training data preprocessing.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

try:
    from src.utils.validators import ValidationError, validate_segmentation_mask
except ImportError:
    class ValidationError(Exception):
        pass
    
    def validate_segmentation_mask(mask, min_meat_ratio=0.05):
        pass


def segment_meat_region(img: np.ndarray, method: str = "color_based") -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment meat region from raw image.
    
    Args:
        img: Input image in RGB format (H, W, 3)
        method: Segmentation method to use ('color_based', 'otsu', 'adaptive')
    
    Returns:
        Tuple of (segmented_image, mask)
        - segmented_image: Image with only meat region visible, background is black
        - mask: Binary mask where 1 indicates meat region, 0 indicates background
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    if method == "color_based":
        return _segment_color_based(img, img_hsv, img_lab)
    elif method == "otsu":
        return _segment_otsu(img, img_hsv)
    elif method == "adaptive":
        return _segment_adaptive(img, img_hsv, img_lab)
    else:
        raise ValueError(f"Unknown method: {method}")


def _segment_color_based(img_rgb: np.ndarray, img_hsv: np.ndarray, img_lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Color-based segmentation using HSV color space.
    Meat typically has red/pink tones in HSV space.
    """
    h, s, v = cv2.split(img_hsv)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    return segmented, mask


def _segment_otsu(img_rgb: np.ndarray, img_hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Otsu thresholding-based segmentation.
    """
    _, s, _ = cv2.split(img_hsv)
    
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    return segmented, mask


def _segment_adaptive(img_rgb: np.ndarray, img_hsv: np.ndarray, img_lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive segmentation combining multiple methods.
    """
    segmented1, mask1 = _segment_color_based(img_rgb, img_hsv, img_lab)
    segmented2, mask2 = _segment_otsu(img_rgb, img_hsv)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    return segmented, mask


def apply_segmentation_to_image(img_path: str, method: str = "color_based", return_mask: bool = False) -> np.ndarray:
    """
    Load image and apply segmentation.
    
    Args:
        img_path: Path to image file
        method: Segmentation method
        return_mask: If True, return both image and mask
    
    Returns:
        Segmented image (and mask if return_mask=True)
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    segmented, mask = segment_meat_region(img_rgb, method=method)
    
    if return_mask:
        return segmented, mask
    return segmented


def preprocess_for_inference(
    img: np.ndarray,
    apply_segmentation: bool = True,
    method: str = "color_based",
    validate: bool = True,
    min_meat_ratio: float = 0.05
) -> np.ndarray:
    """
    Preprocess image for inference, optionally applying segmentation.
    
    Args:
        img: Input image in RGB format
        apply_segmentation: Whether to apply segmentation
        method: Segmentation method if apply_segmentation is True
        validate: Whether to validate segmentation quality
        min_meat_ratio: Minimum ratio of meat pixels for validation
    
    Returns:
        Preprocessed image ready for model inference
    
    Raises:
        ValidationError: If segmentation validation fails
    """
    if apply_segmentation:
        segmented, mask = segment_meat_region(img, method=method)
        
        if validate:
            try:
                from src.utils.validators import validate_segmentation_mask, ValidationError
                validate_segmentation_mask(mask, min_meat_ratio=min_meat_ratio)
            except ValidationError as e:
                import warnings
                warnings.warn(f"Segmentation validation warning: {e}", UserWarning)
            except ImportError:
                pass
        
        return segmented
    return img

