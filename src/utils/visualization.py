"""
Visualization utilities for predictions.
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path


def draw_prediction_on_image(
    img: np.ndarray,
    prediction: Dict[str, Any],
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw prediction results on image.
    
    Args:
        img: Input image (RGB format)
        prediction: Prediction dictionary
        font_scale: Font scale for text
        thickness: Line thickness
    
    Returns:
        Image with predictions drawn
    """
    img_copy = img.copy()
    
    pred_data = prediction.get('prediction', {})
    confidence_data = prediction.get('confidence', {})
    
    base_category = pred_data.get('base_category', 'Unknown')
    usda = pred_data.get('usda', 'Unknown')
    bms = pred_data.get('jmga_bms', 0)
    mi = pred_data.get('mi', 0)
    confidence = confidence_data.get('base', 0)
    
    img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    
    color_primary = (0, 255, 0)
    color_secondary = (255, 255, 0)
    color_bg = (0, 0, 0)
    
    lines = [
        f"Category: {base_category}",
        f"USDA: {usda}",
        f"BMS: {bms}",
        f"MI: {mi:.4f}",
        f"Confidence: {confidence:.2%}"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 30
    padding = 10
    margin = 20
    
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(size[0] for size in text_sizes) + 2 * padding
    
    bg_height = len(lines) * line_height + 2 * padding
    top_left = (margin, margin)
    bottom_right = (margin + max_width, margin + bg_height)
    
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color_bg, -1)
    cv2.addWeighted(overlay, 0.7, img_bgr, 0.3, 0, img_bgr)
    
    y_offset = margin + padding + line_height
    for i, line in enumerate(lines):
        if i == 0:
            cv2.putText(
                img_bgr, line,
                (margin + padding, y_offset + i * line_height),
                font, font_scale, color_primary, thickness + 1
            )
        else:
            cv2.putText(
                img_bgr, line,
                (margin + padding, y_offset + i * line_height),
                font, font_scale * 0.9, color_secondary, thickness
            )
    
    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return result


def save_visualization(
    img: np.ndarray,
    prediction: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Save image with prediction visualization.
    
    Args:
        img: Input image (RGB format)
        prediction: Prediction dictionary
        output_path: Path to save visualization
    
    Returns:
        True if successful
    """
    try:
        visualized = draw_prediction_on_image(img, prediction)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        visualized_bgr = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), visualized_bgr)
        return True
    except Exception as e:
        print(f"Error saving visualization: {e}")
        return False


def create_comparison_image(
    img1: np.ndarray,
    pred1: Dict[str, Any],
    img2: np.ndarray,
    pred2: Dict[str, Any],
    label1: str = "Image 1",
    label2: str = "Image 2"
) -> np.ndarray:
    """
    Create side-by-side comparison of two predictions.
    
    Args:
        img1: First image (RGB)
        pred1: First prediction
        img2: Second image (RGB)
        pred2: Second prediction
        label1: Label for first image
        label2: Label for second image
    
    Returns:
        Combined comparison image
    """
    vis1 = draw_prediction_on_image(img1, pred1)
    vis2 = draw_prediction_on_image(img2, pred2)
    
    h1, w1 = vis1.shape[:2]
    h2, w2 = vis2.shape[:2]
    
    target_height = max(h1, h2)
    
    if h1 != target_height:
        scale = target_height / h1
        new_w = int(w1 * scale)
        vis1 = cv2.resize(vis1, (new_w, target_height))
    
    if h2 != target_height:
        scale = target_height / h2
        new_w = int(w2 * scale)
        vis2 = cv2.resize(vis2, (new_w, target_height))
    
    comparison = np.hstack([vis1, vis2])
    
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_bgr, label1, (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(
        comparison_bgr, label2,
        (vis1.shape[1] + 10, 30), font, 1, (255, 255, 255), 2
    )
    
    return cv2.cvtColor(comparison_bgr, cv2.COLOR_BGR2RGB)

