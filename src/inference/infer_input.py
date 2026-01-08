import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.cowin_model import CowinBMSModel
from src.features.transforms import valid_tfms
from src.features.rules import (
    choose_marbling,
    marbling_to_usda,
    assign_bms,
    assign_aus_meat
)
from src.features.segmentation import preprocess_for_inference
from src.utils.helpers import to_builtin

CLASS_NAMES = ["Select", "Choice", "Prime", "Wagyu", "Japanese A5"]
MODEL_PATH = os.path.join(PROJECT_ROOT, "src/models/efficientNet_v1.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_image_path(data_dir: str, images_dir: str, rel_path: str) -> str:
    """
    Try multiple path combinations to find the image.
    
    Args:
        data_dir: Base data directory
        images_dir: Images subdirectory
        rel_path: Relative path or filename
    
    Returns:
        Absolute path to found image
    
    Raises:
        FileNotFoundError: If image cannot be found
    """
    rel_path = str(rel_path).lstrip("/")
    cands = [
        os.path.join(data_dir, rel_path),
        os.path.join(images_dir, rel_path),
        os.path.join(images_dir, os.path.basename(rel_path)),
        rel_path,  # Absolute path olarak da dene
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found. Tried: {cands}")


def main(image_path: str, apply_segmentation: bool = True):
    """
    Single image inference.
    
    Processes a single image and returns prediction results including:
    - Base category (Select, Choice, Prime, Wagyu, Japanese A5)
    - Marbling Index (MI)
    - USDA grade
    - JMGA BMS score
    - AUS-MEAT score
    - Confidence scores
    
    Args:
        image_path: Path to input image file
        apply_segmentation: Whether to apply segmentation (recommended for raw images)
    
    Raises:
        ValueError: If image_path is None
        FileNotFoundError: If image file not found
        ValueError: If image cannot be read or processed
    """
    if image_path is None:
        raise ValueError("--image argument is required")
    
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        data_dir = os.path.join(PROJECT_ROOT, "kaggle_dataset")
        images_dir = os.path.join(data_dir, "images")
        image_path = resolve_image_path(data_dir, images_dir, image_path)
    elif not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    model = CowinBMSModel().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image could not be read: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if apply_segmentation:
        img = preprocess_for_inference(img, apply_segmentation=True, method="color_based")

    img_tensor = valid_tfms(image=img)["image"]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        mi_pred, logits = model(img_tensor)

    mi_pred = float(np.clip(mi_pred.item(), 0.0, 1.0))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_dataset = CLASS_NAMES[pred_idx]
    base_conf = float(probs[pred_idx])

    marbling = choose_marbling(pred_dataset, mi_pred)
    usda = marbling_to_usda(marbling, pred_dataset, mi_pred)

    tmp_df = pd.DataFrame({
        "dataset_label": [pred_dataset],
        "mi_scaled": [mi_pred]
    })

    bms = int(assign_bms(tmp_df)["bms_score"].iloc[0])
    aus = int(assign_aus_meat(tmp_df)["aus_meat_score"].iloc[0])

    usda_conf = float(np.max(probs))
    bms_conf = float(np.clip(1.0 - abs(mi_pred - 0.5) * 1.2, 0.55, 0.95))

    result = {
        "image": {
            "path": image_path
        },
        "prediction": {
            "base_category": pred_dataset,
            "mi": round(mi_pred, 4),
            "usda": str(usda),
            "marbling_degree": str(marbling),
            "jmga_bms": int(bms),
            "aus_meat": int(aus)
        },
        "confidence": {
            "base": round(base_conf, 3),
            "usda": round(usda_conf, 3),
            "bms": round(bms_conf, 3)
        }
    }

    print(json.dumps(result, indent=2, default=to_builtin))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CowinBMS single image inference"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        dest="image",
        help="Path to input image"
    )
    
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        dest="no_segmentation",
        help="Disable automatic segmentation (use if image is already segmented)"
    )

    args = parser.parse_args()
    if args.image is None:
        parser.error("--image argument is required")
    
    apply_segmentation = not args.no_segmentation
    if not apply_segmentation:
        print("⚠️  Segmentation disabled. Make sure your image matches training data format.")

    main(args.image, apply_segmentation=apply_segmentation)

