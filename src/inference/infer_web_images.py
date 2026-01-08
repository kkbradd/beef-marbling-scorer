import os
import sys
import json
import glob
import cv2
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.cowin_model import CowinBMSModel
from src.features.transforms import valid_tfms
from src.features.rules import choose_marbling, marbling_to_usda, assign_bms, assign_aus_meat
from src.features.segmentation import preprocess_for_inference
from src.utils.helpers import to_builtin

CLASS_NAMES = ["Select", "Choice", "Prime", "Wagyu", "Japanese A5"]

IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.JPG', '*.JPEG', '*.PNG']


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def process_image(img_path: str, model: CowinBMSModel, device: torch.device, apply_segmentation: bool = True):
    """
    Tek bir görseli işleyip sonuçları döndürür.
    
    Args:
        img_path: Path to image
        model: Loaded model
        device: Device to run inference on
        apply_segmentation: Whether to apply segmentation (recommended for raw web images)
    
    Returns:
        Tuple of (result_dict, error_message, segmented_image)
        - result_dict: Prediction results
        - error_message: Error if any, None otherwise
        - segmented_image: Segmented image (RGB format) if segmentation was applied, None otherwise
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, f"Could not read image: {img_path}", None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segmented_img = None
    
    if apply_segmentation:
        img = preprocess_for_inference(img, apply_segmentation=True, method="color_based")
        segmented_img = img.copy()
    
    x = valid_tfms(image=img)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        mi_pred, logits = model(x)
    
    mi_pred = float(mi_pred.item())
    mi_pred = float(np.clip(mi_pred, 0.0, 1.0))
    
    probs = softmax_np(logits)[0]
    pred_idx = int(np.argmax(probs))
    pred_dataset = CLASS_NAMES[pred_idx]
    base_conf = float(probs[pred_idx])
    
    marbling = choose_marbling(pred_dataset, mi_pred)
    usda = marbling_to_usda(marbling, pred_dataset, mi_pred)
    
    tmp_df = pd.DataFrame({"dataset_label": [pred_dataset], "mi_scaled": [mi_pred]})
    bms = int(assign_bms(tmp_df)["bms_score"].iloc[0])
    aus = int(assign_aus_meat(tmp_df)["aus_meat_score"].iloc[0])
    
    usda_conf = float(np.max(probs))
    bms_conf = float(np.clip(1.0 - abs(mi_pred - 0.5) * 1.2, 0.55, 0.95))
    
    result = {
        "image": {
            "path": img_path,
            "filename": os.path.basename(img_path)
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
    
    return result, None, segmented_img


def main():
    """
    Process all images in web_test_images or examples/test_images directory.
    
    This script automatically applies segmentation because images from the web
    are typically raw (not segmented). Since the model was trained on segmented
    images, segmentation should be applied for reliable results.
    
    Processed images:
    - Reads all supported image formats from web_test_images/ or examples/test_images/
    - Applies segmentation (saved to segmented_images/)
    - Runs inference on each image
    - Displays results in console
    - Exports results in JSON format
    
    Outputs:
    - Segmented images saved to segmented_images/
    - JSON results printed to console
    
    Raises:
        FileNotFoundError: If image directory not found
        ValueError: If no images found in directory
    """
    web_test_dir = os.path.join(PROJECT_ROOT, "web_test_images")
    examples_test_dir = os.path.join(PROJECT_ROOT, "examples/test_images")
    model_path = os.path.join(PROJECT_ROOT, "src/models/efficientNet_v1.pth")
    apply_segmentation = True
    
    if not os.path.exists(web_test_dir) or len(glob.glob(os.path.join(web_test_dir, "*.*"))) == 0:
        if os.path.exists(examples_test_dir) and len(glob.glob(os.path.join(examples_test_dir, "*.*"))) > 0:
            web_test_dir = examples_test_dir
            print(f"⚠️  Using example test images from: {web_test_dir}")
        else:
            print(f"Error: No images found in web_test_images/")
            print(f"Please create web_test_images/ directory and add test images to it.")
            print(f"Or ensure example images exist in: examples/test_images/")
            return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    model = CowinBMSModel().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(web_test_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(web_test_dir, ext.upper())))
    
    if len(image_paths) == 0:
        print(f"No images found in {web_test_dir}")
        print(f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        return
    
    print(f"\nFound {len(image_paths)} image(s) to process...")
    if apply_segmentation:
        print("⚠️  Segmentation will be applied to match training data format.")
    
    segmented_output_dir = os.path.join(PROJECT_ROOT, "segmented_images")
    if apply_segmentation:
        os.makedirs(segmented_output_dir, exist_ok=True)
        print(f"💾 Segmented images will be saved to: {segmented_output_dir}")
    
    print("=" * 80)
    
    results = []
    errors = []
    
    for idx, img_path in enumerate(sorted(image_paths), 1):
        image_filename = os.path.basename(img_path)
        print(f"\n[{idx}/{len(image_paths)}] Processing: {image_filename}")
        
        result, error, segmented_img = process_image(img_path, model, device, apply_segmentation=apply_segmentation)
        
        if error:
            print(f"  ❌ Error: {error}")
            errors.append({"image": img_path, "filename": image_filename, "error": error})
        else:
            if apply_segmentation and segmented_img is not None:
                segmented_output_path = os.path.join(segmented_output_dir, image_filename)
                segmented_bgr = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(segmented_output_path, segmented_bgr)
                print(f"  💾 Saved segmented image: {segmented_output_path}")
            
            print(f"  📸 Image: {image_filename}")
            print(f"  ✅ Base Category: {result['prediction']['base_category']}")
            print(f"  ✅ USDA: {result['prediction']['usda']}")
            print(f"  ✅ Marbling: {result['prediction']['marbling_degree']}")
            print(f"  ✅ BMS: {result['prediction']['jmga_bms']}")
            print(f"  ✅ AUS-MEAT: {result['prediction']['aus_meat']}")
            print(f"  ✅ MI: {result['prediction']['mi']}")
            print(f"  ✅ Confidence: {result['confidence']['base']:.3f}")
            results.append(result)
    
    print("\n" + "=" * 80)
    print("\n📊 SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(image_paths)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    if apply_segmentation:
        print(f"💾 Segmented images saved to: {segmented_output_dir}")
    
    if errors:
        print("\n❌ Errors:")
        for err in errors:
            filename = err.get('filename', os.path.basename(err['image']))
            print(f"  - {filename}: {err['error']}")
    
    if results:
        print("\n📋 Processed Images:")
        print("-" * 80)
        for res in results:
            print(f"  • {res['image']['filename']} → {res['prediction']['base_category']} ({res['prediction']['usda']})")
    
    if results:
        print("\n📄 Full Results (JSON):")
        print("=" * 80)
        output = {
            "summary": {
                "total_images": len(image_paths),
                "successful": len(results),
                "errors": len(errors)
            },
            "results": results
        }
        if errors:
            output["errors"] = errors
        
        print(json.dumps(output, indent=2, default=to_builtin))


if __name__ == "__main__":
    main()

