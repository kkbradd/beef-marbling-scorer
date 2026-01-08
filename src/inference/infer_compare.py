"""
Comparison mode: Compare two images side by side.
"""
import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference_engine import get_engine
from src.utils.config import load_config
from src.utils.visualization import create_comparison_image, save_visualization
from src.utils.helpers import to_builtin
import cv2


def main():
    """
    Compare two images and show side-by-side results.
    """
    parser = argparse.ArgumentParser(
        description="Compare two images side-by-side"
    )
    
    parser.add_argument(
        "--image1",
        type=str,
        required=True,
        help="Path to first image"
    )
    
    parser.add_argument(
        "--image2",
        type=str,
        required=True,
        help="Path to second image"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison image (default: outputs/comparisons/)"
    )
    
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable segmentation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    engine = get_engine(config)
    
    print("Processing first image...")
    try:
        result1 = engine.predict(
            args.image1,
            apply_segmentation=not args.no_segmentation,
            save_visualization=False,
            log_prediction=False
        )
    except Exception as e:
        print(f"❌ Error processing image 1: {e}")
        return
    
    print("Processing second image...")
    try:
        result2 = engine.predict(
            args.image2,
            apply_segmentation=not args.no_segmentation,
            save_visualization=False,
            log_prediction=False
        )
    except Exception as e:
        print(f"❌ Error processing image 2: {e}")
        return
    
    img1 = cv2.imread(args.image1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread(args.image2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    label1 = f"{Path(args.image1).stem}\n{result1['prediction']['base_category']} ({result1['prediction']['usda']})"
    label2 = f"{Path(args.image2).stem}\n{result2['prediction']['base_category']} ({result2['prediction']['usda']})"
    
    comparison = create_comparison_image(img1, result1, img2, result2, label1, label2)
    
    if args.output:
        output_path = args.output
    else:
        output_dir = config.get_absolute_path("outputs/comparisons")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"compare_{Path(args.image1).stem}_{Path(args.image2).stem}.jpg"
    
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), comparison_bgr)
    
    print(f"\n✅ Comparison saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n📸 Image 1: {Path(args.image1).name}")
    print(f"   Category: {result1['prediction']['base_category']}")
    print(f"   USDA: {result1['prediction']['usda']}")
    print(f"   BMS: {result1['prediction']['jmga_bms']}")
    print(f"   MI: {result1['prediction']['mi']:.4f}")
    print(f"   Confidence: {result1['confidence']['base']:.3f}")
    
    print(f"\n📸 Image 2: {Path(args.image2).name}")
    print(f"   Category: {result2['prediction']['base_category']}")
    print(f"   USDA: {result2['prediction']['usda']}")
    print(f"   BMS: {result2['prediction']['jmga_bms']}")
    print(f"   MI: {result2['prediction']['mi']:.4f}")
    print(f"   Confidence: {result2['confidence']['base']:.3f}")
    
    comparison_result = {
        "comparison": {
            "image1": {
                "path": args.image1,
                "prediction": result1['prediction'],
                "confidence": result1['confidence']
            },
            "image2": {
                "path": args.image2,
                "prediction": result2['prediction'],
                "confidence": result2['confidence']
            },
            "output_path": str(output_path)
        }
    }
    
    print("\n📄 JSON Output:")
    print("=" * 80)
    print(json.dumps(comparison_result, indent=2, default=to_builtin))


if __name__ == "__main__":
    main()

