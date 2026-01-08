"""
Unified inference engine with caching and error handling.
"""
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from functools import lru_cache

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cowin_model import CowinBMSModel
from src.features.transforms import valid_tfms
from src.features.rules import (
    choose_marbling,
    marbling_to_usda,
    assign_bms,
    assign_aus_meat
)
from src.features.segmentation import preprocess_for_inference
from src.utils.config import Config
from src.utils.validators import (
    safe_read_image,
    validate_image_path,
    ValidationError
)
from src.utils.logging_config import get_logger
from src.utils.prediction_logger import get_prediction_logger
from src.utils.visualization import save_visualization

logger = get_logger("inference")
prediction_logger = get_prediction_logger()


class InferenceEngine:
    """Unified inference engine with caching and error handling."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = self.config.class_names
        
        logger.info(f"Initializing InferenceEngine on device: {self.device}")
    
    @lru_cache(maxsize=1)
    def load_model(self) -> CowinBMSModel:
        """
        Load model with caching.
        
        Returns:
            Loaded model
        """
        if self.model is None:
            model_path = self.config.get_absolute_path(self.config.model_path)
            logger.info(f"Loading model from: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = CowinBMSModel(
                num_classes=self.config.num_classes,
                backbone_name=self.config.backbone_name
            ).to(self.device)
            
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            
            logger.info("Model loaded successfully")
        
        return self.model
    
    def predict(
        self,
        image_path: str,
        apply_segmentation: Optional[bool] = None,
        save_visualization: Optional[bool] = None,
        log_prediction: bool = True
    ) -> Dict[str, Any]:
        """
        Run prediction on single image.
        
        Args:
            image_path: Path to image file
            apply_segmentation: Whether to apply segmentation. If None, uses config.
            save_visualization: Whether to save visualization. If None, uses config.
            log_prediction: Whether to log prediction
        
        Returns:
            Prediction result dictionary
        
        Raises:
            ValidationError: If image validation fails
            ValueError: If prediction fails
        """
        try:
            validated_path = validate_image_path(image_path)
            img_bgr = safe_read_image(validated_path)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            if apply_segmentation is None:
                apply_segmentation = self.config.get("segmentation.enabled", True)
            
            seg_method = self.config.get("segmentation.method", "color_based")
            min_meat_ratio = self.config.get("segmentation.min_meat_ratio", 0.05)
            
            if apply_segmentation:
                use_validation = True
                try:
                    from src.utils.validators import validate_segmentation_mask, ValidationError
                except ImportError:
                    use_validation = False
                
                img = preprocess_for_inference(
                    img,
                    apply_segmentation=True,
                    method=seg_method,
                    validate=use_validation,
                    min_meat_ratio=min_meat_ratio
                )
            
            model = self.load_model()
            
            img_tensor = valid_tfms(image=img)["image"]
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                mi_pred, logits = model(img_tensor)
            
            mi_pred = float(np.clip(mi_pred.item(), 0.0, 1.0))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_idx = int(np.argmax(probs))
            pred_dataset = self.class_names[pred_idx]
            base_conf = float(probs[pred_idx])
            
            conf_threshold = self.config.get("inference.confidence_threshold", 0.5)
            if base_conf < conf_threshold:
                logger.warning(
                    f"Low confidence prediction ({base_conf:.3f}) for image: {validated_path}"
                )
            
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
                    "path": validated_path,
                    "filename": os.path.basename(validated_path)
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
                },
                "warnings": []
            }
            
            if base_conf < conf_threshold:
                result["warnings"].append(f"Low confidence: {base_conf:.3f}")
            
            if save_visualization is None:
                save_visualization = self.config.get("inference.save_visualizations", True)
            
            if save_visualization:
                vis_dir = self.config.get_absolute_path(
                    self.config.get("inference.visualization_output_dir", "outputs/visualizations")
                )
                vis_path = vis_dir / f"vis_{Path(validated_path).stem}.jpg"
                save_visualization(img, result, str(vis_path))
                result["visualization_path"] = str(vis_path)
            
            if log_prediction:
                prediction_logger.log(result)
            
            return result
        
        except ValidationError as e:
            logger.error(f"Validation error for {image_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {e}", exc_info=True)
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def batch_predict(
        self,
        image_paths: List[str],
        apply_segmentation: Optional[bool] = None,
        save_visualizations: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Run prediction on multiple images.
        
        Args:
            image_paths: List of image paths
            apply_segmentation: Whether to apply segmentation
            save_visualizations: Whether to save visualizations
        
        Returns:
            List of prediction results
        """
        results = []
        errors = []
        
        for img_path in image_paths:
            try:
                result = self.predict(
                    img_path,
                    apply_segmentation=apply_segmentation,
                    save_visualization=save_visualizations
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                errors.append({"image": img_path, "error": str(e)})
                results.append(None)
        
        return results, errors


_engine_instance: Optional[InferenceEngine] = None


def get_engine(config: Optional[Config] = None) -> InferenceEngine:
    """
    Get or create global inference engine instance.
    
    Args:
        config: Optional config. Only used on first call.
    
    Returns:
        InferenceEngine instance
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InferenceEngine(config)
    return _engine_instance

