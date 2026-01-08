"""
Prediction logging utilities.
Logs predictions to file for analytics and monitoring.
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd


class PredictionLogger:
    """Logger for model predictions."""
    
    def __init__(self, log_dir: str = "logs/predictions"):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory to save prediction logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "predictions.csv"
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'image_path',
                    'base_category',
                    'mi',
                    'usda',
                    'marbling_degree',
                    'jmga_bms',
                    'aus_meat',
                    'base_confidence',
                    'usda_confidence',
                    'bms_confidence'
                ])
    
    def log(self, prediction_result: Dict[str, Any]):
        """
        Log a prediction result.
        
        Args:
            prediction_result: Dictionary containing prediction results
        """
        timestamp = datetime.now().isoformat()
        
        image_info = prediction_result.get('image', {})
        prediction = prediction_result.get('prediction', {})
        confidence = prediction_result.get('confidence', {})
        
        row = [
            timestamp,
            image_info.get('path', ''),
            prediction.get('base_category', ''),
            prediction.get('mi', 0),
            prediction.get('usda', ''),
            prediction.get('marbling_degree', ''),
            prediction.get('jmga_bms', 0),
            prediction.get('aus_meat', 0),
            confidence.get('base', 0),
            confidence.get('usda', 0),
            confidence.get('bms', 0)
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        json_log = {
            'timestamp': timestamp,
            **prediction_result
        }
        
        json_path = self.log_dir / f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(json_path, 'w') as f:
            json.dump(json_log, f, indent=2, default=str)
    
    def get_recent_predictions(self, limit: int = 100) -> pd.DataFrame:
        """
        Get recent predictions.
        
        Args:
            limit: Maximum number of predictions to return
        
        Returns:
            DataFrame with recent predictions
        """
        if not self.csv_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.csv_path)
        return df.tail(limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get prediction statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.csv_path.exists():
            return {}
        
        df = pd.read_csv(self.csv_path)
        
        if len(df) == 0:
            return {}
        
        stats = {
            'total_predictions': len(df),
            'categories': df['base_category'].value_counts().to_dict(),
            'avg_confidence': df['base_confidence'].mean(),
            'avg_mi': df['mi'].mean(),
            'date_range': {
                'first': df['timestamp'].min(),
                'last': df['timestamp'].max()
            }
        }
        
        return stats


_default_logger: Optional[PredictionLogger] = None


def get_prediction_logger(log_dir: Optional[str] = None) -> PredictionLogger:
    """
    Get or create default prediction logger.
    
    Args:
        log_dir: Optional custom log directory
    
    Returns:
        PredictionLogger instance
    """
    global _default_logger
    if _default_logger is None or log_dir is not None:
        _default_logger = PredictionLogger(log_dir or "logs/predictions")
    return _default_logger

