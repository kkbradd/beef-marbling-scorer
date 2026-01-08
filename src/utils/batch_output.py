"""
Batch output utilities for exporting predictions to various formats.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def predictions_to_dataframe(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of predictions to pandas DataFrame.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        DataFrame with predictions
    """
    rows = []
    
    for pred in predictions:
        if pred is None:
            continue
        
        image_info = pred.get('image', {})
        prediction = pred.get('prediction', {})
        confidence = pred.get('confidence', {})
        
        row = {
            'image_path': image_info.get('path', ''),
            'filename': image_info.get('filename', ''),
            'base_category': prediction.get('base_category', ''),
            'mi': prediction.get('mi', 0),
            'usda': prediction.get('usda', ''),
            'marbling_degree': prediction.get('marbling_degree', ''),
            'jmga_bms': prediction.get('jmga_bms', 0),
            'aus_meat': prediction.get('aus_meat', 0),
            'base_confidence': confidence.get('base', 0),
            'usda_confidence': confidence.get('usda', 0),
            'bms_confidence': confidence.get('bms', 0),
            'warnings': '; '.join(pred.get('warnings', []))
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def export_to_csv(
    predictions: List[Dict[str, Any]],
    output_path: str,
    include_index: bool = False
) -> bool:
    """
    Export predictions to CSV file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output CSV file
        include_index: Whether to include index column
    
    Returns:
        True if successful
    """
    try:
        df = predictions_to_dataframe(predictions)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=include_index)
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False


def export_to_excel(
    predictions: List[Dict[str, Any]],
    output_path: str,
    sheet_name: str = "Predictions"
) -> bool:
    """
    Export predictions to Excel file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output Excel file
        sheet_name: Name of the Excel sheet
    
    Returns:
        True if successful
    """
    try:
        df = predictions_to_dataframe(predictions)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
        
        return True
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False


def export_to_json(
    predictions: List[Dict[str, Any]],
    output_path: str,
    indent: int = 2
) -> bool:
    """
    Export predictions to JSON file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output JSON file
        indent: JSON indentation
    
    Returns:
        True if successful
    """
    import json
    
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_predictions': len([p for p in predictions if p is not None]),
            'predictions': [p for p in predictions if p is not None]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=indent, default=str)
        
        return True
    except Exception as e:
        print(f"Error exporting to JSON: {e}")
        return False


def export_batch_results(
    predictions: List[Dict[str, Any]],
    output_dir: str,
    base_name: Optional[str] = None,
    formats: List[str] = ['csv', 'excel', 'json']
) -> Dict[str, str]:
    """
    Export batch results to multiple formats.
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Output directory
        base_name: Base name for output files (without extension)
        formats: List of formats to export ('csv', 'excel', 'json')
    
    Returns:
        Dictionary mapping format to output file path
    """
    if base_name is None:
        base_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    if 'csv' in formats:
        csv_path = output_dir / f"{base_name}.csv"
        if export_to_csv(predictions, str(csv_path)):
            exported_files['csv'] = str(csv_path)
    
    if 'excel' in formats:
        excel_path = output_dir / f"{base_name}.xlsx"
        if export_to_excel(predictions, str(excel_path)):
            exported_files['excel'] = str(excel_path)
    
    if 'json' in formats:
        json_path = output_dir / f"{base_name}.json"
        if export_to_json(predictions, str(json_path)):
            exported_files['json'] = str(json_path)
    
    return exported_files

