# Examples Directory

This directory contains example files organized by category for testing, demonstration, and reference purposes.

## 📁 Structure

```
examples/
├── test_images/          # Test images for inference
├── metrics/              # Performance metrics and screenshots
│   └── screenshots/      # Application screenshots
├── results/              # Generated results from inference
│   ├── visualizations/   # Prediction visualizations
│   ├── comparisons/      # Side-by-side comparison images
│   ├── exports/          # Exported results (CSV, JSON)
│   └── segmented/        # Segmented meat region images
└── sample_outputs/       # Sample prediction output formats
```

## 📸 Test Images

Sample test images for inference testing.

**Location:** `examples/test_images/`

**Contents:**
- `wagyu-ribeye.jpg` - Wagyu beef sample
- `iStock-844693654_4_480x480.jpeg` - Stock image sample
- `images.jpeg` - Additional test image

**Usage:**
```bash
# Run inference on example image
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg

# Run batch processing
python src/inference/infer_web_images.py
```

## 📊 Metrics

Performance metrics, evaluation reports, and application screenshots.

**Location:** `examples/metrics/`

**Contents:**
- `sample_predictions.csv` - Sample prediction logs
- `sample_prediction.json` - Individual prediction result example
- `screenshots/` - Application UI screenshots

**Metrics Files:**
```bash
# View sample prediction log
cat examples/metrics/sample_predictions.csv

# View individual prediction result
cat examples/metrics/sample_prediction.json
```

**Screenshots:**
The `screenshots/` directory contains:
- API documentation interface screenshots
- Prediction results visualization examples
- Comparison mode outputs
- Batch processing result demonstrations

## 📤 Results

Generated results from various inference operations, organized by type.

### Visualizations
**Location:** `examples/results/visualizations/`

Prediction visualizations with overlaid results (category, USDA, BMS, MI, confidence).

### Comparisons
**Location:** `examples/results/comparisons/`

Side-by-side comparison images showing two predictions together.

### Exports
**Location:** `examples/results/exports/`

Exported results in various formats:
- JSON files with batch predictions
- CSV files with prediction data
- Excel files (if available)

### Segmented Images
**Location:** `examples/results/segmented/`

Segmented meat region images extracted from raw input images using automatic segmentation.

**Usage:**
```bash
# View segmented results
ls examples/results/segmented/

# Compare original and segmented
open examples/test_images/wagyu-ribeye.jpg
open examples/results/segmented/wagyu-ribeye.jpg
```

## 📋 Sample Outputs

Reference examples of output format structure.

**Location:** `examples/sample_outputs/`

**Contents:**
- JSON files showing prediction result format
- Example API response structures

**Example Output Format:**
```json
{
  "image": {
    "path": "...",
    "filename": "..."
  },
  "prediction": {
    "base_category": "Prime",
    "mi": 0.0523,
    "usda": "Prime+",
    "marbling_degree": "Moderately Abundant",
    "jmga_bms": 2,
    "aus_meat": 4
  },
  "confidence": {
    "base": 0.856,
    "usda": 0.856,
    "bms": 0.893
  },
  "warnings": []
}
```

## 🔍 Quick Reference

### Viewing Results

```bash
# List all test images
ls examples/test_images/

# View prediction logs
cat examples/metrics/sample_predictions.csv

# View visualizations
ls examples/results/visualizations/

# View comparisons
ls examples/results/comparisons/

# View segmented images
ls examples/results/segmented/
```

### Testing Workflow

1. **Use test images** from `examples/test_images/` for inference
2. **Check results** in `examples/results/` for outputs
3. **Review metrics** in `examples/metrics/` for performance data
4. **Reference formats** in `examples/sample_outputs/` for integration

## 📝 Notes

- These are example files for reference and demonstration
- Actual outputs will be generated in project root directories:
  - `web_test_images/` - User input images
  - `outputs/` - Generated outputs
  - `segmented_images/` - Segmented results
  - `logs/` - Application logs
- To test with your own images, place them in `web_test_images/` directory
- All example files are tracked in Git for GitHub repository
