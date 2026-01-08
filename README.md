# CowinBMS - Beef Marbling Score Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

**Note**: This project requires a trained model file (`efficientNet_v1.pth`). Please ensure the model file is placed in `src/models/` directory before running inference.

AI-powered beef quality assessment system using computer vision and deep learning. Predicts beef marbling scores and converts them to multiple industry standards (USDA, JMGA BMS, AUS-MEAT).

## 🚀 Features

- **Multi-task Learning**: Predicts both Marbling Index (MI) and beef category classification
- **Multiple Standards**: Converts to USDA, JMGA BMS, and AUS-MEAT standards
- **Automatic Segmentation**: Handles raw images with automatic meat region segmentation
- **Batch Processing**: Process multiple images efficiently
- **REST API**: FastAPI-based REST API for easy integration
- **Visualization**: Automatic visualization of predictions on images
- **Comparison Mode**: Side-by-side comparison of two images
- **Export Options**: Export results to CSV, Excel, or JSON
- **Prediction Logging**: Track all predictions for analytics

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CowinBMS2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model file:
   - Place `efficientNet_v1.pth` in `src/models/` directory
   - Or update the model path in `configs/default.yaml`

## 📖 Usage

### Single Image Inference

Process a single image:
```bash
python src/inference/infer_input.py --image path/to/image.jpg
```

Using example test images:
```bash
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg
```

With segmentation disabled (if image is already segmented):
```bash
python src/inference/infer_input.py --image path/to/image.jpg --no-segmentation
```

### Batch Processing

Process all images in a directory:
```bash
python src/inference/infer_web_images.py
```

Images should be placed in `web_test_images/` directory (or use example images from `examples/test_images/`). Segmented results will be saved to `segmented_images/`.

### Comparison Mode

Compare two images side-by-side:
```bash
python src/inference/infer_compare.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

### Test Set Inference

Test on a random image from the test set:
```bash
python src/inference/infer_test.py
```

## 🌐 API Usage

### Start the API Server

```bash
python src/api/app.py
```

Or using uvicorn directly:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API documentation will be available at `http://localhost:8000/docs`

### API Endpoints

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "apply_segmentation=true" \
  -F "save_visualization=false"
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "apply_segmentation=true" \
  -F "export_format=csv"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## 📊 Output Format

### JSON Response

```json
{
  "image": {
    "path": "path/to/image.jpg",
    "filename": "image.jpg"
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

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

- Model settings (path, backbone, classes)
- Paths (data directories, output directories)
- Segmentation settings (method, thresholds)
- Inference settings (batch size, confidence thresholds)
- API settings (host, port, rate limits)
- Logging settings

## 📁 Project Structure

```
CowinBMS2/
├── src/
│   ├── api/              # FastAPI REST API
│   ├── features/         # Feature engineering (transforms, rules, segmentation)
│   ├── inference/        # Inference scripts
│   ├── models/           # Model definitions
│   └── utils/            # Utilities (config, logging, validation, etc.)
├── configs/              # Configuration files
├── examples/              # Example images and sample outputs
│   ├── test_images/      # Sample test images
│   ├── sample_outputs/   # Example prediction outputs
│   └── metrics/           # Performance metrics and reports
├── web_test_images/       # Input test images (user directory)
├── segmented_images/       # Segmented output images (generated)
├── outputs/               # Outputs (visualizations, exports, comparisons)
│   ├── visualizations/    # Prediction visualizations
│   ├── comparisons/       # Comparison images
│   └── exports/           # Exported results (CSV, Excel, JSON)
└── logs/                  # Log files
    └── predictions/       # Prediction logs
```

## 🔍 Model Architecture

- **Backbone**: EfficientNet-B0 (via timm)
- **Task 1**: Marbling Index regression (1280 → 256 → 1)
- **Task 2**: 5-class classification (1280 → 5)
  - Classes: Select, Choice, Prime, Wagyu, Japanese A5

## 🎯 Prediction Pipeline

1. **Image Loading**: Load and validate input image
2. **Segmentation** (optional): Extract meat region from raw image
3. **Preprocessing**: Resize and normalize for model input
4. **Inference**: Run model to get MI and class predictions
5. **Post-processing**: Apply rule-based conversions:
   - Base category → Marbling degree
   - Marbling degree → USDA grade
   - Base category + MI → BMS score
   - Base category + MI → AUS-MEAT score
6. **Output**: Generate results with confidence scores

## 📝 Notes

- **Segmentation**: If your images are raw (not segmented), keep segmentation enabled. If images are already segmented, disable it for better performance.
- **Confidence Thresholds**: Low confidence predictions (< 0.5) will generate warnings in the output.
- **Rate Limiting**: API endpoints have rate limits (10/minute for single, 5/minute for batch).

## 🧪 Testing

See [TESTING.md](TESTING.md) for comprehensive testing guide.

Quick test:
```bash
# Single image
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg

# API health check
curl http://localhost:8000/health
```

## 🛠️ Development

### Adding New Features

1. Feature code goes in `src/features/`
2. Utilities in `src/utils/`
3. Update config in `configs/default.yaml`
4. Add tests (when available)

### Logging

All predictions are logged to `logs/predictions/predictions.csv` for analytics.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- EfficientNet implementation from `timm`
- Image augmentation from `albumentations`
