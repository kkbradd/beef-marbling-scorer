# Beef Marbling Score Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/kkbradd/beef-marbling-scorer?style=social)](https://github.com/kkbradd/beef-marbling-scorer)
[![GitHub forks](https://img.shields.io/github/forks/kkbradd/beef-marbling-scorer?style=social)](https://github.com/kkbradd/beef-marbling-scorer)

AI-powered beef quality assessment system using computer vision and deep learning. Predicts beef marbling scores and converts them to multiple industry standards (USDA, JMGA BMS, AUS-MEAT).

## 🌟 Features

- **Multi-task Learning**: Predicts both Marbling Index (MI) and beef category classification
- **Multiple Standards**: Converts to USDA, JMGA BMS, and AUS-MEAT standards
- **Automatic Segmentation**: Handles raw images with automatic meat region segmentation
- **Batch Processing**: Process multiple images efficiently
- **REST API**: FastAPI-based REST API for easy integration
- **Visualization**: Automatic visualization of predictions on images
- **Comparison Mode**: Side-by-side comparison of two images
- **Export Options**: Export results to CSV, Excel, or JSON
- **Prediction Logging**: Track all predictions for analytics

## 📸 Example Results

### Visual Comparison

![Comparison Example](https://github.com/kkbradd/beef-marbling-scorer/blob/main/examples/results/comparisons/compare_wagyu-ribeye_iStock-844693654_4_480x480.jpg?raw=true)

Side-by-side comparison showing two beef samples with their predictions (Category, USDA grade, BMS score, MI, and Confidence).

### Sample Prediction Output

**Input:** `examples/test_images/wagyu-ribeye.jpg`

**Output:**
```json
{
  "image": {
    "path": "examples/test_images/wagyu-ribeye.jpg",
    "filename": "wagyu-ribeye.jpg"
  },
  "prediction": {
    "base_category": "Wagyu",
    "mi": 0.0165,
    "usda": "Beyond Prime",
    "marbling_degree": "Very Abundant",
    "jmga_bms": 3,
    "aus_meat": 6
  },
  "confidence": {
    "base": 0.658,
    "usda": 0.658,
    "bms": 0.55
  },
  "warnings": []
}
```

### Batch Export Example

```json
{
  "export_timestamp": "2026-01-07T10:42:30.791686",
  "total_predictions": 1,
  "predictions": [
    {
      "image": {
        "path": "/path/to/image.jpg",
        "filename": "image.jpg"
      },
      "prediction": {
        "base_category": "Wagyu",
        "mi": 0.0165,
        "usda": "Beyond Prime",
        "marbling_degree": "Very Abundant",
        "jmga_bms": 3,
        "aus_meat": 6
      },
      "confidence": {
        "base": 0.658,
        "usda": 0.658,
        "bms": 0.55
      },
      "warnings": []
    }
  ]
}
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/kkbradd/beef-marbling-scorer.git
cd beef-marbling-scorer
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
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg
```

**Example Output:**
```json
{
  "image": {
    "path": "examples/test_images/wagyu-ribeye.jpg"
  },
  "prediction": {
    "base_category": "Wagyu",
    "mi": 0.0165,
    "usda": "Beyond Prime",
    "marbling_degree": "Very Abundant",
    "jmga_bms": 3,
    "aus_meat": 6
  },
  "confidence": {
    "base": 0.658,
    "usda": 0.658,
    "bms": 0.55
  }
}
```

With segmentation disabled (if image is already segmented):
```bash
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg --no-segmentation
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
python src/inference/infer_compare.py \
  --image1 examples/test_images/wagyu-ribeye.jpg \
  --image2 examples/test_images/iStock-844693654_4_480x480.jpeg
```

Output will be saved to `outputs/comparisons/` directory.

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
  -F "file=@examples/test_images/wagyu-ribeye.jpg" \
  -F "apply_segmentation=true" \
  -F "save_visualization=false"
```

**Response:**
```json
{
  "image": {
    "path": "/tmp/...",
    "filename": "wagyu-ribeye.jpg"
  },
  "prediction": {
    "base_category": "Wagyu",
    "mi": 0.0165,
    "usda": "Beyond Prime",
    "marbling_degree": "Very Abundant",
    "jmga_bms": 3,
    "aus_meat": 6
  },
  "confidence": {
    "base": 0.658,
    "usda": 0.658,
    "bms": 0.55
  },
  "warnings": []
}
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch" \
  -F "files=@examples/test_images/wagyu-ribeye.jpg" \
  -F "files=@examples/test_images/iStock-844693654_4_480x480.jpeg" \
  -F "apply_segmentation=true" \
  -F "export_format=csv"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## 📊 Output Format

### Prediction Result Structure

```json
{
  "image": {
    "path": "path/to/image.jpg",
    "filename": "image.jpg"
  },
  "prediction": {
    "base_category": "Prime",           // Select, Choice, Prime, Wagyu, Japanese A5
    "mi": 0.0523,                       // Marbling Index (0-1)
    "usda": "Prime+",                   // USDA grade
    "marbling_degree": "Moderately Abundant",  // Marbling description
    "jmga_bms": 2,                      // JMGA BMS score (1-12)
    "aus_meat": 4                       // AUS-MEAT score (0-9)
  },
  "confidence": {
    "base": 0.856,                      // Base category confidence (0-1)
    "usda": 0.856,                      // USDA grade confidence (0-1)
    "bms": 0.893                        // BMS score confidence (0-1)
  },
  "warnings": []                        // Array of warnings (if any)
}
```

### Supported Categories

- **Select**: Lowest grade, minimal marbling
- **Choice**: Moderate marbling, good quality
- **Prime**: High marbling, premium quality
- **Wagyu**: Exceptional marbling, Japanese breed
- **Japanese A5**: Highest grade, maximum marbling

### Scoring Systems

1. **USDA Grades**: Select, Choice-, Choice, Choice+, Prime-, Prime, Prime+, Beyond Prime
2. **JMGA BMS**: 1-12 scale (Japanese Marbling Standard)
3. **AUS-MEAT**: 0-9 scale (Australian Meat Standards)

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
beef-marbling-scorer/
├── src/
│   ├── api/              # FastAPI REST API
│   ├── features/         # Feature engineering (transforms, rules, segmentation)
│   ├── inference/        # Inference scripts
│   ├── models/           # Model definitions
│   └── utils/            # Utilities (config, logging, validation, etc.)
├── configs/              # Configuration files
├── examples/             # Example images and sample outputs
│   ├── test_images/      # Sample test images
│   ├── sample_outputs/   # Example prediction outputs
│   ├── metrics/          # Performance metrics and reports
│   │   └── screenshots/  # Application screenshots
│   └── results/          # Generated results from inference
│       ├── visualizations/ # Prediction visualizations
│       ├── comparisons/    # Side-by-side comparison images
│       ├── exports/        # Exported results (CSV, JSON)
│       └── segmented/      # Segmented meat region images
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

## 📚 Example Results

Check the `examples/` directory for:
- **Test Images**: Sample beef images in `examples/test_images/`
- **Sample Outputs**: Example prediction outputs in `examples/sample_outputs/`
- **Generated Results**: 
  - Comparison images in `examples/results/comparisons/`
  - Segmented images in `examples/results/segmented/`
  - Export files in `examples/results/exports/`
- **Metrics**: Performance data in `examples/metrics/`
- **Screenshots**: Application UI screenshots in `examples/metrics/screenshots/`

## 🛠️ Development

### Adding New Features

1. Feature code goes in `src/features/`
2. Utilities in `src/utils/`
3. Update config in `configs/default.yaml`
4. Add tests (when available)

### Logging

All predictions are logged to `logs/predictions/predictions.csv` for analytics.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- EfficientNet implementation from `timm`
- Image augmentation from `albumentations`

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/kkbradd/beef-marbling-scorer/issues).

---

**Made with ❤️ for beef quality assessment**
