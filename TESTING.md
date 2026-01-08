# Testing Guide - CowinBMS

This document contains all commands and checkpoints needed to test the project.

## 📋 Prerequisites

### 1. Dependencies Check

```bash
# Check Python version (3.8+ required)
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Model File Check

```bash
# Ensure model file exists
ls -lh src/models/efficientNet_v1.pth
```

## 🖼️ Inference Script Tests

### Test 1: Single Image Inference (`infer_input.py`)

**Command:**

```bash
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg
```

**Checkpoints:**

- ✅ Is JSON output properly formatted?
- ✅ Is `base_category` correct? (Select, Choice, Prime, Wagyu, Japanese A5)
- ✅ Is `mi` value in range 0-1?
- ✅ Is `usda` value logical? (consistent with base_category?)
- ✅ Are `jmga_bms` and `aus_meat` scores logical?
- ✅ Are `confidence` values in range 0-1?

**Expected Output:**

```json
{
  "image": {
    "path": "..."
  },
  "prediction": {
    "base_category": "...",
    "mi": 0.xxxx,
    "usda": "...",
    "marbling_degree": "...",
    "jmga_bms": X,
    "aus_meat": X
  },
  "confidence": {
    "base": 0.xxx,
    "usda": 0.xxx,
    "bms": 0.xxx
  }
}
```

**Test Without Segmentation:**

```bash
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg --no-segmentation
```

### Test 2: Batch Processing (`infer_web_images.py`)

**Command:**

```bash
python src/inference/infer_web_images.py
```

**Checkpoints:**

- ✅ Were all images in `web_test_images/` or `examples/test_images/` processed?
- ✅ Are segmentation results in `segmented_images/`?
- ✅ Were results shown for each image?
- ✅ Are error messages clear if any?
- ✅ Are all results in JSON output?

**Expected Output:**

```
Found X image(s) to process...
⚠️  Segmentation will be applied to match training data format.
💾 Segmented images will be saved to: segmented_images/

[1/X] Processing: image1.jpg
  💾 Saved segmented image: segmented_images/image1.jpg
  📸 Image: image1.jpg
  ✅ Base Category: ...
  ...
```

**Check Commands:**

```bash
# Check segmentation results
ls -lh segmented_images/

# Visually check segmentation quality (open images)
open segmented_images/
```

### Test 3: Comparison Mode (`infer_compare.py`)

**Command:**

```bash
python src/inference/infer_compare.py \
  --image1 examples/test_images/wagyu-ribeye.jpg \
  --image2 examples/test_images/iStock-844693654_4_480x480.jpeg
```

**Checkpoints:**

- ✅ Was comparison image created?
- ✅ Do both images appear side-by-side correctly?
- ✅ Are prediction results shown for both images?
- ✅ Is comparison image in `outputs/comparisons/`?

**Expected Output:**

```
Processing first image...
Processing second image...

✅ Comparison saved to: outputs/comparisons/compare_...

COMPARISON SUMMARY
================================================================================

📸 Image 1: ...
   Category: ...
   USDA: ...
   ...
```

**Check Commands:**

```bash
# Open comparison image
ls outputs/comparisons/
open outputs/comparisons/*.jpg
```

### Test 4: Test Set Inference (`infer_test.py`)

**Command:**

```bash
python src/inference/infer_test.py
```

**Checkpoints:**

- ✅ Was a random image selected from test set?
- ✅ Are true labels shown?
- ✅ Is prediction correct?
- ✅ Can comparison with ground truth be made?

## 🌐 API Tests

### API Server Startup

**Terminal 1 - API Server:**

```bash
python src/api/app.py
```

Or:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test 1: Health Check

**Command:**

```bash
curl http://localhost:8000/health
```

**Expected Output:**

```json
{"status": "healthy", "model_loaded": true}
```

### Test 2: Root Endpoint

**Command:**

```bash
curl http://localhost:8000/
```

**Expected Output:**

```json
{
  "message": "CowinBMS API",
  "version": "1.0.0",
  "endpoints": {...}
}
```

### Test 3: Single Prediction API

**Command:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@examples/test_images/wagyu-ribeye.jpg" \
  -F "apply_segmentation=true" \
  -F "save_visualization=false"
```

**Python Test:**

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("examples/test_images/wagyu-ribeye.jpg", "rb")}
data = {
    "apply_segmentation": True,
    "save_visualization": False
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Checkpoints:**

- ✅ Is HTTP status code 200?
- ✅ Is JSON response properly formatted?
- ✅ Are prediction results correct?
- ✅ Does error handling work? (send invalid file)

**Error Tests:**

```bash
# Invalid file type
curl -X POST "http://localhost:8000/predict" -F "file=@README.md"

# Too large file (10MB+)
# Test with a large file
```

### Test 4: Batch Prediction API

**Command:**

```bash
curl -X POST "http://localhost:8000/batch" \
  -F "files=@examples/test_images/wagyu-ribeye.jpg" \
  -F "files=@examples/test_images/iStock-844693654_4_480x480.jpeg" \
  -F "apply_segmentation=true" \
  -F "export_format=csv"
```

**Python Test:**

```python
import requests

url = "http://localhost:8000/batch"
files = [
    ("files", open("examples/test_images/wagyu-ribeye.jpg", "rb")),
    ("files", open("examples/test_images/iStock-844693654_4_480x480.jpeg", "rb"))
]
data = {
    "apply_segmentation": True,
    "export_format": "csv"
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(f"Total: {result['summary']['total_files']}")
print(f"Successful: {result['summary']['successful']}")
print(f"Errors: {result['summary']['errors']}")
```

**Checkpoints:**

- ✅ Were all files processed?
- ✅ Was export file created? (`outputs/exports/`)
- ✅ Is summary correct?
- ✅ Are errors listed if any?

**Export Check:**

```bash
# Check export files
ls -lh outputs/exports/
cat outputs/exports/*.csv  # Show CSV content
```

### Test 5: API Documentation

**Open in Browser:**

```
http://localhost:8000/docs
```

**Checkpoints:**

- ✅ Does Swagger UI open?
- ✅ Are all endpoints visible?
- ✅ Can you test with "Try it out"?
- ✅ Are request/response examples correct?

### Test 6: Rate Limiting (if slowapi is installed)

**Send rapid requests:**

```bash
# Send 10+ requests (within 1 minute)
for i in {1..15}; do
  curl -X POST "http://localhost:8000/predict" \
    -F "file=@web_test_images/wagyu-ribeye.jpg"
  echo "Request $i"
done
```

**Expected:**

- First 10 requests successful
- 11th request onwards: 429 (Too Many Requests) error

## 📊 Log File Checks

### Prediction Logs

```bash
# Check prediction logs
cat logs/predictions/predictions.csv

# Last 10 predictions
tail -n 10 logs/predictions/predictions.csv

# JSON logs
ls -lh logs/predictions/*.json
```

### Application Logs

```bash
# Main log file
tail -f logs/app.log

# Error logs
grep ERROR logs/app.log
```

## 🔍 Detailed Checklist

### ✅ Inference Scripts

- [X] `infer_input.py` works for single image
- [X] `infer_input.py` works without segmentation
- [X] `infer_web_images.py` does batch processing
- [X] `infer_web_images.py` saves segmentation results
- [X] `infer_compare.py` creates comparison image
- [X] `infer_test.py` shows example from test set

### ✅ API Endpoints

- [X] Health check works
- [X] Single prediction endpoint works
- [X] Batch prediction endpoint works
- [X] Error handling works (invalid file)
- [X] File size validation works
- [X] Rate limiting works (if installed)
- [X] CORS works (cross-origin requests)

### ✅ Output Files

- [X] Images in `segmented_images/` folder
- [ ] Visualizations in `outputs/visualizations/` folder
- [X] Comparisons in `outputs/comparisons/` folder
- [X] Export files in `outputs/exports/` folder
- [X] Logs in `logs/predictions/` folder

### ✅ Validation

- [X] Segmentation quality check works
- [X] Image validation works
- [X] Confidence threshold warnings shown
- [X] Error messages are clear

### ✅ Performance

- [X] Model caching works (fast after first load)
- [X] Batch processing is efficient
- [X] API response time reasonable (<2 seconds)

## 🐛 Common Issues and Solutions

### Issue: Model file not found

```bash
# Solution: Check model path
ls src/models/efficientNet_v1.pth
# or check path in config file
cat configs/default.yaml
```

### Issue: CUDA out of memory

```bash
# Solution: Switch to CPU or reduce batch size
# Model automatically switches to CPU but check anyway
```

### Issue: Import errors

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: Rate limiting not working

```bash
# Solution: Install slowapi (optional)
pip install slowapi
# Or continue without rate limiting
```

### Issue: Poor segmentation quality

```bash
# Solution: Change segmentation method (in config)
# or adjust validation threshold
```

## 📝 Test Scenarios Summary

### Minimum Test (Quick)

```bash
# 1. Single image test
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg

# 2. API health check
curl http://localhost:8000/health

# 3. API single prediction
curl -X POST "http://localhost:8000/predict" -F "file=@examples/test_images/wagyu-ribeye.jpg"
```

### Comprehensive Test

```bash
# All inference scripts
python src/inference/infer_input.py --image web_test_images/wagyu-ribeye.jpg
python src/inference/infer_web_images.py
python src/inference/infer_compare.py --image1 web_test_images/wagyu-ribeye.jpg --image2 web_test_images/iStock-844693654_4_480x480.jpeg
python src/inference/infer_test.py

# API tests (with server running)
curl http://localhost:8000/health
curl http://localhost:8000/
curl -X POST "http://localhost:8000/predict" -F "file=@web_test_images/wagyu-ribeye.jpg"
curl -X POST "http://localhost:8000/batch" -F "files=@web_test_images/wagyu-ribeye.jpg" -F "export_format=csv"
```

## ✅ Success Criteria

- ✅ All scripts run without errors
- ✅ API works for all endpoints
- ✅ Output files created in correct locations
- ✅ Log files are consistent
- ✅ Error handling works
- ✅ Response formats are correct
