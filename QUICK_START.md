# Quick Start - Testing Guide

## 🚀 Quick Test (5 Minutes)

### 1️⃣ Single Image Test

```bash
python src/inference/infer_input.py --image examples/test_images/wagyu-ribeye.jpg
```

**Check:** Is JSON output coming? ✅

---

### 2️⃣ Batch Test

```bash
python src/inference/infer_web_images.py
```

**Check:**

- Were all images processed? ✅
- Are images in `segmented_images/` folder? ✅
- Check example results in `examples/results/` folder? ✅

---

### 3️⃣ API Test (3 Steps)

**A) Start API** (Terminal 1):

```bash
python src/api/app.py
```

**B) Test in Another Terminal** (Terminal 2):

```bash
# Health check
curl http://localhost:8000/health

# Prediction test
curl -X POST "http://localhost:8000/predict" \
  -F "file=@examples/test_images/wagyu-ribeye.jpg"
```

**C) Open Documentation in Browser:**

```
http://localhost:8000/docs
```

---

## 📋 Detailed Tests

See `TESTING.md` file for full testing guide.

## ✅ Success Criteria

- ✅ Scripts run without errors
- ✅ JSON output properly formatted
- ✅ API returns 200 status code
- ✅ Output files created
- ✅ Log files written
