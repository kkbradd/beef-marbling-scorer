import os
import sys
import json
import random
import cv2
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.cowin_model import CowinBMSModel
from src.features.transforms import valid_tfms
from src.features.rules import choose_marbling, marbling_to_usda, assign_bms, assign_aus_meat
from src.utils.helpers import to_builtin

CLASS_NAMES = ["Select", "Choice", "Prime", "Wagyu", "Japanese A5"]


def resolve_image_path(data_dir: str, images_dir: str, rel_path: str) -> str:
    rel_path = str(rel_path).lstrip("/")

    cands = [
        os.path.join(data_dir, rel_path),
        os.path.join(images_dir, rel_path),
        os.path.join(images_dir, os.path.basename(rel_path)),
    ]

    for p in cands:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"Image not found. Tried: {cands}")


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def main():
    data_dir = os.path.join(PROJECT_ROOT, "kaggle_dataset")
    images_dir = os.path.join(data_dir, "images")
    test_csv = os.path.join(data_dir, "test.csv")
    model_path = os.path.join(PROJECT_ROOT, "src/models/efficientNet_v1.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowinBMSModel().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    df = pd.read_csv(test_csv)

    row = df.sample(1).iloc[0]
    img_path = resolve_image_path(data_dir, images_dir, row["image_path"])

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    true_dataset = row.get("dataset_grade", None)
    if true_dataset is not None:
        try:
            true_dataset = CLASS_NAMES[int(true_dataset)]
        except (ValueError, TypeError, IndexError) as e:
            true_dataset = str(true_dataset)

    true_mi = row.get("MI", None)
    true_mi = float(true_mi) if true_mi is not None else None

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
            "path": img_path
        },
        "true": {
            "dataset_grade": to_builtin(true_dataset) if true_dataset is not None else None,
            "mi": round(to_builtin(true_mi), 4) if true_mi is not None else None
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

    print(json.dumps(result, indent=2, default=to_builtin))


if __name__ == "__main__":
    main()
 