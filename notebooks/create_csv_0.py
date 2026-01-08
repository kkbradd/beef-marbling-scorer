import os
from re import I
import pandas as pd

ROOT = "data/raw"
records = []

label_map = {
    "0": "Select",
    "1": "Choice",
    "2": "Prime",
    "3": "Wagyu",
    "4": "Japanese A5"
}

for split in ["train", "valid", "test"]:
    split_path = os.path.join(ROOT, split)
    for cls in ["0", "1", "2", "3", "4"]:
        cls_path = os.path.join(split_path, cls)
        if not os.path.exists(cls_path):
            continue
        for img in os.listdir(cls_path):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                records.append({
                    "image_id": img,
                    "path": os.path.join(cls_path, img),
                    "split": split,                    
                    "dataset_grade": int(cls),
                    "dataset_label": label_map[cls],
                    "source": "beefsenseML",
                    "MI": None,
                    "marbling_grade": None,
                    "usda_grade": None,
                    "BMS": None,
                    "high_tier": None
                })

df = pd.DataFrame(records)
df.to_csv("data/dataset_master.csv", index = False)
print("dataset master is created:", len(df))



