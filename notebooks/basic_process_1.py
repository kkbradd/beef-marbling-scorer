import cv2
import os
import pandas as pd

df = pd.read_csv("data/dataset_master.csv")

OUT_DIR = "data/processed/images"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 512

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return img

processed_paths = []

for _, row in df.iterrows():
    img = preprocess_image(row["path"])
    if img is None:
        processed_paths.append(None)
        continue

    out_path = os.path.join(OUT_DIR, row["image_id"])
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    processed_paths.append(out_path)

df["processed_path"] = processed_paths
df.to_csv("data/dataset_master.csv", index=False)

print("Preprocessing completed.")
