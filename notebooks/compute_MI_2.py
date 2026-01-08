import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
CSV_PATH = "data/dataset_master.csv"
IMAGE_PATH_COLUMN = "processed_path"   # burada raw yerine processed kullanıyoruz
LABEL_COLUMN = "dataset_label"

# ===============================
# FAT + FINENESS FUNCTIONS
# ===============================

def compute_fineness(fat_mask):
    """
    Marbling inceliği:
    Çok sayıda küçük yağ komponenti -> yüksek fineness
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fat_mask, connectivity=8
    )

    # Background (0) hariç
    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return 0.0

    mean_area = np.mean(areas)
    component_count = len(areas)

    fineness = component_count / (mean_area + 1e-6)
    return fineness


def compute_mi_features(img_path):
    """
    Returns:
        fat_ratio, fineness
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Fat mask (white marbling) ---
    _, fat_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # --- Muscle mask (ignore dark background) ---
    _, muscle_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    fat_pixels = np.sum(fat_mask == 255)
    muscle_pixels = np.sum(muscle_mask == 255)

    if muscle_pixels == 0:
        return None, None

    fat_ratio = fat_pixels / muscle_pixels
    fineness = compute_fineness(fat_mask)

    return fat_ratio, fineness


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    print("Loading dataset_master.csv...")
    df = pd.read_csv(CSV_PATH)

    fat_ratios = []
    fineness_scores = []

    print("Computing MI features (fat_ratio + fineness)...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row[IMAGE_PATH_COLUMN]
        fr, fin = compute_mi_features(img_path)

        fat_ratios.append(fr)
        fineness_scores.append(fin)

    df["fat_ratio"] = fat_ratios
    df["fineness"] = fineness_scores

    df.to_csv(CSV_PATH, index=False)
    print("MI features saved to dataset_master.csv")

    # ===============================
    # SANITY CHECKS
    # ===============================
    print("\n=== FAT RATIO BY CLASS ===")
    print(df.groupby(LABEL_COLUMN)["fat_ratio"].describe())

    print("\n=== FINENESS BY CLASS ===")
    print(df.groupby(LABEL_COLUMN)["fineness"].describe())

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
