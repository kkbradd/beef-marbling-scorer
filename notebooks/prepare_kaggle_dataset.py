import pandas as pd
import shutil
from pathlib import Path

SRC_CSV = "data/dataset_master.csv"
OUT_DIR = "kaggle_dataset"

df = pd.read_csv(SRC_CSV)

for split in ["train", "valid", "test"]:
    df_s = df[df["split"] == split]

    rows = []
    for _, r in df_s.iterrows():
        src = Path(r["processed_path"])
        dst = Path(OUT_DIR) / "images" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

        rows.append({
            "image_path": f"images/{dst.name}",
            "MI": r["MI"],
            "dataset_grade": r["dataset_grade"]
        })

    pd.DataFrame(rows).to_csv(
        Path(OUT_DIR) / f"{split}.csv",
        index=False
    )
