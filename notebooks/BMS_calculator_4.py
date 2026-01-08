import numpy as np
import pandas as pd

CSV_PATH = "data/dataset_master.csv"

def scale_local(x):
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin == 0:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def assign_bms(df):
    df["bms_score"] = None

    mask_sc = df["dataset_label"].isin(["Select", "Choice"])
    df.loc[mask_sc, "bms_score"] = 1

    mask_prime = df["dataset_label"] == "Prime"
    df.loc[mask_prime, "bms_score"] = 2

    mask_wagyu = df["dataset_label"] == "Wagyu"
    if mask_wagyu.any():
        local_mi = scale_local(df.loc[mask_wagyu, "mi_scaled"].values)
        bms_vals = 3 + np.floor(local_mi * 5).astype(int)
        bms_vals = np.clip(bms_vals, 3, 7)
        df.loc[mask_wagyu, "bms_score"] = bms_vals

    mask_a5 = df["dataset_label"] == "Japanese A5"
    if mask_a5.any():
        local_mi = scale_local(df.loc[mask_a5, "mi_scaled"].values)
        bms_vals = 8 + np.floor(local_mi * 5).astype(int)
        bms_vals = np.clip(bms_vals, 8, 12)
        df.loc[mask_a5, "bms_score"] = bms_vals

    df["bms_score"] = df["bms_score"].astype(int)
    return df

def main():
    df = pd.read_csv(CSV_PATH)

    required = {"dataset_label", "mi_scaled"}
    if not required.issubset(df.columns):
        raise ValueError("Required columns missing")

    df = assign_bms(df)
    df.to_csv(CSV_PATH, index=False)

    print("\n=== BMS distribution by dataset_label ===")
    print(pd.crosstab(df["dataset_label"], df["bms_score"], normalize="index").round(3))

    print("\n=== BMS summary ===")
    print(df.groupby("dataset_label")["bms_score"].describe())

    print("\nDone.")

if __name__ == "__main__":
    main()
