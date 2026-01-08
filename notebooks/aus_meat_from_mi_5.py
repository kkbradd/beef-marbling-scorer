import numpy as np
import pandas as pd

CSV_PATH = "data/dataset_master.csv"

def scale_local(x):
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin == 0:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def assign_aus_meat(df):
    df["aus_meat_score"] = None

    mask_select = df["dataset_label"] == "Select"
    if mask_select.any():
        local = scale_local(df.loc[mask_select, "mi_scaled"].values)
        df.loc[mask_select, "aus_meat_score"] = np.where(local < 0.5, 0, 1)

    mask_choice = df["dataset_label"] == "Choice"
    if mask_choice.any():
        local = scale_local(df.loc[mask_choice, "mi_scaled"].values)
        df.loc[mask_choice, "aus_meat_score"] = np.where(local < 0.5, 2, 3)

    mask_prime = df["dataset_label"] == "Prime"
    if mask_prime.any():
        local = scale_local(df.loc[mask_prime, "mi_scaled"].values)
        df.loc[mask_prime, "aus_meat_score"] = np.where(local < 0.5, 4, 5)

    mask_wagyu = df["dataset_label"] == "Wagyu"
    if mask_wagyu.any():
        local = scale_local(df.loc[mask_wagyu, "mi_scaled"].values)
        df.loc[mask_wagyu, "aus_meat_score"] = np.where(local < 0.5, 6, 7)

    mask_a5 = df["dataset_label"] == "Japanese A5"
    if mask_a5.any():
        local = scale_local(df.loc[mask_a5, "mi_scaled"].values)
        df.loc[mask_a5, "aus_meat_score"] = np.where(local < 0.5, 8, 9)

    df["aus_meat_score"] = df["aus_meat_score"].astype(int)
    return df

def main():
    df = pd.read_csv(CSV_PATH)

    required = {"dataset_label", "mi_scaled"}
    if not required.issubset(df.columns):
        raise ValueError("Required columns missing")

    df = assign_aus_meat(df)
    df.to_csv(CSV_PATH, index=False)

    print("\n=== AUS-MEAT distribution ===")
    print(pd.crosstab(df["dataset_label"], df["aus_meat_score"], normalize="index").round(3))

    print("\nDone.")

if __name__ == "__main__":
    main()
