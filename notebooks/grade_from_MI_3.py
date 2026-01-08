import numpy as np
import pandas as pd

CSV_PATH = "data/dataset_master.csv"

def robust_z(x):
    x = x.astype(float)
    med = np.nanmedian(x)
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
    return (x - med) / iqr

def scale_01(x, q_low=0.01, q_high=0.99):
    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    z = (x - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)

CHOICE_THRESHOLDS = [
    (0.33, "Small"),
    (0.66, "Modest"),
    (1.00, "Moderate"),
]

PRIME_THRESHOLDS = [
    (0.50, "Slightly Abundant"),
    (1.00, "Moderately Abundant"),
]

def choose_marbling(dataset_label, mi_scaled):
    if dataset_label == "Select":
        return "Slight"

    if dataset_label == "Choice":
        for t, lab in CHOICE_THRESHOLDS:
            if mi_scaled <= t:
                return lab

    if dataset_label == "Prime":
        for t, lab in PRIME_THRESHOLDS:
            if mi_scaled <= t:
                return lab

    if dataset_label in ["Wagyu", "Japanese A5"]:
        return "Very Abundant"

    return None

def marbling_to_usda(marbling, dataset_label, mi_scaled):
    if marbling == "Slight":
        return "Select"

    if marbling == "Small":
        return "Choice-"

    if marbling == "Modest":
        return "Choice"

    if marbling == "Moderate":
        return "Choice+"

    if marbling == "Slightly Abundant":
        return "Prime-"

    if marbling == "Moderately Abundant":
        if mi_scaled <= 0.75:
            return "Prime"
        else:
            return "Prime+"

    if marbling == "Very Abundant":
        return "Beyond Prime"

    return None

def main():
    df = pd.read_csv(CSV_PATH)

    required = {"fat_ratio", "fineness", "dataset_label"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    df["fat_ratio_log"] = np.log1p(df["fat_ratio"].fillna(0))
    df["fineness_log"] = np.log1p(df["fineness"].fillna(0))

    df["fat_ratio_n"] = robust_z(df["fat_ratio_log"].values)
    df["fineness_n"] = robust_z(df["fineness_log"].values)

    df["mi_score"] = 0.6 * df["fat_ratio_n"] + 0.4 * df["fineness_n"]
    df["mi_scaled"] = scale_01(df["mi_score"].values)

    df["marbling_grade"] = df.apply(
        lambda r: choose_marbling(r["dataset_label"], r["mi_scaled"]),
        axis=1
    )

    df["usda_grade"] = df.apply(
        lambda r: marbling_to_usda(
            r["marbling_grade"],
            r["dataset_label"],
            r["mi_scaled"]
        ),
        axis=1
    )

    df.to_csv(CSV_PATH, index=False)

    print("\n=== Marbling distribution ===")
    print(pd.crosstab(df["dataset_label"], df["marbling_grade"], normalize="index").round(3))

    print("\n=== USDA distribution ===")
    print(pd.crosstab(df["dataset_label"], df["usda_grade"], normalize="index").round(3))

    print("\n=== MI summary ===")
    print(df.groupby("dataset_label")["mi_scaled"].describe())

    print("\nDone.")

if __name__ == "__main__":
    main()
