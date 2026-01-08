import numpy as np


def scale_local(x: np.ndarray) -> np.ndarray:
    """
    Scale an array to [0, 1] using its own min / max.
    NaNs are ignored in the min / max computation.
    """
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin == 0:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def assign_bms(df):
    """
    Assign JMGA BMS scores based on dataset label and local MI scaling.
    For single samples, uses direct mapping instead of local scaling.

    Expects a DataFrame with:
      - 'dataset_label'  (Select / Choice / Prime / Wagyu / Japanese A5)
      - 'mi_scaled'      (float in [0, 1])

    Returns the same DataFrame with an integer 'bms_score' column.
    """
    df = df.copy()
    df["bms_score"] = None
    is_single = len(df) == 1

    mask_sc = df["dataset_label"].isin(["Select", "Choice"])
    df.loc[mask_sc, "bms_score"] = 1

    mask_prime = df["dataset_label"] == "Prime"
    df.loc[mask_prime, "bms_score"] = 2

    mask_wagyu = df["dataset_label"] == "Wagyu"
    if mask_wagyu.any():
        if is_single:
            mi_val = df.loc[mask_wagyu, "mi_scaled"].iloc[0]
            bms_val = 3 + int(np.floor(mi_val * 5))
            bms_val = np.clip(bms_val, 3, 7)
            df.loc[mask_wagyu, "bms_score"] = bms_val
        else:
            local_mi = scale_local(df.loc[mask_wagyu, "mi_scaled"].values)
            bms_vals = 3 + np.floor(local_mi * 5).astype(int)
            bms_vals = np.clip(bms_vals, 3, 7)
            df.loc[mask_wagyu, "bms_score"] = bms_vals

    mask_a5 = df["dataset_label"] == "Japanese A5"
    if mask_a5.any():
        if is_single:
            mi_val = df.loc[mask_a5, "mi_scaled"].iloc[0]
            bms_val = 8 + int(np.floor(mi_val * 5))
            bms_val = np.clip(bms_val, 8, 12)
            df.loc[mask_a5, "bms_score"] = bms_val
        else:
            local_mi = scale_local(df.loc[mask_a5, "mi_scaled"].values)
            bms_vals = 8 + np.floor(local_mi * 5).astype(int)
            bms_vals = np.clip(bms_vals, 8, 12)
            df.loc[mask_a5, "bms_score"] = bms_vals

    df["bms_score"] = df["bms_score"].astype(int)
    return df


__all__ = ["scale_local", "assign_bms"]
