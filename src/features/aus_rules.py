import numpy as np

from .bms_rules import scale_local


def mi_to_aus(dataset_label, mi):
    """
    Legacy, simple AUS-MEAT mapping from dataset label + MI.
    Kept for backwards compatibility; prefer `assign_aus_meat` for
    DataFrame-based workflows.
    """
    if dataset_label == "Select":
        return 0 if mi < 0.5 else 1
    if dataset_label == "Choice":
        return 2 if mi < 0.5 else 3
    if dataset_label == "Prime":
        return 4 if mi < 0.5 else 5
    if dataset_label == "Wagyu":
        return 6 if mi < 0.5 else 7
    if dataset_label == "Japanese A5":
        return 8 if mi < 0.5 else 9


def assign_aus_meat(df):
    """
    Assign AUS-MEAT scores using local MI scaling within each dataset label.
    For single samples, uses direct mapping instead of local scaling.

    Expects a DataFrame with:
      - 'dataset_label'
      - 'mi_scaled'

    Returns the same DataFrame with an integer 'aus_meat_score' column.
    """
    df = df.copy()
    df["aus_meat_score"] = None
    is_single = len(df) == 1

    mask_select = df["dataset_label"] == "Select"
    if mask_select.any():
        if is_single:
            mi_val = df.loc[mask_select, "mi_scaled"].iloc[0]
            df.loc[mask_select, "aus_meat_score"] = 0 if mi_val < 0.5 else 1
        else:
            local = scale_local(df.loc[mask_select, "mi_scaled"].values)
            df.loc[mask_select, "aus_meat_score"] = np.where(local < 0.5, 0, 1)

    mask_choice = df["dataset_label"] == "Choice"
    if mask_choice.any():
        if is_single:
            mi_val = df.loc[mask_choice, "mi_scaled"].iloc[0]
            df.loc[mask_choice, "aus_meat_score"] = 2 if mi_val < 0.5 else 3
        else:
            local = scale_local(df.loc[mask_choice, "mi_scaled"].values)
            df.loc[mask_choice, "aus_meat_score"] = np.where(local < 0.5, 2, 3)

    mask_prime = df["dataset_label"] == "Prime"
    if mask_prime.any():
        if is_single:
            mi_val = df.loc[mask_prime, "mi_scaled"].iloc[0]
            df.loc[mask_prime, "aus_meat_score"] = 4 if mi_val < 0.5 else 5
        else:
            local = scale_local(df.loc[mask_prime, "mi_scaled"].values)
            df.loc[mask_prime, "aus_meat_score"] = np.where(local < 0.5, 4, 5)

    mask_wagyu = df["dataset_label"] == "Wagyu"
    if mask_wagyu.any():
        if is_single:
            mi_val = df.loc[mask_wagyu, "mi_scaled"].iloc[0]
            df.loc[mask_wagyu, "aus_meat_score"] = 6 if mi_val < 0.5 else 7
        else:
            local = scale_local(df.loc[mask_wagyu, "mi_scaled"].values)
            df.loc[mask_wagyu, "aus_meat_score"] = np.where(local < 0.5, 6, 7)

    mask_a5 = df["dataset_label"] == "Japanese A5"
    if mask_a5.any():
        if is_single:
            mi_val = df.loc[mask_a5, "mi_scaled"].iloc[0]
            df.loc[mask_a5, "aus_meat_score"] = 8 if mi_val < 0.5 else 9
        else:
            local = scale_local(df.loc[mask_a5, "mi_scaled"].values)
            df.loc[mask_a5, "aus_meat_score"] = np.where(local < 0.5, 8, 9)

    df["aus_meat_score"] = df["aus_meat_score"].astype(int)
    return df


__all__ = ["mi_to_aus", "assign_aus_meat"]
