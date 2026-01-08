import numpy as np

BASE_MI_STATS = {
    "Select": (0.00, 0.03),
    "Choice": (0.01, 0.05),
    "Prime":  (0.02, 0.08),
    "Wagyu":  (0.05, 0.15),
    "Japanese A5": (0.08, 0.20),
}

CHOICE_THRESHOLDS = [
    (0.33, "Small"),
    (0.66, "Modest"),
    (1.00, "Moderate"),
]

PRIME_THRESHOLDS = [
    (0.33, "Slightly Abundant"),        # Prime-
    (0.67, "Moderately Abundant"),
]

def normalize_mi_by_base(mi, base_label):
    """
    Normalize MI into [0,1] within base category.
    """
    if base_label not in BASE_MI_STATS:
        return 0.0

    lo, hi = BASE_MI_STATS[base_label]
    if hi - lo <= 1e-6:
        return 0.0

    return float(np.clip((mi - lo) / (hi - lo), 0.0, 1.0))


def choose_marbling(dataset_label, mi_scaled):
    """
    Map base category + relative MI into marbling descriptor.
    """
    mi_rel = normalize_mi_by_base(mi_scaled, dataset_label)

    if dataset_label == "Select":
        return "Slight"

    if dataset_label == "Choice":
        for t, lab in CHOICE_THRESHOLDS:
            if mi_rel <= t:
                return lab

    if dataset_label == "Prime":
        for t, lab in PRIME_THRESHOLDS:
            if mi_rel <= t:
                return lab
        return "Moderately Abundant"

    if dataset_label in ["Wagyu", "Japanese A5"]:
        return "Very Abundant"

    return None


def marbling_to_usda(marbling, dataset_label, mi_scaled):
    """
    Convert marbling descriptor into USDA-like grade.
    """
    mi_rel = normalize_mi_by_base(mi_scaled, dataset_label)

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
        if dataset_label == "Prime":
            if mi_rel <= 0.33:
                return "Prime-"
            elif mi_rel <= 0.67:
                return "Prime"
            else:
                return "Prime+"
        else:
            return "Prime"

    if marbling == "Very Abundant":
        return "Beyond Prime"

    return None


__all__ = [
    "choose_marbling",
    "marbling_to_usda",
    "normalize_mi_by_base",
]
