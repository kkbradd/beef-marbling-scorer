import numpy as np


def to_builtin(x):
    """
    Convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

