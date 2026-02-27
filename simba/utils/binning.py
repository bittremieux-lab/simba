import numpy as np


def float_to_ordinal_class(array: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert continuous float values [0, 1] to discrete ordinal classes [0, n_classes-1]."""
    return round_to_ordinal(array * (n_classes - 1)).astype(int)


def round_to_ordinal(array: np.ndarray) -> np.ndarray:
    """Round values to nearest integer with custom rounding (1.5 → 2)."""
    return np.floor(array + 0.51)
