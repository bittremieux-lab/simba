import numpy as np


def round_to_ordinal(array: np.ndarray) -> np.ndarray:
    """Round values to nearest integer with custom rounding (1.5 → 2)."""
    return np.floor(array + 0.51)
