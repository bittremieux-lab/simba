ION_ACTIVATION = ["HCD", "CID"]
IONIZATION_METHODS = ["NSI", "ESI", "APCI"]


def encode_ion_activation(ion_activation: str):
    """Encode ion activation method as one-hot vector.

    Args:
        ion_activation: Ion activation method (e.g., 'HCD')

    Returns:
        list: One-hot encoded vector
    """
    return [1 if ion_activation == ia else 0 for ia in ION_ACTIVATION]


def encode_ionization_method(ionization_method: str):
    """Encode ionization method as one-hot vector.

    Args:
        ionization_method: Ionization method (e.g., 'ESI')

    Returns:
        list: One-hot encoded vector
    """
    return [1 if ionization_method == im else 0 for im in IONIZATION_METHODS]
