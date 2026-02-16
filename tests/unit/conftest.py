"""Pytest configuration and fixtures for unit tests."""

import numpy as np
import pytest

from simba.core.data.spectrum import SpectrumExt


@pytest.fixture
def create_test_spectrum():
    """Factory fixture to create test spectrum instances with default values."""

    def _create(**kwargs):
        """Helper function to create a test spectrum with default values."""
        smiles = kwargs.get("smiles", "CCO")
        defaults = {
            "identifier": "test_spectrum",
            "precursor_mz": 100.0,
            "precursor_charge": 1,
            "mz": np.array([100.0, 200.0]),
            "intensity": np.array([0.5, 1.0]),
            "retention_time": 1.5,
            "params": {"smiles": smiles},
            "library": "test",
            "inchi": "test_inchi",
            "smiles": smiles,
            "ionmode": "positive",
            "adduct": "[M+H]+",
            "ce": 20.0,
            "ion_activation": "CID",
            "ionization_method": "ESI",
            "bms": "test_scaffold",
            "superclass": "test",
            "classe": "test",
            "subclass": "test",
            "inchi_key": None,
            "spectrum_hash": None,
        }
        defaults.update(kwargs)
        if "smiles" in kwargs:
            defaults["params"]["smiles"] = kwargs["smiles"]
        return SpectrumExt(**defaults)

    return _create
