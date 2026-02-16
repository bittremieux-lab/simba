"""Unit tests for SpectrumExt class."""

import numpy as np
import pytest


pytestmark = pytest.mark.unit


class TestSpectrumExtSetParams:
    """Test set_params method."""

    def test_set_params(self, create_test_spectrum):
        """Test setting params dictionary."""
        spectrum = create_test_spectrum()

        params = {"key1": "value1", "key2": 123}
        spectrum.set_params(params)

        assert spectrum.params == params


class TestSpectrumExtSetSpectrumVector:
    """Test set_spectrum_vector method."""

    def test_set_spectrum_vector(self, create_test_spectrum):
        """Test setting spectrum vector."""
        spectrum = create_test_spectrum()

        vector = np.array([1.0, 2.0, 3.0])
        spectrum.set_spectrum_vector(vector)

        assert np.array_equal(spectrum.spectrum_vector, vector)


class TestSpectrumExtSetMurckoScaffold:
    """Test set_murcko_scaffold method."""

    def test_set_murcko_scaffold(self, create_test_spectrum):
        """Test setting Murcko scaffold."""
        spectrum = create_test_spectrum(bms="test")

        scaffold = "c1ccccc1"
        spectrum.set_murcko_scaffold(scaffold)

        assert spectrum.murcko_scaffold == scaffold


class TestSpectrumExtSetSmiles:
    """Test set_smiles method."""

    def test_set_smiles(self, create_test_spectrum):
        """Test setting SMILES string."""
        spectrum = create_test_spectrum()

        smiles = "CCCO"
        spectrum.set_smiles(smiles)

        assert spectrum.smiles == smiles


class TestSpectrumExtSetMaxPeak:
    """Test set_max_peak method."""

    def test_set_max_peak(self, create_test_spectrum):
        """Test setting maximum peak amplitude."""
        spectrum = create_test_spectrum()

        max_peak = 1000.0
        spectrum.set_max_peak(max_peak)

        assert spectrum.max_peak == max_peak


class TestSpectrumExtSerialization:
    """Test __getstate__ and __setstate__ methods."""

    def test_getstate_setstate_roundtrip(self, create_test_spectrum):
        """Test serialization and deserialization roundtrip."""
        spectrum = create_test_spectrum(
            library="test_library",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            superclass="Organic compounds",
            classe="Alcohols",
            subclass="Primary alcohols",
            bms="CCO",
            inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            spectrum_hash="test_hash_123",
        )

        state = spectrum.__getstate__()

        new_spectrum = create_test_spectrum()
        new_spectrum.__setstate__(state)

        assert new_spectrum.smiles == "CCO"
        assert new_spectrum.library == "test_library"
        assert new_spectrum.inchi == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        assert new_spectrum.ionmode == "positive"
        assert new_spectrum.adduct == "[M+H]+"
        assert new_spectrum.ce == 20.0
        assert new_spectrum.ion_activation == "CID"
        assert new_spectrum.ionization_method == "ESI"
        assert new_spectrum.retention_time == 1.5
        assert new_spectrum.superclass == "Organic compounds"
        assert new_spectrum.classe == "Alcohols"
        assert new_spectrum.subclass == "Primary alcohols"
        assert new_spectrum.murcko_scaffold == "CCO"
        assert new_spectrum.inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        assert new_spectrum.spectrum_hash == "test_hash_123"

    def test_setstate_missing_ce(self, create_test_spectrum):
        """Test __setstate__ with missing 'ce' field."""
        spectrum = create_test_spectrum()

        state = spectrum.__getstate__()
        del state["ce"]

        spectrum.__setstate__(state)
        assert spectrum.ce == 0.0

    def test_setstate_missing_ion_activation(self, create_test_spectrum):
        """Test __setstate__ with missing 'ion_activation' field."""
        spectrum = create_test_spectrum()

        state = {
            "identifier": "test_spectrum",
            "precursor_mz": 100.0,
            "precursor_charge": 1,
            "mz": np.array([100.0, 200.0]),
            "intensity": np.array([0.5, 1.0]),
            "annotation": None,
            "proforma": None,
            "params": {},
            "intensity_array": None,
            "mz_array": None,
            "spectrum_vector": "",
            "smiles": "CCO",
            "max_peak": "",
            "library": "test",
            "inchi": "test_inchi",
            "ionmode": "positive",
            "adduct": "[M+H]+",
            "ce": 20.0,
            "retention_time": 1.5,
            "superclass": "test",
            "classe": "test",
            "subclass": "test",
            "murcko_scaffold": "test",
        }

        spectrum.__setstate__(state)
        assert spectrum.ion_activation == ""

    def test_setstate_missing_ionization_method(self, create_test_spectrum):
        """Test __setstate__ with missing 'ionization_method' field."""
        spectrum = create_test_spectrum()

        state = {
            "identifier": "test_spectrum",
            "precursor_mz": 100.0,
            "precursor_charge": 1,
            "mz": np.array([100.0, 200.0]),
            "intensity": np.array([0.5, 1.0]),
            "annotation": None,
            "proforma": None,
            "smiles": "CCO",
            "library": "test",
            "inchi": "test_inchi",
            "ionmode": "positive",
            "adduct": "[M+H]+",
            "ce": 20.0,
            "ion_activation": "CID",
            "retention_time": 1.5,
            "superclass": "test",
            "classe": "test",
            "subclass": "test",
            "murcko_scaffold": "test",
            "params": {},
            "intensity_array": None,
            "mz_array": None,
            "spectrum_vector": "",
            "max_peak": "",
        }

        spectrum.__setstate__(state)
        assert spectrum.ionization_method == ""

    def test_setstate_missing_inchi_key(self, create_test_spectrum):
        """Test __setstate__ with missing 'inchi_key' field."""
        spectrum = create_test_spectrum()

        state = {
            "identifier": "test_spectrum",
            "precursor_mz": 100.0,
            "precursor_charge": 1,
            "mz": np.array([100.0, 200.0]),
            "intensity": np.array([0.5, 1.0]),
            "annotation": None,
            "proforma": None,
            "smiles": "CCO",
            "library": "test",
            "inchi": "test_inchi",
            "ionmode": "positive",
            "adduct": "[M+H]+",
            "ce": 20.0,
            "ion_activation": "CID",
            "ionization_method": "ESI",
            "retention_time": 1.5,
            "superclass": "test",
            "classe": "test",
            "subclass": "test",
            "murcko_scaffold": "test",
            "params": {},
            "intensity_array": None,
            "mz_array": None,
            "spectrum_vector": "",
            "max_peak": "",
        }

        spectrum.__setstate__(state)
        assert spectrum.inchi_key == ""

    def test_setstate_missing_spectrum_hash(self, create_test_spectrum):
        """Test __setstate__ with missing 'spectrum_hash' field."""
        spectrum = create_test_spectrum()

        state = {
            "identifier": "test_spectrum",
            "precursor_mz": 100.0,
            "precursor_charge": 1,
            "mz": np.array([100.0, 200.0]),
            "intensity": np.array([0.5, 1.0]),
            "annotation": None,
            "proforma": None,
            "smiles": "CCO",
            "library": "test",
            "inchi": "test_inchi",
            "ionmode": "positive",
            "adduct": "[M+H]+",
            "ce": 20.0,
            "ion_activation": "CID",
            "ionization_method": "ESI",
            "retention_time": 1.5,
            "superclass": "test",
            "classe": "test",
            "subclass": "test",
            "murcko_scaffold": "test",
            "inchi_key": "test_key",
            "params": {},
            "intensity_array": None,
            "mz_array": None,
            "spectrum_vector": "",
            "max_peak": "",
        }

        spectrum.__setstate__(state)
        assert spectrum.spectrum_hash is None
