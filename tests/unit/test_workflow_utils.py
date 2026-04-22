"""Tests for simba/workflows/utils.py"""

from simba.workflows.utils import filter_invalid_smiles


class TestFilterInvalidSmiles:
    def test_all_valid_kept(self, create_test_spectrum):
        spectra = [
            create_test_spectrum(smiles="CCO"),
            create_test_spectrum(smiles="CCCO"),
            create_test_spectrum(smiles="c1ccccc1"),
        ]
        result = filter_invalid_smiles(spectra)
        assert len(result) == 3

    def test_invalid_smiles_removed(self, create_test_spectrum):
        spectra = [
            create_test_spectrum(smiles="CCO"),
            create_test_spectrum(smiles="not_a_smiles!!!"),
            create_test_spectrum(smiles="CCCO"),
        ]
        result = filter_invalid_smiles(spectra)
        assert len(result) == 2
        assert all(s.smiles in ("CCO", "CCCO") for s in result)

    def test_br_tag_smiles_removed(self, create_test_spectrum):
        """SMILES with embedded <br> HTML tags (seen in Janne's data) must be removed."""
        broken = "CC(C)CC1CC(=<br>C2)CC"
        spectra = [
            create_test_spectrum(smiles="CCO"),
            create_test_spectrum(smiles=broken),
        ]
        result = filter_invalid_smiles(spectra)
        assert len(result) == 1
        assert result[0].smiles == "CCO"

    def test_empty_list(self):
        result = filter_invalid_smiles([])
        assert result == []

    def test_all_invalid_returns_empty(self, create_test_spectrum):
        spectra = [
            create_test_spectrum(smiles="!!!"),
            create_test_spectrum(smiles="<br>"),
        ]
        result = filter_invalid_smiles(spectra)
        assert result == []

    def test_logging_on_invalid(self, create_test_spectrum, caplog):
        import logging

        spectra = [
            create_test_spectrum(smiles="CCO"),
            create_test_spectrum(smiles="bad!!smiles"),
        ]
        with caplog.at_level(logging.WARNING):
            filter_invalid_smiles(spectra)
        assert any("unparseable SMILES" in r.message for r in caplog.records)

    def test_logging_all_valid(self, create_test_spectrum, caplog):

        spectra = [create_test_spectrum(smiles="CCO")]
        result = filter_invalid_smiles(spectra)
        assert len(result) == 1
