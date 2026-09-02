"""Shared pytest configuration and fixtures for SIMBA tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def hydra_config():
    """Create a Hydra DictConfig for testing."""
    from hydra import compose, initialize_config_dir

    from simba.utils.config_utils import get_config_path

    config_path = get_config_path()

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="config")
        return cfg


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_mgf(fixtures_dir):
    """Path to sample MGF file (standard format)."""
    return str(fixtures_dir / "sample_spectra.mgf")


@pytest.fixture
def sample_mgf_casmi(fixtures_dir):
    """Path to sample MGF file (CASMI2022 format with SMILES)."""
    return str(fixtures_dir / "sample_spectra_casmi.mgf")


@pytest.fixture
def sample_training_spectra(tmp_path):
    """Create a small MGF file with SMILES for training tests."""
    mgf_path = tmp_path / "spectra.mgf"

    spectra_data = []
    for i in range(14):
        if i % 2 == 0:
            smiles = "CCO"
            formula = "C2H6O"
            inchikey = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
            mz_values = [45.034, 46.041, 47.049, 48.052, 49.055, 50.058]
            intensities = [100.0, 50.0, 25.0, 20.0, 15.0, 10.0]
        else:
            smiles = "CC(C)O"
            formula = "C3H8O"
            inchikey = "KFZMGEQAYNKOFK-UHFFFAOYSA-N"
            mz_values = [43.018, 45.034, 60.058, 61.062, 62.065, 63.068]
            intensities = [80.0, 60.0, 100.0, 40.0, 30.0, 20.0]

        spectrum = f"""BEGIN IONS
PEPMASS=100.0
RT=100.0
IONMODE=Positive
ADDUCT=[M+H]+
CHARGE=1
ID={i}
SMILES={smiles}
FORMULA={formula}
INCHIKEY={inchikey}
SCAN NUMBER={i}
NUM_PEAKS={len(mz_values)}
{mz_values[0]} {intensities[0]}
{mz_values[1]} {intensities[1]}
{mz_values[2]} {intensities[2]}
{mz_values[3]} {intensities[3]}
{mz_values[4]} {intensities[4]}
{mz_values[5]} {intensities[5]}
END IONS

"""
        spectra_data.append(spectrum)

    mgf_path.write_text("".join(spectra_data))
    return mgf_path


@pytest.fixture
def mock_model(mocker, hydra_config):
    """Create a mocked SIMBA model with random weights for testing."""
    from simba.core.models.similarity_models import SimilarityModelMultitask

    cfg = hydra_config

    model = SimilarityModelMultitask(
        d_model=int(cfg.model.transformer.d_model),
        n_layers=int(cfg.model.transformer.n_layers),
        use_element_wise=True,
        use_cosine_distance=cfg.model.tasks.cosine_similarity.use_cosine_distance,
    )
    model.eval()

    mocker.patch(
        "simba.core.models.similarity_models.SimilarityModelMultitask.load_from_checkpoint",
        return_value=model,
    )

    return model
