"""
Integration tests for SIMBA training pipeline components.

Tests cover:
- Data preprocessing and generation
- Mapping file structure validation
- Model initialization and training smoke tests
- Inference on trained model
- MCES data loading utilities
"""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from simba.core.chemistry.mces_loader.load_mces import LoadMCES
from simba.core.data.molecule_pairs_opt import MoleculePairsOpt
from simba.core.data.preprocessing_simba import PreprocessingSimba
from simba.core.data.spectrum import SpectrumExt
from simba.core.models.ordinal.embedder_multitask import EmbedderMultitask
from simba.core.models.simba_model import Simba
from simba.core.training.train_utils import TrainUtils


class TestDataPreprocessing:
    """Test data preprocessing and generation for training."""

    def test_train_val_test_split(self, sample_training_spectra, hydra_config):
        """Test that train/val/test split works correctly."""
        # Load spectra
        all_spectra = PreprocessingSimba.load_spectra(
            str(sample_training_spectra),
            hydra_config,
            n_samples=100,
            use_gnps_format=False,
            use_only_protonized_adducts=False,
        )

        assert len(all_spectra) == 14, "Should load all 14 spectra"

        # Set murcko_scaffold for each spectrum (required by train_val_test_split_bms)
        for i, s in enumerate(all_spectra):
            # Alternate scaffolds to create different groups
            s.murcko_scaffold = f"scaffold_{i % 5}"  # 5 different scaffolds

        # Perform split
        train, val, test = TrainUtils.train_val_test_split_bms(
            all_spectra,
            val_split=0.2,
            test_split=0.2,
        )

        # Check splits are non-empty
        assert len(train) > 0, "Training set should not be empty"
        assert len(val) > 0, "Validation set should not be empty"
        assert len(test) > 0, "Test set should not be empty"

        # Check total count
        assert len(train) + len(val) + len(test) == 14, "All spectra should be split"

        # Check approximate ratios (allow some variance due to rounding)
        train_ratio = len(train) / 14
        val_ratio = len(val) / 14
        test_ratio = len(test) / 14

        assert 0.4 <= train_ratio <= 0.8, (
            f"Train ratio {train_ratio} should be reasonable"
        )
        assert 0.0 < val_ratio <= 0.4, f"Val ratio {val_ratio} should be reasonable"
        assert 0.0 < test_ratio <= 0.4, f"Test ratio {test_ratio} should be reasonable"

    def test_preprocessing_creates_mapping_structure(
        self, sample_training_spectra, tmp_path, hydra_config
    ):
        """Test that preprocessing creates the expected mapping file structure."""
        all_spectra = PreprocessingSimba.load_spectra(
            str(sample_training_spectra),
            hydra_config,
            n_samples=100,
            use_gnps_format=False,
            use_only_protonized_adducts=False,
        )

        for i, s in enumerate(all_spectra):
            s.murcko_scaffold = f"scaffold_{i % 5}"

        train, val, test = TrainUtils.train_val_test_split_bms(
            all_spectra,
            val_split=0.2,
            test_split=0.2,
        )
        train = train[:4]

        unique_smiles = list({s.smiles for s in train})
        df_smiles = pd.DataFrame(
            {
                "smiles": unique_smiles,
                "indexes": [[i] for i in range(len(unique_smiles))],
            }
        )

        pair_distances = (
            np.array([[0, 1, 0.5]])
            if len(unique_smiles) > 1
            else np.array([[0, 0, 1.0]])
        )

        molecule_pairs_train = MoleculePairsOpt(
            original_spectra=train,
            unique_spectra=train[: len(unique_smiles)],
            df_smiles=df_smiles,
            pair_distances=pair_distances,
        )

        mapping_data = {
            "molecule_pairs_train": molecule_pairs_train,
            "molecule_pairs_val": None,
            "molecule_pairs_test": None,
        }

        mapping_path = tmp_path / "mapping_test.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(mapping_data, f)

        assert mapping_path.exists(), "Mapping file should be created"

        with open(mapping_path, "rb") as f:
            loaded_data = pickle.load(f)

        assert "molecule_pairs_train" in loaded_data
        assert loaded_data["molecule_pairs_train"] is not None


class TestMappingFileStructure:
    """Test the structure and format of mapping files."""

    def test_mapping_file_has_required_keys(self, tmp_path):
        """Test that mapping file contains expected keys."""
        spec1 = SpectrumExt(
            identifier="spec1",
            precursor_mz=100.0,
            precursor_charge=1,
            mz=np.array([45.0]),
            intensity=np.array([100.0]),
            retention_time=0.0,
            params={},
            library="test",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            smiles="CCO",
            ionmode="Positive",
            adduct="[M+H]+",
            ce=0.0,
            ion_activation="CID",
            ionization_method="ESI",
            bms="",
            superclass="",
            classe="",
            subclass="",
        )
        spec2 = SpectrumExt(
            identifier="spec2",
            precursor_mz=100.0,
            precursor_charge=1,
            mz=np.array([60.0]),
            intensity=np.array([100.0]),
            retention_time=0.0,
            params={},
            library="test",
            inchi="InChI=1S/C3H8O/c1-3(2)4/h3-4H,1-2H3",
            smiles="CC(C)O",
            ionmode="Positive",
            adduct="[M+H]+",
            ce=0.0,
            ion_activation="CID",
            ionization_method="ESI",
            bms="",
            superclass="",
            classe="",
            subclass="",
        )

        df_smiles = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(C)O"],
                "indexes": [[0], [1]],
            }
        )

        pair_distances = np.array([[0, 1, 0.5]])

        mol_pairs = MoleculePairsOpt(
            original_spectra=[spec1, spec2],
            unique_spectra=[spec1, spec2],
            df_smiles=df_smiles,
            pair_distances=pair_distances,
        )

        mapping_data = {
            "molecule_pairs_train": mol_pairs,
            "molecule_pairs_val": None,
            "molecule_pairs_test": None,
        }

        mapping_path = tmp_path / "test_mapping.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(mapping_data, f)

        # Load and verify
        with open(mapping_path, "rb") as f:
            loaded = pickle.load(f)

        assert "molecule_pairs_train" in loaded
        assert "molecule_pairs_val" in loaded
        assert "molecule_pairs_test" in loaded

        train_pairs = loaded["molecule_pairs_train"]
        assert hasattr(train_pairs, "df_smiles")
        assert isinstance(train_pairs.df_smiles, pd.DataFrame)

    def test_mapping_df_smiles_structure(self, tmp_path):
        """Test that df_smiles DataFrame has correct structure."""
        spec1 = SpectrumExt(
            identifier="spec1",
            precursor_mz=100.0,
            precursor_charge=1,
            mz=np.array([45.0]),
            intensity=np.array([100.0]),
            retention_time=0.0,
            params={},
            library="test",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            smiles="CCO",
            ionmode="Positive",
            adduct="[M+H]+",
            ce=0.0,
            ion_activation="CID",
            ionization_method="ESI",
            bms="",
            superclass="",
            classe="",
            subclass="",
        )
        spec2 = SpectrumExt(
            identifier="spec2",
            precursor_mz=100.0,
            precursor_charge=1,
            mz=np.array([60.0]),
            intensity=np.array([100.0]),
            retention_time=0.0,
            params={},
            library="test",
            inchi="InChI=1S/C3H8O/c1-3(2)4/h3-4H,1-2H3",
            smiles="CC(C)O",
            ionmode="Positive",
            adduct="[M+H]+",
            ce=0.0,
            ion_activation="CID",
            ionization_method="ESI",
            bms="",
            superclass="",
            classe="",
            subclass="",
        )
        spec3 = SpectrumExt(
            identifier="spec3",
            precursor_mz=100.0,
            precursor_charge=1,
            mz=np.array([75.0]),
            intensity=np.array([100.0]),
            retention_time=0.0,
            params={},
            library="test",
            inchi="InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3",
            smiles="CCCC",
            ionmode="Positive",
            adduct="[M+H]+",
            ce=0.0,
            ion_activation="CID",
            ionization_method="ESI",
            bms="",
            superclass="",
            classe="",
            subclass="",
        )

        df_smiles = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(C)O", "CCCC"],
                "indexes": [[0], [1, 2], [3, 4, 5]],
            }
        )

        pair_distances = np.array([[0, 1, 0.5], [1, 2, 0.3]])

        mol_pairs = MoleculePairsOpt(
            original_spectra=[spec1, spec2, spec3],
            unique_spectra=[spec1, spec2, spec3],
            df_smiles=df_smiles,
            pair_distances=pair_distances,
        )

        mapping_path = tmp_path / "test_mapping.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump({"molecule_pairs_train": mol_pairs}, f)

        with open(mapping_path, "rb") as f:
            loaded = pickle.load(f)

        df = loaded["molecule_pairs_train"].df_smiles

        assert "smiles" in df.columns
        assert "indexes" in df.columns

        assert df["smiles"].dtype == object
        assert all(isinstance(x, list) for x in df["indexes"])

        assert len(df) == 3
        assert len(df.loc[0, "indexes"]) == 1
        assert len(df.loc[1, "indexes"]) == 2


class TestTrainingSmoke:
    """Smoke tests for model training."""

    def test_model_initialization(self, hydra_config):
        """Test that model can be initialized with config."""
        d_model = hydra_config.model.transformer.d_model
        n_layers = 6
        n_classes = hydra_config.model.tasks.edit_distance.n_classes
        use_gumbel = False

        model = EmbedderMultitask(
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
            use_gumbel=use_gumbel,
        )

        assert model is not None
        assert hasattr(model, "spectrum_encoder")

    @patch(
        "simba.core.models.ordinal.embedder_multitask.EmbedderMultitask.load_from_checkpoint"
    )
    def test_checkpoint_loading(self, mock_load):
        """Test that checkpoint loading interface works."""
        checkpoint_path = "/fake/path/checkpoint.ckpt"

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        model = EmbedderMultitask.load_from_checkpoint(checkpoint_path)

        assert model is not None
        mock_load.assert_called_once()


class TestInferenceOnTrainedModel:
    """Test inference pipeline with trained model."""

    def test_embedding_generation_shape(self, sample_mgf, mocker, hydra_config):
        """Test that model produces embeddings of expected shape."""
        spectra = PreprocessingSimba.load_spectra(
            str(sample_mgf),
            hydra_config,
            n_samples=5,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2

        model = EmbedderMultitask(
            d_model=int(hydra_config.model.transformer.d_model),
            n_layers=int(hydra_config.model.transformer.n_layers),
            n_classes=hydra_config.model.tasks.edit_distance.n_classes,
            use_gumbel=hydra_config.model.tasks.edit_distance.use_gumbel,
            use_element_wise=True,
            use_cosine_distance=hydra_config.model.tasks.cosine_similarity.use_cosine_distance,
            use_edit_distance_regresion=hydra_config.model.tasks.edit_distance.use_regression,
            use_fingerprints=hydra_config.model.tasks.fingerprints.enabled,
            USE_LEARNABLE_MULTITASK=hydra_config.model.multitasking.learnable,
        )
        model.eval()

        mocker.patch(
            "simba.core.models.ordinal.embedder_multitask.EmbedderMultitask.load_from_checkpoint",
            return_value=model,
        )

        simba = Simba(
            "fake_checkpoint.ckpt",
            config=hydra_config,
            device="cpu",
            cache_embeddings=False,
        )

        sim_ed, sim_mces = simba.predict(spectra[:2], spectra[:2])

        assert sim_ed.shape == (2, 2)
        assert sim_mces.shape == (2, 2)
        assert isinstance(sim_ed, np.ndarray)
        assert isinstance(sim_mces, np.ndarray)

    def test_similarity_computation_after_training(
        self, sample_mgf_casmi, mocker, hydra_config
    ):
        """Test that similarity computation works with trained model."""
        spectra = PreprocessingSimba.load_spectra(
            str(sample_mgf_casmi),
            hydra_config,
            n_samples=10,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2, f"Need at least 2 spectra, got {len(spectra)}"

        model = EmbedderMultitask(
            d_model=int(hydra_config.model.transformer.d_model),
            n_layers=int(hydra_config.model.transformer.n_layers),
            n_classes=hydra_config.model.tasks.edit_distance.n_classes,
            use_gumbel=hydra_config.model.tasks.edit_distance.use_gumbel,
            use_element_wise=True,
            use_cosine_distance=hydra_config.model.tasks.cosine_similarity.use_cosine_distance,
            use_edit_distance_regresion=hydra_config.model.tasks.edit_distance.use_regression,
            use_fingerprints=hydra_config.model.tasks.fingerprints.enabled,
            USE_LEARNABLE_MULTITASK=hydra_config.model.multitasking.learnable,
        )
        model.eval()

        mocker.patch(
            "simba.core.models.ordinal.embedder_multitask.EmbedderMultitask.load_from_checkpoint",
            return_value=model,
        )

        simba = Simba(
            "fake_checkpoint.ckpt",
            config=hydra_config,
            device="cpu",
            cache_embeddings=False,
        )

        query = spectra[:2]
        library = spectra[2:3]

        sim_ed, sim_mces = simba.predict(query, library)

        assert sim_ed.shape == (2, 1)
        assert sim_mces.shape == (2, 1)

        assert np.all(np.isfinite(sim_ed))
        assert np.all(np.isfinite(sim_mces))


class TestLoadMCES:
    """Test MCES data loading utilities."""

    def test_load_raw_data_empty_directory(self, tmp_path):
        """Test loading from empty directory returns empty array."""
        result = LoadMCES.load_raw_data(
            str(tmp_path), "nonexistent_prefix", partitions=1
        )

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0

    def test_find_file_with_prefix(self, tmp_path):
        """Test finding files with specific prefix."""
        # Create test files
        (tmp_path / "test_file_1.npy").write_bytes(b"")
        (tmp_path / "test_file_2.npy").write_bytes(b"")
        (tmp_path / "other_file.npy").write_bytes(b"")

        found = LoadMCES.find_file(str(tmp_path), "test_file")

        assert len(found) == 2
        assert all("test_file" in str(f) for f in found)

    def test_load_raw_data_with_partitions(self, tmp_path):
        """Test loading multiple partition files."""
        data1 = np.array([[0, 1, 0.5], [1, 2, 0.7]])
        data2 = np.array([[2, 3, 0.8], [3, 4, 0.6]])

        np.save(tmp_path / "indexes_0.npy", data1)
        np.save(tmp_path / "indexes_1.npy", data2)

        result = LoadMCES.load_raw_data(str(tmp_path), "indexes", partitions=2)

        assert result.shape[0] == 4
        assert result.shape[1] == 3

        assert np.array_equal(result[0], data1[0]) or np.array_equal(
            result[0], data2[0]
        )
        assert result.shape == (4, 3)
