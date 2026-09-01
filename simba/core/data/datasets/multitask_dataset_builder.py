import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from simba.core.chemistry.chem_utils import (
    ADDUCT_TO_MASS,
    normalize_instrument_type,
    theoretical_precursor_mz,
)
from simba.core.chemistry.tanimoto import Tanimoto
from simba.core.data.datasets.multitask_dataset import (
    CustomDatasetMultitasking,
)
from simba.core.data.encoding import (
    ION_ACTIVATION,
    IONIZATION_METHODS,
    encode_adduct_mass,
    encode_ion_activation,
    encode_ionization_method,
)
from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.data.preprocessor import Preprocessor
from simba.utils.binning import float_to_ordinal_class
from simba.utils.logger_setup import logger


class MultitaskDataBuilder:
    """
    Class for building the dataset for multitask learning.
    """

    @staticmethod
    def from_molecule_pairs_to_dataset(
        molecule_pairs_input: MoleculePairsOpt,
        max_num_peaks: int,
        training: bool = False,  # shuffle the spectrum 0 and 1 for data augmentation
        n_classes: int = 6,
        use_fingerprints: bool = False,
        use_adduct: bool = False,
        use_ce: bool = False,
        use_ion_activation: bool = False,
        use_ion_method: bool = False,
        use_ion_mode: bool = False,
        precursor_mass_mode: str = "measured",
        precursor_noise_mode: str = "legacy",
        iceberg_spectra_prob: float = 0.0,
    ) -> CustomDatasetMultitasking:
        """
        Load data from molecule pairs into a Pytorch dataset for multitask learning.
        Includes preprocessing of the spectra.

        Parameters
        ----------
        molecule_pairs_input: MoleculePairsOpt
            The molecule pairs to load into the dataset.
        max_num_peaks: int
            The maximum number of peaks in a spectrum. Other peaks will be removed.
        training: bool
            Dataset for training or not.
        n_classes: int
            Number of classes for edit distance.
        use_fingerprints: bool
            Use fingerprints or not.
        use_adduct: bool
            Use adduct information or not.
        use_ce: bool
            Use collision energy or not.
        precursor_mass_mode: str
            "measured" (default): use each spectrum's own precursor m/z, as read
            from the MGF. "theoretical": compute it instead from the molecule's
            SMILES (RDKit ExactMolWt) and adduct, discarding the measured value.
        precursor_noise_mode: str
            Passed through to the built dataset, which applies it as the
            training-time precursor-mass augmentation mode -- see
            `Augmentation.augment`'s `precursor_noise_mode` for the options.
        iceberg_spectra_prob: float
            Passed through to the built dataset, which uses it as the
            per-sample probability of drawing a synthetic ICEBERG-predicted
            spectrum instead of a real one for a molecule that has both
            (only meaningful for the train split -- val/test use
            deterministic first/last selection regardless).

        Returns
        -------
        CustomDatasetMultitasking
            The Pytorch dataset.
        """
        if precursor_mass_mode not in ("measured", "theoretical"):
            raise ValueError(f"Unknown precursor_mass_mode: {precursor_mass_mode!r}")
        # copy spectrums to avoid overwriting
        print(
            f"DEBUG: Size of incoming spectra: {len(molecule_pairs_input.original_spectra)}"
        )
        print(
            f"DEBUG: Size of pair distances: {molecule_pairs_input.pair_distances.shape}"
        )
        molecule_pairs = MoleculePairsOpt(
            original_spectra=[
                copy.copy(s) for s in molecule_pairs_input.original_spectra
            ],
            unique_spectra=molecule_pairs_input.spectra,
            df_smiles=molecule_pairs_input.df_smiles,
            pair_distances=molecule_pairs_input.pair_distances,
            extra_distances=molecule_pairs_input.extra_distances,
        )

        # Preprocess the spectra
        pp = Preprocessor()
        logger.info("Preprocess all spectra ...")
        molecule_pairs.original_spectra = pp.preprocess_all_spectra(
            molecule_pairs.original_spectra,
            max_num_peaks=max_num_peaks,
            training=training,
        )

        # Get the mz, intensity values and precursor data
        mz = np.zeros(
            (len(molecule_pairs.original_spectra), max_num_peaks),
            dtype=np.float32,
        )
        intensity = np.zeros(
            (len(molecule_pairs.original_spectra), max_num_peaks),
            dtype=np.float32,
        )
        precursor_mass = np.zeros(
            (len(molecule_pairs.original_spectra), 1), dtype=np.float32
        )
        precursor_charge = np.zeros(
            (len(molecule_pairs.original_spectra), 1), dtype=np.int32
        )
        ionmode = np.zeros((len(molecule_pairs.original_spectra), 1), dtype=np.float32)
        adduct = np.zeros(
            (
                len(molecule_pairs.original_spectra),
                len(ADDUCT_TO_MASS.keys()),
            ),
            dtype=np.float32,
        )
        ce = np.zeros((len(molecule_pairs.original_spectra), 1), dtype=np.int32)
        ia = np.zeros(
            (
                len(molecule_pairs.original_spectra),
                len(ION_ACTIVATION),
            ),
            dtype=np.int32,
        )
        im = np.zeros(
            (
                len(molecule_pairs.original_spectra),
                len(IONIZATION_METHODS),
            ),
            dtype=np.int32,
        )
        instrument = np.full(
            len(molecule_pairs.original_spectra), "unknown", dtype=object
        )

        logger.info("Loading mz, intensity and precursor data ...")
        print(
            f"DEBUG: Size of original spectra loaded into CustomDataset {len(molecule_pairs.original_spectra)}"
        )
        for i, spec in enumerate(molecule_pairs.original_spectra):
            # check for maximum length
            length = len(spec.mz) if len(spec.mz) <= max_num_peaks else max_num_peaks

            # assign the values to the array
            mz[i, 0:length] = np.array(spec.mz[0:length])
            intensity[i, 0:length] = np.array(spec.intensity[0:length])

            if precursor_mass_mode == "theoretical":
                mol = Chem.MolFromSmiles(spec.smiles)
                neutral_mass = ExactMolWt(mol)
                precursor_mass[i] = theoretical_precursor_mz(
                    neutral_mass, spec.params["adduct"]
                )
            else:
                precursor_mass[i] = spec.precursor_mz
            precursor_charge[i] = spec.precursor_charge
            instrument[i] = normalize_instrument_type(
                getattr(spec, "instrument", None)
            )

            if use_ion_mode:
                if (spec.ionmode is None) or (
                    spec.ionmode == "None"
                ):  # TODO: check if the 2nd condition is needed
                    ionmode[i] = 0
                else:
                    ionmode[i] = 1.0 if spec.ionmode == "positive" else -1.0
            if use_adduct:
                adduct[i] = encode_adduct_mass(spec.params["adduct"])

            if use_ce:
                if (spec.ce is None) or (spec.ce == "None"):
                    ce[i] = 0  # TODO: array dtype -> int
                else:
                    ce[i] = spec.ce

            if use_ion_activation:
                if (spec.ion_activation is None) or (spec.ion_activation == "None"):
                    ia[i] = np.zeros(len(ION_ACTIVATION), dtype=np.int32)
                else:
                    ia[i] = encode_ion_activation(spec.ion_activation)

            if use_ion_method:
                if (spec.ionization_method is None) or (
                    spec.ionization_method == "None"
                ):
                    im[i] = np.zeros(
                        len(IONIZATION_METHODS),
                        dtype=np.int32,
                    )
                else:
                    im[i] = encode_ionization_method(spec.ionization_method)

        # logger.info("Normalizing intensities")
        # Normalize the intensity array
        # intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))

        # Adjust ED towards a N classification problem
        ed = float_to_ordinal_class(
            molecule_pairs_input.pair_distances[:, 2].reshape(-1, 1),
            n_classes=n_classes,
        )

        if molecule_pairs.extra_distances is None:
            raise ValueError("extra_distances must be provided for multitask training.")
        mces = molecule_pairs.extra_distances.reshape(-1, 1)

        if use_fingerprints:
            logger.info("Computing molecular fingerprints...")
            fingerprint_0 = np.array(
                [
                    np.array(Tanimoto.compute_fingerprint(s.params["smiles"]))
                    for s in molecule_pairs_input.spectra
                ]
            )
        else:
            fingerprint_0 = np.array([0 for m in molecule_pairs_input.spectra])

        dictionary_data = {
            "index_unique_0": molecule_pairs_input.pair_distances[:, 0].reshape(-1, 1),
            "index_unique_1": molecule_pairs_input.pair_distances[:, 1].reshape(-1, 1),
            "ed": ed,
            "mces": mces,
            # "fingerprint_0": fingerprint_0,
        }

        return CustomDatasetMultitasking(
            dictionary_data,
            training=training,
            iceberg_spectra_prob=iceberg_spectra_prob,
            mz=mz,
            intensity=intensity,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            instrument=instrument,
            precursor_noise_mode=precursor_noise_mode,
            df_smiles=molecule_pairs_input.df_smiles,
            use_fingerprints=use_fingerprints,
            fingerprint_0=fingerprint_0,
            max_num_peaks=max_num_peaks,
            use_adduct=use_adduct,
            use_ion_mode=use_ion_mode,
            ionmode=ionmode,
            adduct=adduct,
            use_ce=use_ce,
            ce=ce,
            use_ion_activation=use_ion_activation,
            ion_activation=ia,
            use_ion_method=use_ion_method,
            ion_method=im,
        )
