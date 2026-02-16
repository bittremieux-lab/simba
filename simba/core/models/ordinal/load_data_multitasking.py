import copy

import numpy as np

from simba.core.chemistry.chem_utils import ADDUCT_TO_MASS
from simba.core.chemistry.tanimoto import Tanimoto
from simba.core.data.encoding import (
    ION_ACTIVATION,
    IONIZATION_METHODS,
    encode_adduct,
    encode_ion_activation,
    encode_ionization_method,
)
from simba.core.data.molecule_pairs_opt import MoleculePairsOpt
from simba.core.data.preprocessor import Preprocessor
from simba.core.models.ordinal.ordinal_classification import (
    OrdinalClassification,
)
from simba.core.models.transformers.CustomDatasetMultitasking import (
    CustomDatasetMultitasking,
)
from simba.utils.logger_setup import logger


class LoadDataMultitasking:
    """
    using unique identifiers
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

        Returns
        -------
        CustomDatasetMultitasking
            The Pytorch dataset.
        """
        # copy spectrums to avoid overwriting
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
        if use_adduct:
            ionmode = np.zeros(
                (len(molecule_pairs.original_spectra), 1), dtype=np.float32
            )
            adduct = np.zeros(
                (
                    len(molecule_pairs.original_spectra),
                    len(ADDUCT_TO_MASS.keys()),
                ),
                dtype=np.float32,
            )
        if use_ce:
            ce = np.zeros((len(molecule_pairs.original_spectra), 1), dtype=np.int32)
        if use_ion_activation:
            ia = np.zeros(
                (
                    len(molecule_pairs.original_spectra),
                    len(ION_ACTIVATION),
                ),
                dtype=np.int32,
            )
        if use_ion_method:
            im = np.zeros(
                (
                    len(molecule_pairs.original_spectra),
                    len(IONIZATION_METHODS),
                ),
                dtype=np.int32,
            )

        logger.info("Loading mz, intensity and precursor data ...")
        for i, spec in enumerate(molecule_pairs.original_spectra):
            # check for maximum length
            length = len(spec.mz) if len(spec.mz) <= max_num_peaks else max_num_peaks

            # assign the values to the array
            mz[i, 0:length] = np.array(spec.mz[0:length])
            intensity[i, 0:length] = np.array(spec.intensity[0:length])

            precursor_mass[i] = spec.precursor_mz
            precursor_charge[i] = spec.precursor_charge

            if use_adduct:
                if (spec.ionmode is None) or (
                    spec.ionmode == "None"
                ):  # TODO: check if the 2nd condition is needed
                    ionmode[i] = 0
                else:
                    ionmode[i] = 1.0 if spec.ionmode == "positive" else -1.0
                adduct[i] = encode_adduct(spec.adduct)

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
        ed = OrdinalClassification.from_float_to_class(
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
            mz=mz,
            intensity=intensity,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            df_smiles=molecule_pairs_input.df_smiles,
            use_fingerprints=use_fingerprints,
            fingerprint_0=fingerprint_0,
            max_num_peaks=max_num_peaks,
            use_adduct=use_adduct,
            ionmode=(ionmode if use_adduct else None),
            adduct=(adduct if use_adduct else None),
            use_ce=use_ce,
            ce=(ce if use_ce else None),
            use_ion_activation=use_ion_activation,
            ion_activation=(ia if use_ion_activation else None),
            use_ion_method=use_ion_method,
            ion_method=(im if use_ion_method else None),
        )
