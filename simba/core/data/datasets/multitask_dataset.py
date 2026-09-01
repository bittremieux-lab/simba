import random

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from simba.core.chemistry.chem_utils import ADDUCT_TO_MASS
from simba.core.data.augmentation import Augmentation
from simba.core.data.encoding import (
    ION_ACTIVATION,
    IONIZATION_METHODS,
)


class CustomDatasetMultitasking(Dataset):
    def __init__(
        self,
        your_dict,
        training=False,
        prob_aug=0.50,
        iceberg_spectra_prob=0.0,
        mz=None,
        intensity=None,
        precursor_mass=None,
        precursor_charge=None,
        instrument=None,
        precursor_noise_mode="legacy",
        df_smiles=None,
        use_fingerprints=False,
        fingerprint_0=None,
        max_num_peaks=None,
        use_adduct=False,
        ionmode=None,
        adduct=None,
        use_ce=False,
        ce=None,
        use_ion_activation=False,
        ion_activation=None,
        use_ion_method=False,
        ion_method=None,
        use_ion_mode=False,
    ):
        self.data = your_dict
        self.keys = list(your_dict.keys())
        self.training = training
        self.prob_aug = prob_aug
        self.iceberg_spectra_prob = iceberg_spectra_prob

        self.mz = mz
        self.intensity = intensity
        self.precursor_mass = precursor_mass
        self.precursor_charge = precursor_charge
        self.instrument = instrument
        self.precursor_noise_mode = precursor_noise_mode
        self.df_smiles = df_smiles  ### df with rows smiles, indexes
        self.use_fingerprints = use_fingerprints
        self.use_adduct = use_adduct
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method
        self.use_ion_mode = use_ion_mode

        if self.use_fingerprints:
            self.fingerprint_0 = fingerprint_0
        self.max_num_peaks = max_num_peaks

        self.adduct_mass = adduct
        self.ionmode = ionmode
        self.ce = ce

        self.ion_activation = ion_activation

        self.ion_method = ion_method

    def __len__(self):
        return len(self.data[self.keys[0]])
        # return len(self.keys)

    def get_original_dictionary(self, max_num_peaks=100):
        """
        get a dictionary containing the spectrums mapped
        """
        len_data = self.data[self.keys[0]].shape[0]
        ## Get the mz, intensity values and precursor data

        dictionary = {}
        dictionary["mz_0"] = np.zeros((len_data, max_num_peaks), dtype=np.float32)
        dictionary["intensity_0"] = np.zeros(
            (len_data, max_num_peaks), dtype=np.float32
        )
        dictionary["mz_1"] = np.zeros((len_data, max_num_peaks), dtype=np.float32)
        dictionary["intensity_1"] = np.zeros(
            (len_data, max_num_peaks), dtype=np.float32
        )
        dictionary["ed"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["mces"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["precursor_mass_0"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["precursor_charge_0"] = np.zeros((len_data, 1), dtype=np.int32)
        dictionary["precursor_mass_1"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["precursor_charge_1"] = np.zeros((len_data, 1), dtype=np.int32)

        ### add extra metadata in case it is necessary
        if self.use_adduct:
            dictionary["ionmode_0"] = np.zeros((len_data, 1), dtype=np.float32)
            dictionary["ionmode_1"] = np.zeros((len_data, 1), dtype=np.float32)
            dictionary["adduct_0"] = np.zeros(
                (len_data, len(ADDUCT_TO_MASS.keys())), dtype=np.float32
            )
            dictionary["adduct_1"] = np.zeros(
                (len_data, len(ADDUCT_TO_MASS.keys())), dtype=np.float32
            )

        if self.use_ce:
            dictionary["ce_0"] = np.zeros((len_data, 1), dtype=np.float32)
            dictionary["ce_1"] = np.zeros((len_data, 1), dtype=np.float32)

        if self.use_ion_activation:
            dictionary["ion_activation_0"] = np.zeros(
                (len_data, len(ION_ACTIVATION)), dtype=np.float32
            )
            dictionary["ion_activation_1"] = np.zeros(
                (len_data, len(ION_ACTIVATION)), dtype=np.float32
            )

        if self.use_ion_method:
            dictionary["ion_method_0"] = np.zeros(
                (len_data, len(IONIZATION_METHODS)), dtype=np.float32
            )
            dictionary["ion_method_1"] = np.zeros(
                (len_data, len(IONIZATION_METHODS)), dtype=np.float32
            )

        if self.use_fingerprints:
            print("Defining fingerprints ...")
            dictionary["fingerprint_0"] = np.zeros((len_data, 2048), dtype=np.int32)

        for idx in tqdm(range(0, len_data)):
            sample_unique = {k: self.data[k][idx] for k in self.keys}

            indexes_unique_0 = sample_unique["index_unique_0"]
            indexes_unique_1 = sample_unique["index_unique_1"]

            print(f"value of indexes_unique_0 {indexes_unique_0} ")
            indexes_original_0 = self.df_smiles.loc[int(indexes_unique_0), "indexes"][0]

            indexes_original_1 = self.df_smiles.loc[int(indexes_unique_1), "indexes"][0]

            dictionary["mz_0"][idx] = self.mz[indexes_original_0].astype(np.float32)
            dictionary["intensity_0"][idx] = self.intensity[indexes_original_0].astype(
                np.float32
            )

            dictionary["mz_1"][idx] = self.mz[indexes_original_1].astype(np.float32)
            dictionary["intensity_1"][idx] = self.intensity[indexes_original_1].astype(
                np.float32
            )
            dictionary["precursor_mass_0"][idx] = self.precursor_mass[
                indexes_original_0
            ].astype(np.float32)
            dictionary["precursor_mass_1"][idx] = self.precursor_mass[
                indexes_original_1
            ].astype(np.float32)
            dictionary["precursor_charge_0"][idx] = self.precursor_charge[
                indexes_original_0
            ].astype(np.float32)
            dictionary["precursor_charge_1"][idx] = self.precursor_charge[
                indexes_original_1
            ].astype(np.float32)
            dictionary["ed"][idx] = sample_unique["ed"].astype(np.float32)
            dictionary["mces"][idx] = sample_unique["mces"].astype(np.float32)
            if self.use_ion_mode:
                dictionary["ionmode_0"][idx] = self.ionmode[indexes_original_0].astype(
                    np.float32
                )
                dictionary["ionmode_1"][idx] = self.ionmode[indexes_original_1].astype(
                    np.float32
                )
            if self.use_adduct:
                dictionary["adduct_0"][idx] = self.adduct_mass[
                    indexes_original_0
                ].astype(np.float32)
                dictionary["adduct_1"][idx] = self.adduct_mass[
                    indexes_original_1
                ].astype(np.float32)

            if self.use_ce:
                dictionary["ce_0"][idx] = self.ce[indexes_original_0].astype(np.float32)
                dictionary["ce_1"][idx] = self.ce[indexes_original_1].astype(np.float32)

            if self.use_ion_activation:
                dictionary["ion_activation_0"][idx] = self.ion_activation[
                    indexes_original_0
                ].astype(np.float32)
                dictionary["ion_activation_1"][idx] = self.ion_activation[
                    indexes_original_1
                ].astype(np.float32)

            if self.use_ion_method:
                dictionary["ion_method_0"][idx] = self.ion_method[
                    indexes_original_0
                ].astype(np.float32)
                dictionary["ion_method_1"][idx] = self.ion_method[
                    indexes_original_1
                ].astype(np.float32)

            if self.use_fingerprints:
                dictionary["fingerprint_0"][idx] = self.fingerprint_0[
                    indexes_original_0
                ].astype(np.float32)

        return dictionary

    def _sample_spectrum_index(self, mol_idx: int) -> int:
        """Pick one spectrum index for this molecule -- with probability
        `self.iceberg_spectra_prob`, a synthetic ICEBERG-predicted one
        instead of a real one, if this molecule actually has any (see
        NOTES_014_2_ICEBERG_AUGMENTATION.md). Falls back to a real spectrum
        whenever synthetic ones aren't available (df_smiles lacks the
        column entirely, or this particular molecule has none) or the
        random draw doesn't land on synthetic -- iceberg_spectra_prob=0.0
        (the default) reproduces the original random.choice(indexes)
        behavior exactly, unaffected by whether the column exists at all."""
        if self.iceberg_spectra_prob > 0 and "synthetic_indexes" in self.df_smiles.columns:
            synthetic_indexes = self.df_smiles.loc[mol_idx, "synthetic_indexes"]
            if synthetic_indexes and random.random() < self.iceberg_spectra_prob:
                return random.choice(synthetic_indexes)
        return random.choice(self.df_smiles.loc[mol_idx, "indexes"])

    def __getitem__(self, idx):
        sample = {k: self.data[k][idx] for k in self.keys}

        idx_0 = sample["index_unique_0"]
        idx_1 = sample["index_unique_1"]

        if self.training:
            # select random samples
            idx_0_original = self._sample_spectrum_index(int(idx_0[0]))
            idx_1_original = self._sample_spectrum_index(int(idx_1[0]))
        else:
            # select the first index
            idx_0_original = self.df_smiles.loc[int(idx_0[0]), "indexes"][0]
            # select the last index
            idx_1_original = self.df_smiles.loc[int(idx_1[0]), "indexes"][-1]

        # Get the original spectrum based on indexes
        spectrum_sample = {}
        spectrum_sample["mz_0"] = self.mz[idx_0_original].astype(np.float32)
        spectrum_sample["intensity_0"] = self.intensity[idx_0_original].astype(
            np.float32
        )
        spectrum_sample["mz_1"] = self.mz[idx_1_original].astype(np.float32)
        spectrum_sample["intensity_1"] = self.intensity[idx_1_original].astype(
            np.float32
        )
        spectrum_sample["precursor_mass_0"] = self.precursor_mass[
            idx_0_original
        ].astype(np.float32)
        spectrum_sample["precursor_mass_1"] = self.precursor_mass[
            idx_1_original
        ].astype(np.float32)
        spectrum_sample["precursor_charge_0"] = self.precursor_charge[
            idx_0_original
        ].astype(np.float32)
        spectrum_sample["precursor_charge_1"] = self.precursor_charge[
            idx_1_original
        ].astype(np.float32)
        spectrum_sample["ed"] = sample["ed"].astype(np.float32)
        spectrum_sample["mces"] = sample["mces"].astype(np.float32)

        if self.instrument is not None:
            spectrum_sample["instrument_0"] = self.instrument[idx_0_original]
            spectrum_sample["instrument_1"] = self.instrument[idx_1_original]

        # Pair identity metadata (molecule + resolved spectrum indices, SMILES) --
        # not used by the model itself (forward() only reads specific known keys,
        # never iterates batch generically, so these pass through untouched), but
        # needed downstream to save per-pair validation results to CSV. Molecule
        # index alone isn't enough: a molecule can have multiple spectra, and for
        # a self-pair (idx_0==idx_1 at the molecule level) the resolved spectrum
        # indices can genuinely differ (this branch always takes indexes[0] for
        # side 0 and indexes[-1] for side 1) -- so self-pairs aren't necessarily
        # "identical spectrum vs itself", they can be two different real
        # measurements of the same molecule. Both indices are recorded so that
        # distinction is visible rather than assumed.
        spectrum_sample["mol_idx_0"] = int(idx_0[0])
        spectrum_sample["mol_idx_1"] = int(idx_1[0])
        spectrum_sample["spec_idx_0"] = int(idx_0_original)
        spectrum_sample["spec_idx_1"] = int(idx_1_original)
        spectrum_sample["smiles_0"] = self.df_smiles.loc[int(idx_0[0]), "canon_smiles"]
        spectrum_sample["smiles_1"] = self.df_smiles.loc[int(idx_1[0]), "canon_smiles"]

        if self.use_fingerprints:
            ind = int(idx_0[0])
            if self.training:
                if (ind % 2) == 0:
                    spectrum_sample["fingerprint_0"] = self.fingerprint_0[ind].astype(
                        np.float32
                    )
                else:
                    # return 0s
                    spectrum_sample["fingerprint_0"] = 0 * self.fingerprint_0[
                        ind
                    ].astype(np.float32)
            else:
                spectrum_sample["fingerprint_0"] = self.fingerprint_0[ind].astype(
                    np.float32
                )

        if self.use_ion_mode:
            spectrum_sample["ionmode_0"] = self.ionmode[idx_0_original].astype(
                np.float32
            )
            spectrum_sample["ionmode_1"] = self.ionmode[idx_1_original].astype(
                np.float32
            )

        else:
            spectrum_sample["ionmode_0"] = 0 * self.ionmode[idx_0_original].astype(
                np.float32
            )
            spectrum_sample["ionmode_1"] = 0 * self.ionmode[idx_1_original].astype(
                np.float32
            )

        if self.use_adduct:
            spectrum_sample["adduct_0"] = self.adduct_mass[idx_0_original].astype(
                np.float32
            )
            spectrum_sample["adduct_1"] = self.adduct_mass[idx_1_original].astype(
                np.float32
            )
        else:
            spectrum_sample["adduct_0"] = 0 * self.adduct_mass[idx_0_original].astype(
                np.float32
            )
            spectrum_sample["adduct_1"] = 0 * self.adduct_mass[idx_1_original].astype(
                np.float32
            )
        if self.use_ce:
            spectrum_sample["ce_0"] = self.ce[idx_0_original].astype(np.float32)
            spectrum_sample["ce_1"] = self.ce[idx_1_original].astype(np.float32)
        else:
            spectrum_sample["ce_0"] = 0 * self.ce[idx_0_original].astype(np.float32)
            spectrum_sample["ce_1"] = 0 * self.ce[idx_1_original].astype(np.float32)
        if self.use_ion_activation:
            spectrum_sample["ion_activation_0"] = self.ion_activation[
                idx_0_original
            ].astype(np.float32)
            spectrum_sample["ion_activation_1"] = self.ion_activation[
                idx_1_original
            ].astype(np.float32)
        else:
            spectrum_sample["ion_activation_0"] = 0 * self.ion_activation[
                idx_0_original
            ].astype(np.float32)
            spectrum_sample["ion_activation_1"] = 0 * self.ion_activation[
                idx_1_original
            ].astype(np.float32)
        if self.use_ion_method:
            spectrum_sample["ion_method_0"] = self.ion_method[idx_0_original].astype(
                np.float32
            )
            spectrum_sample["ion_method_1"] = self.ion_method[idx_1_original].astype(
                np.float32
            )
        else:
            spectrum_sample["ion_method_0"] = 0 * self.ion_method[
                idx_0_original
            ].astype(np.float32)
            spectrum_sample["ion_method_1"] = 0 * self.ion_method[
                idx_1_original
            ].astype(np.float32)
        if self.training and (random.random() < self.prob_aug):
            # augmentation
            spectrum_sample = Augmentation.augment(
                spectrum_sample,
                max_num_peaks=self.max_num_peaks,
                precursor_noise_mode=self.precursor_noise_mode,
            )

        # normalize
        spectrum_sample = Augmentation.normalize_intensities(spectrum_sample)
        return spectrum_sample
