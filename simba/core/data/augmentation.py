import copy
import random

import numpy as np

from simba.core.chemistry.chem_utils import resample_precursor_mz


class Augmentation:
    @staticmethod
    def zero_and_pack(arr, zero_idx):
        """
        put zeros to the array and move the nonzero values to the beginning
        """
        mask = np.ones(len(arr), dtype=bool)
        mask[zero_idx] = False
        complement_idx = np.where(mask)[0]

        arr = arr.copy()
        arr[zero_idx] = 0
        nonzero = arr[complement_idx]
        return np.concatenate(
            [nonzero, np.zeros(len(arr) - len(nonzero), dtype=arr.dtype)]
        )

    @staticmethod
    def augment(
        data_sample, training=False, max_num_peaks=None, precursor_noise_mode="legacy"
    ):
        new_sample = copy.deepcopy(data_sample)
        # new_sample = Augmentation.inversion(new_sample)
        # new_sample = Augmentation.add_noise_to_precursor_mass(new_sample)

        # peak augmentation
        new_sample = Augmentation.normalize_max(new_sample)
        new_sample = Augmentation.peak_augmentation_removal_noise(new_sample)
        new_sample = Augmentation.peak_augmentation_max_peaks(
            new_sample, max_peaks=max_num_peaks
        )

        # precursor mass noise: "legacy" = uniform +/-1%; "mist_cf" = Gaussian
        # noise scaled by instrument ppm tolerance; "none" = no perturbation.
        # new_sample = Augmentation.add_false_precursor_masses_negatives(new_sample)
        if precursor_noise_mode == "legacy":
            new_sample = Augmentation.add_false_precursor_masses_positives(new_sample)
        elif precursor_noise_mode == "mist_cf":
            new_sample = Augmentation.resample_precursor_masses_mist_cf(new_sample)
        elif precursor_noise_mode != "none":
            raise ValueError(f"Unknown precursor_noise_mode: {precursor_noise_mode!r}")

        new_sample = Augmentation.random_peak_dropout(new_sample)
        # normalize
        # new_sample = Augmentation.normalize_intensities(new_sample)

        new_sample = Augmentation.masking_metadata(new_sample)
        return new_sample

    @staticmethod
    def normalize_max(data_sample):
        for sufix in ["_0", "_1"]:
            # put a threshold
            intensity_column = "intensity" + sufix
            mz_column = "mz" + sufix

            # normalization
            intensity = np.array(data_sample[intensity_column])
            mz = data_sample[mz_column]

            intensity = intensity / np.max(intensity, keepdims=True)

            # apply
            data_sample[intensity_column] = intensity
            data_sample[mz_column] = mz

        return data_sample

    @staticmethod
    def peak_augmentation_max_peaks(data_sample, p_augmentation=0.50, max_peaks=100):
        # first normalize to maximum

        ## half of the time select maximum 20, the other half something between 5 and the maximum number of peaks

        if random.random() < p_augmentation:
            if random.random() < 0.5:
                max_augmented_peaks = 20
            else:
                max_augmented_peaks = int(max(20, random.random() * max_peaks))

            for sufix in ["_0", "_1"]:
                # put a threshold
                intensity_column = "intensity" + sufix
                mz_column = "mz" + sufix
                intensity = data_sample[intensity_column]
                mz = data_sample[mz_column]

                # order intensities by amplitude
                intensity_ordered_indexes = np.argsort(intensity)[
                    ::-1
                ]  # flip the order to have the max at the beginning
                indexes_to_be_erased = intensity_ordered_indexes[
                    max_augmented_peaks:-1
                ]

                intensity = Augmentation.zero_and_pack(
                    intensity, indexes_to_be_erased
                )
                mz = Augmentation.zero_and_pack(mz, indexes_to_be_erased)

                # apply
                data_sample[intensity_column] = intensity
                data_sample[mz_column] = mz
            return data_sample
        else:
            return data_sample

    @staticmethod
    def peak_augmentation_removal_noise(
        data_sample, max_percentage=0.01, p_augmentation=0.5
    ):
        if random.random() < p_augmentation:
            # first normalize to maximum
            for sufix in ["_0", "_1"]:
                # put a threshold
                intensity_column = "intensity" + sufix
                mz_column = "mz" + sufix
                intensity = data_sample[intensity_column]
                mz = data_sample[mz_column]

                # remove noise peaks
                max_amplitude = random.random() * max_percentage

                # indexes_to_modify=intensity < max_amplitude
                indexes_to_be_erased = intensity < (
                    max_amplitude * np.max(intensity, keepdims=True)
                )
                intensity = Augmentation.zero_and_pack(
                    intensity, indexes_to_be_erased
                )
                mz = Augmentation.zero_and_pack(mz, indexes_to_be_erased)

                # apply
                data_sample[intensity_column] = intensity
                data_sample[mz_column] = mz
            # put a threshold based on peak amplitude

            return data_sample
        else:
            return data_sample

    @staticmethod
    def normalize_intensities(data_sample, intensity_labels=None):
        if intensity_labels is None:
            intensity_labels = ["intensity_0", "intensity_1"]
        # sqrt root
        for intensity_column in intensity_labels:
            intensity = data_sample[intensity_column]
            intensity = np.sqrt(intensity)
            intensity = intensity / np.sqrt(np.sum(intensity**2, keepdims=True))
            data_sample[intensity_column] = intensity
        return data_sample

    @staticmethod
    def add_false_precursor_masses_positives(
        sample, max_noise=0.01, p_augmentation=0.2
    ):
        """
        Perturb each side's precursor mass independently by up to +/-max_noise
        (fraction of that side's own precursor m/z), simulating instrument
        measurement variability (legacy/Sebastian's noise step).
        """

        if random.random() < p_augmentation:
            added_noise_factor_0 = random.uniform(-max_noise, max_noise)
            added_noise_factor_1 = random.uniform(-max_noise, max_noise)

            pmz_0 = sample["precursor_mass_0"]
            pmz_1 = sample["precursor_mass_1"]
            sample["precursor_mass_0"] = pmz_0 + added_noise_factor_0 * pmz_0
            sample["precursor_mass_1"] = pmz_1 + added_noise_factor_1 * pmz_1
            return sample
        else:
            return sample

    @staticmethod
    def resample_precursor_masses_mist_cf(sample, p_augmentation=0.2):
        """
        MIST-CF/BUDDY-style precursor m/z resampling: replace each side's
        precursor mass with a Gaussian-perturbed value, std = instrument-
        specific ppm tolerance / 5. Falls back to the "unknown" instrument
        tier when a side's instrument type isn't present in `sample`.
        """
        if random.random() < p_augmentation:
            for suffix in ("_0", "_1"):
                pmz = sample["precursor_mass" + suffix]
                instrument = sample.get("instrument" + suffix, "unknown")
                sample["precursor_mass" + suffix] = resample_precursor_mz(
                    pmz, instrument, np.random
                )
            return sample
        else:
            return sample

    @staticmethod
    def add_false_precursor_masses_negatives(
        sample, max_noise=1.0, p_augmentation=0.10
    ):
        """
        create a pair where the precursor masses are almost the same
        """

        if random.random() < p_augmentation:
            added_noise_factor_0 = random.uniform(-max_noise, max_noise)
            added_noise_factor_1 = random.uniform(-max_noise, max_noise)

            sample["precursor_mass_0"] = (
                sample["precursor_mass_0"]
                + added_noise_factor_0 * (sample["precursor_mass_0"])
            )
            sample["precursor_mass_1"] = (
                sample["precursor_mass_1"]
                + added_noise_factor_1 * (sample["precursor_mass_1"])
            )
            return sample
        else:
            return sample

    @staticmethod
    def random_peak_dropout(data_sample, dropout_rate=0.10, p_augmentation=0.5):
        """
        Randomly zero out a percentage of peaks to simulate partial data loss.
        """
        if random.random() < p_augmentation:
            for suffix in ["_0", "_1"]:
                int_key = "intensity" + suffix
                mz_key = "mz" + suffix
                intensity_array = data_sample[int_key]
                mz_array = data_sample[mz_key]

                n_peaks = len(intensity_array)
                n_drop = int(n_peaks * dropout_rate)
                # choose random peaks to drop
                drop_indices = np.array(random.sample(range(n_peaks), n_drop))
                intensity_array = Augmentation.zero_and_pack(
                    intensity_array, drop_indices
                )
                mz_array = Augmentation.zero_and_pack(mz_array, drop_indices)

                data_sample[int_key] = intensity_array
                data_sample[mz_key] = mz_array
        return data_sample

    @staticmethod
    def masking_metadata(data_sample, p_aug=0.5):
        """
        Randomly mask metadata fields for augmentation.
        """
        keys_found = list(data_sample.keys())
        if random.random() < p_aug:
            if random.random() < 0.5:
                if "ionmode_0" in keys_found:
                    data_sample["ionmode_0"] = 0 * data_sample["ionmode_0"]
                    data_sample["ionmode_1"] = 0 * data_sample["ionmode_1"]
                    data_sample["precursor_charge_0"] = (
                        0 * data_sample["precursor_charge_0"]
                    )
                    data_sample["precursor_charge_1"] = (
                        0 * data_sample["precursor_charge_1"]
                    )
            if random.random() < 0.5:
                if "adduct_0" in keys_found:
                    data_sample["adduct_0"] = 0 * data_sample["adduct_0"]
                    data_sample["adduct_1"] = 0 * data_sample["adduct_1"]
            if random.random() < 0.5:
                if "ce_0" in keys_found:
                    data_sample["ce_0"] = 0 * data_sample["ce_0"]
                    data_sample["ce_1"] = 0 * data_sample["ce_1"]
            if random.random() < 0.5:
                if "ion_activation_0" in keys_found:
                    data_sample["ion_activation_0"] = (
                        0 * data_sample["ion_activation_0"]
                    )
                    data_sample["ion_activation_1"] = (
                        0 * data_sample["ion_activation_1"]
                    )

        else:
            data_sample["ionmode_0"] = 0 * data_sample["ionmode_0"]
            data_sample["ionmode_1"] = 0 * data_sample["ionmode_1"]
            data_sample["precursor_charge_0"] = (
                0 * data_sample["precursor_charge_0"]
            )
            data_sample["precursor_charge_1"] = (
                0 * data_sample["precursor_charge_1"]
            )
            data_sample["adduct_0"] = 0 * data_sample["adduct_0"]
            data_sample["adduct_1"] = 0 * data_sample["adduct_1"]
            data_sample["ce_0"] = 0 * data_sample["ce_0"]
            data_sample["ce_1"] = 0 * data_sample["ce_1"]
            data_sample["ion_activation_0"] = (
                0 * data_sample["ion_activation_0"]
            )
            data_sample["ion_activation_1"] = (
                0 * data_sample["ion_activation_1"]
            )
        return data_sample
