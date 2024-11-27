import random
import copy


class Augmentation:

    @staticmethod
    def augment(data_sample):
        new_sample = copy.deepcopy(data_sample)
        #new_sample = Augmentation.inversion(new_sample)
        #new_sample = Augmentation.add_noise_to_precursor_mass(new_sample)

        new_sample = Augmentation.add_false_precursor_masses(new_sample)
        return new_sample

    @staticmethod
    def inversion(data_sample):
        # inversion

        new_sample = {}
        new_sample["mz_0"] = data_sample["mz_1"]
        new_sample["mz_1"] = data_sample["mz_0"]

        new_sample["intensity_0"] = data_sample["intensity_1"]
        new_sample["intensity_1"] = data_sample["intensity_0"]

        new_sample["precursor_mass_0"] = data_sample["precursor_mass_1"]
        new_sample["precursor_mass_1"] = data_sample["precursor_mass_0"]

        new_sample["precursor_charge_0"] = data_sample["precursor_charge_1"]
        new_sample["precursor_charge_1"] = data_sample["precursor_charge_0"]

        new_sample["similarity"] = data_sample["similarity"]
        return new_sample

    @staticmethod
    def add_noise_to_precursor_mass(sample, max_noise=1.0):

        added_noise_factor_0 = random.uniform(-max_noise, max_noise)
        added_noise_factor_1 = random.uniform(-max_noise, max_noise)
        sample["precursor_mass_0"] = sample[
            "precursor_mass_0"
        ] + added_noise_factor_0 * (sample["precursor_mass_0"])
        sample["precursor_mass_1"] = sample[
            "precursor_mass_1"
        ] + added_noise_factor_1 * (sample["precursor_mass_1"])
        return sample

    @staticmethod
    def add_false_precursor_masses(sample, max_noise=0.01):
        '''
        create a pair where the precursor masses are almost the same
        '''
        added_noise_factor_0 = random.uniform(-max_noise, max_noise)
        added_noise_factor_1 = random.uniform(-max_noise, max_noise)

        pmz= sample["precursor_mass_0"].copy()
        sample["precursor_mass_0"] = pmz + added_noise_factor_0 * (pmz)
        sample["precursor_mass_1"] = pmz + added_noise_factor_1 * (pmz)

        return sample