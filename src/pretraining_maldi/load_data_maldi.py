from src.pretraining_maldi.CustomDataset_Maldi import CustomDatasetMaldi
import numpy as np
from src.preprocessor import Preprocessor
from tqdm import tqdm
from src.molecular_pairs_set import MolecularPairsSet
import copy


class LoadDataMaldi:

    @staticmethod
    def from_spectra_to_dataset(
        spectrums_input,
        max_num_peaks=100,
        training=False,  # shuffle the spectrum 0 and 1 for data augmentation,
        min_intensity=0.00,
    ):
        """
        preprocess the spectra and convert it for being used in Pytorch
        """
        spectrums = [copy.copy(s) for s in spectrums_input]

        ## Preprocess the data
        pp = Preprocessor()
        print("Preprocessing all the data for MALDI...")

        spectrums = pp.preprocess_all_spectrums(spectrums, min_intensity=min_intensity)
        print("Finished preprocessing ")

        # basic features
        mz_0 = np.zeros((len(spectrums), max_num_peaks), dtype=np.float32)
        number_of_peaks = np.zeros((len(spectrums), 1), dtype=np.float32)
        intensity_0 = np.zeros((len(spectrums), max_num_peaks), dtype=np.float32)
        precursor_mass_0 = np.zeros((len(spectrums), 1), dtype=np.float32)
        precursor_charge_0 = np.zeros((len(spectrums), 1), dtype=np.int32)

        # for self supervising approach
        sampled_mz = np.zeros((len(spectrums), max_num_peaks), dtype=np.float32)
        sampled_intensity = np.zeros((len(spectrums), max_num_peaks), dtype=np.float32)
        flips = np.zeros((len(spectrums), max_num_peaks), dtype=np.int32)

        print("Starting the loading of the data ...")
        # fill arrays
        for i, s in enumerate(spectrums):
            # check for maximum length
            length_0 = len(s.mz) if len(s.mz) <= max_num_peaks else max_num_peaks
            # assign the values to the array
            mz_0[i, 0:length_0] = np.array(s.mz[0:length_0])
            intensity_0[i, 0:length_0] = np.array(s.intensity[0:length_0])

            number_of_peaks[i] = length_0
            precursor_mass_0[i] = s.precursor_mz
            precursor_charge_0[i] = s.precursor_charge

        print("Normalizing intensities")
        # Normalize the intensity array
        intensity_0 = intensity_0 / np.sqrt(
            np.sum(intensity_0**2, axis=1, keepdims=True)
        )

        print("Creating dictionaries")
        dictionary_data = {
            "mz_0": mz_0,
            "intensity_0": intensity_0,
            "precursor_mass_0": precursor_mass_0,
            "precursor_charge_0": precursor_charge_0,
            "number_peaks": number_of_peaks.astype(int),
            "sampled_mz": sampled_mz,
            "sampled_intensity": sampled_intensity,
            "flips": flips,
        }

        return CustomDatasetMaldi(dictionary_data, training=training)
