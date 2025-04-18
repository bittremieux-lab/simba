import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from simba.config import Config
from simba.spectrum_ext import SpectrumExt
from simba.preprocessing_utils import PreprocessingUtils
import copy
import random


class Preprocessor:

    def __init__(self, bin_width=1, min_mz=10, max_mz=1400):
        # Define the parameters for binning
        self.bin_width = bin_width  # Adjust as needed to control the bin size
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.num_bins = int((max_mz - min_mz) / bin_width) + 1

    def preprocess_all_spectrums(
        self,
        spectrums,
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=100,
        # max_num_peaks=40,
        scale_intensity=None,
        # scale_intensity="root",
        training=False,
        random_seed=42,
    ):
        random.seed(random_seed)
        for i, spectrum in tqdm(enumerate(spectrums)):
            # try:
            if training:
                min_intensity = 0.00
            else:
                min_intensity = 0.01

            spectrums[i] = self.preprocess_spectrum(
                spectrum,
                fragment_tol_mass=fragment_tol_mass,
                fragment_tol_mode=fragment_tol_mode,
                min_intensity=min_intensity,
                max_num_peaks=max_num_peaks,
                scale_intensity=scale_intensity,
            )
        # except:
        #    print('Error preprocessing spectrum')

        # preprocess np vectors
        # all_spectrums= self.process_all_spectrum_vectors(spectrums)
        return spectrums

    def preprocess_all_spectrums_variable_max_peaks(
        self,
        spectrums,
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=100,
        # max_num_peaks=40,
        scale_intensity=None,
        # scale_intensity="root",
        training=False,
        random_seed=42,
    ):
        random.seed(random_seed)
        for i, spectrum in tqdm(enumerate(spectrums)):
            # try:
            if training:
                min_intensity = 0.00
            else:
                min_intensity = 0.01

            spectrums[i] = self.preprocess_spectrum(
                spectrum,
                fragment_tol_mass=fragment_tol_mass,
                fragment_tol_mode=fragment_tol_mode,
                min_intensity=min_intensity,
                max_num_peaks=max_num_peaks,
                scale_intensity=scale_intensity,
            )

            length_mz = len(spectrums[i].mz)
            if length_mz > 20:
                spectrums[i] = self.preprocess_spectrum(
                    spectrum,
                    fragment_tol_mass=fragment_tol_mass,
                    fragment_tol_mode=fragment_tol_mode,
                    min_intensity=min_intensity,
                    max_num_peaks=max(20, int(length_mz / (2.5))),
                    scale_intensity=scale_intensity,
                )
        # except:
        #    print('Error preprocessing spectrum')

        # preprocess np vectors
        # all_spectrums= self.process_all_spectrum_vectors(spectrums)
        return spectrums

    def preprocess_spectrum(
        self,
        spectrum,
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=100,
        # max_num_peaks=40,
        scale_intensity=None,
        # scale_intensity="root",
    ):

        # Process the spectrum.
        return (
            spectrum.remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
            # .set_mz_range(min_mz=self.min_mz, max_mz=self.max_mz)
            .filter_intensity(
                min_intensity=min_intensity, max_num_peaks=max_num_peaks
            ).scale_intensity(scale_intensity)
        )

    def return_spectrum_vector(self, spectrum):
        # Initialize an empty numpy array for bin intensities
        binned_spectrum = np.zeros(self.num_bins, dtype=np.float64)

        # Iterate through the data and assign intensities to bins
        for mz, intensity in zip(spectrum.mz, spectrum.intensity):
            if (mz > self.min_mz) and (mz < self.max_mz):
                bin_index = int((mz - self.min_mz) / self.bin_width)
                if intensity > binned_spectrum[bin_index]:
                    binned_spectrum[bin_index] = intensity
        return binned_spectrum

    def process_all_spectrum_vectors(self, spectrums):
        """
        save spectrum vectors and apply preprocessing
        """
        for i, spectrum in tqdm(enumerate(spectrums)):
            spectrum_vector = self.return_vector_and_preprocess(spectrum)
            spectrum.set_spectrum_vector(spectrum_vector)
        return spectrums

    def return_vector_and_preprocess(self, spectrum):
        spectrum_vector = self.return_spectrum_vector(spectrum)
        spectrum_vector = self.preprocess_vector(spectrum_vector)
        return spectrum_vector

    def preprocess_vector(self, spectrum_vector, min_intensity=0.01):

        # scale values using the maximum
        maximum = np.max(spectrum_vector)
        if maximum != 0:
            spectrum_vector = spectrum_vector / maximum
        # remove small values
        spectrum_vector[spectrum_vector < min_intensity] = 0
        return spectrum_vector

    def get_all_binned_spectrums(self, spectrums):
        all_binned_spectrums = np.zeros(
            (len(spectrums), self.num_bins), dtype=np.float64
        )
        for i, spectrum in enumerate(spectrums):
            all_binned_spectrums[i] = spectrum.spectrum_vector
        return all_binned_spectrums

    def return_valid_spectra_n_peaks(self, spectra_original, min_peaks):
        """
        return valid spectra based on the minimum number of peaks
        """
        input_spectra = [copy.deepcopy(s) for s in spectra_original]
        preprocessed_spectra = self.preprocess_all_spectrums(input_spectra)
        valid_indexes = [
            i for i, s in enumerate(preprocessed_spectra) if len(s.mz) > min_peaks
        ]
        return [spectra_original[i] for i in valid_indexes]
