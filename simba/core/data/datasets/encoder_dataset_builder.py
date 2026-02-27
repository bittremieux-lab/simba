import copy

import numpy as np

from simba.core.data.datasets.encoder_dataset import CustomDatasetEncoder
from simba.core.data.preprocessor import Preprocessor


def prepare_encoder_dataset(spectra, max_num_peaks=100):
    """
    Preprocess spectra and create encoder dataset.

    Args:
        spectra: Input spectra to preprocess
        max_num_peaks: Maximum number of peaks to keep per spectrum

    Returns:
        CustomDatasetEncoder ready for training/inference
    """
    # Preprocess the data
    pp = Preprocessor()
    spectra = [copy.deepcopy(s) for s in spectra]
    spectra = pp.preprocess_all_spectra(spectra, max_num_peaks=max_num_peaks)

    # Get the mz, intensity values and precursor data
    mz = np.zeros((len(spectra), max_num_peaks), dtype=np.float32)
    intensity = np.zeros((len(spectra), max_num_peaks), dtype=np.float32)
    precursor_mass = np.zeros((len(spectra), 1), dtype=np.float32)
    precursor_charge = np.zeros((len(spectra), 1), dtype=np.int32)

    for i, spectrum in enumerate(spectra):
        # check for maximum length
        length = (
            len(spectrum.mz) if len(spectrum.mz) <= max_num_peaks else max_num_peaks
        )

        # assign the values to the array
        mz[i, 0:length] = np.array(spectrum.mz[0:length])
        intensity[i, 0:length] = np.array(spectrum.intensity[0:length])

        precursor_mass[i] = spectrum.precursor_mz
        precursor_charge[i] = spectrum.precursor_charge

    # Normalize the intensity array
    intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))

    spectrum_data = {
        "mz": mz,
        "intensity": intensity,
        "precursor_mass": precursor_mass,
        "precursor_charge": precursor_charge,
    }

    return CustomDatasetEncoder(spectrum_data)
