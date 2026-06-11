import copy

import numpy as np
from metabo_depthcharge.spec.adducts import encode_adduct
from metabo_depthcharge.spec.metadata_parsers import encode_collision_energy

from simba.core.data.datasets.encoder_dataset import CustomDatasetEncoder
from simba.core.data.encoding import (
    ION_ACTIVATION,
    IONIZATION_METHODS,
    encode_ion_activation,
    encode_ionization_method,
)
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

    # Metadata fields (initialized with default values)
    ionmode = np.zeros((len(spectra), 1), dtype=np.float32)
    adduct = np.zeros(len(spectra), dtype=np.int64)
    ce = np.zeros((len(spectra), 1), dtype=np.float32)
    ia = np.zeros((len(spectra), len(ION_ACTIVATION)), dtype=np.int32)
    im = np.zeros((len(spectra), len(IONIZATION_METHODS)), dtype=np.int32)

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

        # Extract metadata if available
        # Ion mode
        if hasattr(spectrum, 'ionmode') and spectrum.ionmode is not None and spectrum.ionmode != "None":
            ionmode[i] = 1.0 if spectrum.ionmode.lower() == "positive" else -1.0
        else:
            ionmode[i] = 0.0

        # Adduct — categorical index using metabo-depthcharge vocabulary
        adduct_str = getattr(spectrum, "adduct", None) or ""
        adduct[i] = encode_adduct(adduct_str)

        # Collision Energy
        ce[i] = encode_collision_energy(getattr(spectrum, "ce", None))

        # Ion Activation
        if hasattr(spectrum, 'ion_activation') and spectrum.ion_activation is not None and spectrum.ion_activation != "None":
            ia[i] = encode_ion_activation(spectrum.ion_activation)
        else:
            ia[i] = np.zeros(len(ION_ACTIVATION), dtype=np.int32)

        # Ionization Method
        if hasattr(spectrum, 'ionization_method') and spectrum.ionization_method is not None and spectrum.ionization_method != "None":
            im[i] = encode_ionization_method(spectrum.ionization_method)
        else:
            im[i] = np.zeros(len(IONIZATION_METHODS), dtype=np.int32)

    # Normalize the intensity array
    intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))

    spectrum_data = {
        "mz": mz,
        "intensity": intensity,
        "precursor_mass": precursor_mass,
        "precursor_charge": precursor_charge,
        "ionmode": ionmode,
        "adduct": adduct,
        "ce": ce,
        "ion_activation": ia,
        "ion_method": im,
    }

    return CustomDatasetEncoder(spectrum_data)
