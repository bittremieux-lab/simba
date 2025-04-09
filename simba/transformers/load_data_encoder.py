from simba.transformers.CustomDatasetUnique import CustomDatasetUnique
import numpy as np
from simba.preprocessor import Preprocessor
from tqdm import tqdm
from simba.molecule_pairs_opt import MoleculePairsOpt
import copy
from simba.transformers.load_data_base import LoadDataBase
from simba.transformers.CustomDatasetEncoder import CustomDatasetEncoder


class LoadDataEncoder(LoadDataBase):
    """
    load data for encoder from spectra
    """

    def from_spectrums_to_dataset(
        spectrums_input,
        max_num_peaks=100,
        training=False,
    ):

        dict_spectrum_data = LoadDataBase.load_spectrum_data(
            spectrums_input, max_num_peaks=max_num_peaks
        )

        return CustomDatasetEncoder(dict_spectrum_data)
