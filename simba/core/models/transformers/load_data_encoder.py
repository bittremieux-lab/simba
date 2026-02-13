from simba.core.data.datasets.encoder_dataset import CustomDatasetEncoder
from simba.core.models.transformers.load_data_base import LoadDataBase


class LoadDataEncoder(LoadDataBase):
    """
    load data for encoder from spectra
    """

    @staticmethod
    def from_spectrums_to_dataset(
        spectrums_input,
        max_num_peaks=100,
        training=False,
    ):
        dict_spectrum_data = LoadDataBase.load_spectrum_data(
            spectrums_input, max_num_peaks=max_num_peaks
        )

        return CustomDatasetEncoder(dict_spectrum_data)
