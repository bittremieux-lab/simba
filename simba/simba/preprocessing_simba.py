from simba.load_data import LoadData
from simba.preprocessor import Preprocessor
import copy
from simba.loader_saver import LoaderSaver


class PreprocessingSimba:

    def load_spectra(
        file_name, config, min_peaks=6, n_samples=100000000, use_gnps_format=False
    ):
        # load
        print(file_name)
        if file_name.endswith(".mgf"):
            print("File name ends with mgf")
            loader_saver = LoaderSaver(
                block_size=100,
                pickle_nist_path=None,
                pickle_gnps_path=None,
                pickle_janssen_path=None,
            )
            all_spectrums = loader_saver.get_all_spectrums(
                file_name,
                n_samples,
                use_tqdm=True,
                use_nist=False,
                config=config,
                use_janssen=not (use_gnps_format),
            )
        elif file_name.endswith(".pkl"):
            all_spectrums = LoadData.get_all_spectrums_casmi(
                file_name,
                config=config,
            )
        else:
            print("Error: unrecognized file extension")
        # preprocess
        all_spectrums_processed = [copy.deepcopy(s) for s in all_spectrums]

        pp = Preprocessor()
        ### remove extra peaks in janssen
        all_spectrums_processed = [
            pp.preprocess_spectrum(
                s,
                fragment_tol_mass=10,
                fragment_tol_mode="ppm",
                min_intensity=0.01,
                max_num_peaks=1000,
                scale_intensity=None,
            )
            for s in all_spectrums_processed
        ]

        # remove spectra that does not have at least min peaks
        filtered_spectra = [
            s_original
            for s_original, s_processed in zip(all_spectrums, all_spectrums_processed)
            if len(s_processed.mz) >= min_peaks
        ]

        return filtered_spectra
