
from src.load_data import LoadData
from src.preprocessor import Preprocessor
import copy 

class PreprocessingSimba:

    def load_spectra(mgf_file, config, min_peaks=6, ):
        #load
        from src.loader_saver import LoaderSaver
        loader_saver = LoaderSaver(
                    block_size=100,
                    pickle_nist_path='',
                    pickle_gnps_path=None,
                    pickle_janssen_path='',
                )
        all_spectrums = loader_saver.get_all_spectrums(
                        mgf_file,
                        100000000,
                        use_tqdm=True,
                        use_nist=False,
                        config=config,
                        use_janssen=False,
                    )
        #all_spectrums= LoadData.get_all_spectrums_mgf(
        #        file=mgf_file,
        #        config=config,
        #    )

        #preprocess
        all_spectrums_processed= [copy.deepcopy(s) for s in all_spectrums]

        pp=Preprocessor()
        ### remove extra peaks in janssen
        all_spectrums_processed = [pp.preprocess_spectrum(
                    s,
                    fragment_tol_mass=10,
                    fragment_tol_mode="ppm",
                    min_intensity=0.01,
                    max_num_peaks=1000,
                    scale_intensity=None,
                ) for s in all_spectrums_processed]

        # remove spectra that does not have at least min peaks
        filtered_spectra= [s_original for s_original, s_processed in zip(all_spectrums,all_spectrums_processed) if len(s_processed.mz)>=min_peaks]


        return filtered_spectra