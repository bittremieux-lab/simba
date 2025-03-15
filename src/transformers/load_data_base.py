from src.transformers.CustomDatasetUnique import CustomDatasetUnique
import numpy as np
from src.preprocessor import Preprocessor
from tqdm import tqdm
from src.molecule_pairs_opt import MoleculePairsOpt
import copy

class LoadDataBase:



    def load_spectrum_data(spectrums_input, max_num_peaks=100,):
        ## Preprocess the data
        pp = Preprocessor()
        print("Preprocessing all the data ...")
        
        spectrums=[
                    copy.deepcopy(s) for s in spectrums_input
                ]


        spectrums = pp.preprocess_all_spectrums(spectrums, max_num_peaks=max_num_peaks)
        #spectrums = pp.preprocess_all_spectrums_variable_max_peaks(spectrums, max_num_peaks=max_num_peaks)

        print("Finished preprocessing ")

        ## Get the mz, intensity values and precursor data
        mz = np.zeros(
            (len(spectrums), max_num_peaks), dtype=np.float32
        )
        intensity = np.zeros(
            (len(spectrums), max_num_peaks), dtype=np.float32
        )
        precursor_mass = np.zeros(
            (len(spectrums), 1), dtype=np.float32
        )
        precursor_charge = np.zeros(
            (len(spectrums), 1), dtype=np.int32
        )

        print("loading data")
        for i, l in enumerate(spectrums):
            # check for maximum length
            length = len(l.mz) if len(l.mz) <= max_num_peaks else max_num_peaks

            # assign the values to the array
            mz[i, 0:length] = np.array(l.mz[0:length])
            intensity[i, 0:length] = np.array(l.intensity[0:length])

            precursor_mass[i] = l.precursor_mz
            precursor_charge[i] = l.precursor_charge

        print("Normalizing intensities")
        # Normalize the intensity array
        intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))


        return {'mz':mz,
                'intensity':intensity,
                'precursor_mass':precursor_mass,
                'precursor_charge': precursor_charge}
    