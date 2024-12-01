from src.transformers.CustomDatasetMultitasking import CustomDatasetMultitasking
import numpy as np
from src.preprocessor import Preprocessor
from tqdm import tqdm
from src.molecule_pairs_opt import MoleculePairsOpt
import copy
from src.ordinal_classification.ordinal_classification import OrdinalClassification
from src.tanimoto import Tanimoto

class LoadDataMultitasking:
    """
    using unique identifiers
    """

    @staticmethod
    def from_molecule_pairs_to_dataset(
        molecule_pairs_input,
        max_num_peaks=100,
        training=False,  # shuffle the spectrum 0 and 1 for data augmentation
        N_classes=6,
        use_fingerprints=False,
    ):
        """
        preprocess the spectra and convert it for being used in Pytorch
        """
        # copy spectrums to avoid overwriting
        molecule_pairs = MoleculePairsOpt(
            spectrums_original=[
                copy.copy(s) for s in molecule_pairs_input.spectrums_original
            ],
            spectrums_unique=molecule_pairs_input.spectrums,
            df_smiles=molecule_pairs_input.df_smiles,
            indexes_tani_unique=molecule_pairs_input.indexes_tani,
            tanimotos=molecule_pairs_input.tanimotos,
        )

        ## Preprocess the data
        pp = Preprocessor()
        print("Preprocessing all the data ...")
        molecule_pairs.spectrums_original = pp.preprocess_all_spectrums(
            molecule_pairs.spectrums_original,
            training=training,
        )

        print("Finished preprocessing ")

        ## Get the mz, intensity values and precursor data
        mz = np.zeros(
            (len(molecule_pairs.spectrums_original), max_num_peaks), dtype=np.float32
        )
        intensity = np.zeros(
            (len(molecule_pairs.spectrums_original), max_num_peaks), dtype=np.float32
        )
        precursor_mass = np.zeros(
            (len(molecule_pairs.spectrums_original), 1), dtype=np.float32
        )
        precursor_charge = np.zeros(
            (len(molecule_pairs.spectrums_original), 1), dtype=np.int32
        )

        print("loading data")
        for i, l in enumerate(molecule_pairs.spectrums_original):
            # check for maximum length
            length = len(l.mz) if len(l.mz) <= max_num_peaks else max_num_peaks

            # assign the values to the array
            mz[i, 0:length] = np.array(l.mz[0:length])
            intensity[i, 0:length] = np.array(l.intensity[0:length])

            precursor_mass[i] = l.precursor_mz
            precursor_charge[i] = l.precursor_charge

        print("Normalizing intensities")
        # Normalize the intensity array
        #intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))


        ## Adjust similarity towards a N classification problem
        similarity = OrdinalClassification.from_float_to_class(molecule_pairs_input.indexes_tani[:, 2].reshape(-1, 1), N_classes=N_classes)
        #similarity= molecule_pairs_input.indexes_tani[:, 2].reshape(-1,1)

        similarity2= molecule_pairs.tanimotos.reshape(-1,1)

        if use_fingerprints:
            print('Computing molecular fingerprints')
            fingerprint_0= np.array([np.array(Tanimoto.compute_fingerprint(s.params['smiles'])) for s in molecule_pairs_input.spectrums])
        else:
            fingerprint_0= np.array([0 for m in molecule_pairs_input.spectrums])

        print("Creating dictionaries")
        dictionary_data = {
            "index_unique_0": molecule_pairs_input.indexes_tani[:, 0].reshape(-1, 1),
            "index_unique_1": molecule_pairs_input.indexes_tani[:, 1].reshape(-1, 1),
            "similarity": similarity,
            "similarity2" : similarity2,
            # "fingerprint_0": fingerprint_0,
        }

        return CustomDatasetMultitasking(
            dictionary_data,
            training=training,
            mz=mz,
            intensity=intensity,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            df_smiles=molecule_pairs_input.df_smiles,
            use_fingerprints=use_fingerprints,
            fingerprint_0=fingerprint_0,
        )
