from simba.transformers.CustomDatasetUnique import CustomDatasetUnique
import numpy as np
from simba.preprocessor import Preprocessor
from tqdm import tqdm
from simba.molecule_pairs_opt import MoleculePairsOpt
import copy


class LoadDataClass:
    """
    using unique identifiers
    """

    @staticmethod
    def from_molecule_pairs_to_dataset(
        molecule_pairs_input,
        max_num_peaks=100,
        training=False,  # shuffle the spectrum 0 and 1 for data augmentation
        N_classes=6,
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
        )

        ## Preprocess the data
        pp = Preprocessor()
        print("Preprocessing all the data ...")
        molecule_pairs.spectrums_original = pp.preprocess_all_spectrums(
            molecule_pairs.spectrums_original
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
        intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))

        print("Creating dictionaries")

        print("Adapt code for classification, assuming a 6 label problem")

        similarity_classification = np.round(
            (N_classes * molecule_pairs_input.indexes_tani[:, 2])
        ).astype(int)

        dictionary_data = {
            "index_unique_0": molecule_pairs_input.indexes_tani[:, 0].reshape(-1, 1),
            "index_unique_1": molecule_pairs_input.indexes_tani[:, 1].reshape(-1, 1),
            "similarity": similarity_classification,
            # "fingerprint": fingerprints,
        }

        return CustomDatasetUnique(
            dictionary_data,
            training=training,
            mz=mz,
            intensity=intensity,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            df_smiles=molecule_pairs_input.df_smiles,
        )
