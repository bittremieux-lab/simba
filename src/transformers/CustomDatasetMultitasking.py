import torch
from torch.utils.data import Dataset
import random
from src.transformers.augmentation import Augmentation
import numpy as np
from tqdm import tqdm


class CustomDatasetMultitasking(Dataset):
    def __init__(
        self,
        your_dict,
        training=False,
        prob_aug=1.0,
        #prob_aug=0.2,
        mz=None,
        intensity=None,
        precursor_mass=None,
        precursor_charge=None,
        df_smiles=None,
        use_fingerprints=False,
        fingerprint_0=None,
        max_num_peaks=None,
    ):
        self.data = your_dict
        self.keys = list(your_dict.keys())
        self.training = training
        self.prob_aug = prob_aug

        self.mz = mz
        self.intensity = intensity
        self.precursor_mass = precursor_mass
        self.precursor_charge = precursor_charge
        self.df_smiles = df_smiles  ### df with rows smiles, indexes
        self.use_fingerprints=use_fingerprints
        if self.use_fingerprints:
            self.fingerprint_0=fingerprint_0 
        self.max_num_peaks=max_num_peaks

    def __len__(self):
        return len(self.data[self.keys[0]])
        # return len(self.keys)

    def get_original_dictionary(self, max_num_peaks=100):
        """
        get a dictionary containing the spectrums mapped
        """
        len_data = self.data[self.keys[0]].shape[0]
        ## Get the mz, intensity values and precursor data

        dictionary = {}
        dictionary["mz_0"] = np.zeros((len_data, max_num_peaks), dtype=np.float32)
        dictionary["intensity_0"] = np.zeros(
            (len_data, max_num_peaks), dtype=np.float32
        )
        dictionary["mz_1"] = np.zeros((len_data, max_num_peaks), dtype=np.float32)
        dictionary["intensity_1"] = np.zeros(
            (len_data, max_num_peaks), dtype=np.float32
        )
        dictionary["similarity"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["similarity2"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["precursor_mass_0"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["precursor_charge_0"] = np.zeros((len_data, 1), dtype=np.int32)
        dictionary["precursor_mass_1"] = np.zeros((len_data, 1), dtype=np.float32)
        dictionary["precursor_charge_1"] = np.zeros((len_data, 1), dtype=np.int32)

        if self.use_fingerprints:
            print('Defining fingerprints ...')
            dictionary["fingerprint_0"] = np.zeros((len_data, 2048), dtype=np.int32)

        for idx in tqdm((range(0, len_data))):
            sample_unique = {k: self.data[k][idx] for k in self.keys}

            indexes_unique_0 = sample_unique["index_unique_0"]
            indexes_unique_1 = sample_unique["index_unique_1"]

            indexes_original_0 = self.df_smiles.loc[int(indexes_unique_0), "indexes"][0]

            indexes_original_1 = self.df_smiles.loc[int(indexes_unique_1), "indexes"][0]

            dictionary["mz_0"][idx] = self.mz[indexes_original_0].astype(np.float32)
            dictionary["intensity_0"][idx] = self.intensity[indexes_original_0].astype(
                np.float32
            )

            dictionary["mz_1"][idx] = self.mz[indexes_original_1].astype(np.float32)
            dictionary["intensity_1"][idx] = self.intensity[indexes_original_1].astype(
                np.float32
            )
            dictionary["precursor_mass_0"][idx] = self.precursor_mass[
                indexes_original_0
            ].astype(np.float32)
            dictionary["precursor_mass_1"][idx] = self.precursor_mass[
                indexes_original_1
            ].astype(np.float32)
            dictionary["precursor_charge_0"][idx] = self.precursor_charge[
                indexes_original_0
            ].astype(np.float32)
            dictionary["precursor_charge_1"][idx] = self.precursor_charge[
                indexes_original_1
            ].astype(np.float32)
            dictionary["similarity"][idx] = sample_unique["similarity"].astype(
                np.float32
            )
            dictionary["similarity2"][idx] = sample_unique["similarity2"].astype(
                np.float32
            )

            if self.use_fingerprints:
                dictionary["fingerprint_0"][idx] =self.fingeprint_0[indexes_original_0].astype(np.float32)

        return dictionary

    def __getitem__(self, idx):
        # key = self.keys[idx]
        # sample = self.data[key]
        # print(idx)
        sample_unique = {k: self.data[k][idx] for k in self.keys}

        # indexes_unique_0 = list(sample_unique['index_unique_0'])
        # indexes_unique_1 = list(sample_unique['index_unique_1'])

        indexes_unique_0 = sample_unique["index_unique_0"]
        indexes_unique_1 = sample_unique["index_unique_1"]

        # for each unique value 0 sample from the distribution
        # indexes_original_0 = [random.choice(self.df_smiles.loc[int(index),'indexes']) for index in indexes_unique_0]
        # indexes_original_1 = [random.choice(self.df_smiles.loc[int(index),'indexes']) for index in indexes_unique_1]

        if self.training:
            # select random samples
            indexes_original_0 = random.choice(
                self.df_smiles.loc[int(indexes_unique_0[0]), "indexes"]
            )
            indexes_original_1 = random.choice(
                self.df_smiles.loc[int(indexes_unique_1[0]), "indexes"]
            )
        else:
            # select the first index
            indexes_original_0 = self.df_smiles.loc[
                int(indexes_unique_0[0]), "indexes"
            ][0]
            # select the last index
            indexes_original_1 = self.df_smiles.loc[
                int(indexes_unique_1[0]), "indexes"
            ][-1]

        ## now get an original spectra based on indexes
        sample = {}

        sample["mz_0"] = self.mz[indexes_original_0].astype(np.float32)
        sample["intensity_0"] = self.intensity[indexes_original_0].astype(np.float32)

        sample["mz_1"] = self.mz[indexes_original_1].astype(np.float32)
        sample["intensity_1"] = self.intensity[indexes_original_1].astype(np.float32)
        sample["precursor_mass_0"] = self.precursor_mass[indexes_original_0].astype(
            np.float32
        )
        sample["precursor_mass_1"] = self.precursor_mass[indexes_original_1].astype(
            np.float32
        )
        sample["precursor_charge_0"] = self.precursor_charge[indexes_original_0].astype(
            np.float32
        )
        sample["precursor_charge_1"] = self.precursor_charge[indexes_original_1].astype(
            np.float32
        )
        sample["similarity"] = sample_unique["similarity"].astype(np.float32)
        sample["similarity2"] = sample_unique["similarity2"].astype(np.float32)

        if self.use_fingerprints:
            sample["fingerprint_0"] = self.fingerprint_0[int(indexes_unique_0[0])].astype(
            np.float32
        )

        # print(sample["mz_0"]).shape
        # print(sample["intensity_0"].shape)
        # print(sample["precursor_charge_0"].shape)
        # print(sample["precursor_mass_0"].shape)

        # Convert your sample to PyTorch tensors if needed
        # e.g., use torch.tensor(sample) if sample is a numpy array

        if self.training:
            if random.random() < self.prob_aug:
                # augmentation
                sample = Augmentation.augment(sample, max_num_peaks=self.max_num_peaks)

        # normalize
        sample = Augmentation.normalize_intensities(sample)
        return sample
