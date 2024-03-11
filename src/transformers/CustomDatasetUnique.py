import torch
from torch.utils.data import Dataset
import random
from src.transformers.augmentation import Augmentation
import numpy as np


class CustomDatasetUnique(Dataset):
    def __init__(self, your_dict, training=False, prob_aug=0.1,mz=None,
                                   intensity=None,
                                   precursor_mass=None,
                                   precursor_charge=None,
                                   df_smiles=None):
        self.data = your_dict
        self.keys = list(your_dict.keys())
        self.training = training
        self.prob_aug = prob_aug

        self.mz=mz
        self.intensity=intensity
        self.precursor_mass=precursor_mass
        self.precursor_charge=precursor_charge
        self.df_smiles= df_smiles ### df with rows smiles, indexes 

    def __len__(self):
        return len(self.data[self.keys[0]])
        # return len(self.keys)

    def __getitem__(self, idx):
        # key = self.keys[idx]
        # sample = self.data[key]
        # print(idx)
        sample_unique = {k: self.data[k][idx] for k in self.keys}
    
        #indexes_unique_0 = list(sample_unique['index_unique_0'])
        #indexes_unique_1 = list(sample_unique['index_unique_1'])

        indexes_unique_0 = sample_unique['index_unique_0']
        indexes_unique_1 = sample_unique['index_unique_1']

        # for each unique value 0 sample from the distribution
        #indexes_original_0 = [random.choice(self.df_smiles.loc[int(index),'indexes']) for index in indexes_unique_0]
        #indexes_original_1 = [random.choice(self.df_smiles.loc[int(index),'indexes']) for index in indexes_unique_1]

        indexes_original_0 = random.choice(self.df_smiles.loc[int(indexes_unique_0[0]),'indexes']) 
        indexes_original_1 = random.choice(self.df_smiles.loc[int(indexes_unique_1[0]),'indexes']) 

        ## now get an original spectra based on indexes
        sample={}

        
        sample['mz_0']= self.mz[indexes_original_0].astype(np.float32)
        sample["intensity_0"]= self.intensity[indexes_original_0].astype(np.float32)

        sample['mz_1']= self.mz[indexes_original_1].astype(np.float32)
        sample["intensity_1"]= self.intensity[indexes_original_1].astype(np.float32)
        sample["precursor_mass_0"]= self.precursor_mass[indexes_original_0].astype(np.float32)
        sample["precursor_mass_1"]= self.precursor_mass[indexes_original_1].astype(np.float32)
        sample["precursor_charge_0"]= self.precursor_charge[indexes_original_0].astype(np.float32)
        sample["precursor_charge_1"]= self.precursor_charge[indexes_original_1].astype(np.float32)
        sample["similarity"]= sample_unique['similarity'].astype(np.float32)

        #print(sample["mz_0"]).shape
        #print(sample["intensity_0"].shape)
        #print(sample["precursor_charge_0"].shape)
        #print(sample["precursor_mass_0"].shape)

        # Convert your sample to PyTorch tensors if needed
        # e.g., use torch.tensor(sample) if sample is a numpy array

        if self.training:
            if random.random() < self.prob_aug:
                # augmentation
                sample = Augmentation.augment(sample)
      
        return sample
