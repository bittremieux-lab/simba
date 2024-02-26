import torch
from torch.utils.data import Dataset
import random
from src.pretraining_maldi.self_supervision import SelfSupervision


class CustomDatasetMaldi(Dataset):
    def __init__(self, your_dict, training=False, prob_aug=0.15):
        self.data = your_dict
        self.keys = list(your_dict.keys())
        self.training = training
        self.prob_aug = prob_aug

    def __len__(self):
        return len(self.data[self.keys[0]])
        # return len(self.keys)

    def __getitem__(self, idx):
        # key = self.keys[idx]
        # sample = self.data[key]
        # print(idx)
        sample = {k: self.data[k][idx] for k in self.keys}
        # Convert your sample to PyTorch tensors if needed
        # e.g., use torch.tensor(sample) if sample is a numpy array

        # select peaks
        sample = SelfSupervision.modify_peaks(sample)
        return sample
