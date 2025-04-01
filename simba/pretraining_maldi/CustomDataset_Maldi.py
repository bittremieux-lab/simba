import torch
from torch.utils.data import Dataset
import random
from simba.pretraining_maldi.self_supervision import SelfSupervision
import random


class CustomDatasetMaldi(Dataset):
    def __init__(self, your_dict, training=False):
        self.data = your_dict
        self.keys = list(your_dict.keys())
        self.training = training

    def __len__(self):
        return len(self.data[self.keys[0]])
        # return len(self.keys)

    def __getitem__(self, idx):
        # key = self.keys[idx]
        # sample = self.data[key]
        # print(idx)
        ## get a random index
        sample = {k: self.data[k][idx] for k in self.keys}

        # Convert your sample to PyTorch tensors if needed
        # e.g., use torch.tensor(sample) if sample is a numpy array

        # select peaks
        sample = SelfSupervision.modify_peaks(sample, self.data)
        return sample

    @staticmethod
    def random_integer_excluding(N, m):
        # Generate a random integer in the range [0, N - 1]
        rand_int = random.randint(0, N - 1)

        # Adjust the random integer if it equals m
        if rand_int == m:
            rand_int += 1

        return rand_int
