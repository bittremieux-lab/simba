import torch
from torch.utils.data import Dataset
import random
from src.transformers.augmentation import Augmentation
class CustomDataset(Dataset):
    def __init__(self, your_dict, training=False, prob_aug=0.5):
        self.data = your_dict
        self.keys = list(your_dict.keys())
        self.training= training
        self.prob_aug = prob_aug 

    def __len__(self):
        return len(self.data[self.keys[0]])
        #return len(self.keys)

    def __getitem__(self, idx):
        #key = self.keys[idx]
        #sample = self.data[key]
        #print(idx)
        sample = {k:self.data[k][idx] for k in self.keys}
        # Convert your sample to PyTorch tensors if needed
        # e.g., use torch.tensor(sample) if sample is a numpy array
        
        if self.training:
           if random.random() < self.prob_aug:
                # augmentation
                sample = Augmentation.augment(sample)
        return sample