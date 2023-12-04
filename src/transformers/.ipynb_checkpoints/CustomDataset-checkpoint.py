import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, your_dict):
        self.data = your_dict
        self.keys = list(your_dict.keys())

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
        
        return sample