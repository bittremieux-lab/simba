from torch.utils.data import Dataset

from simba.core.data.augmentation import Augmentation


class CustomDatasetEncoder(Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = list(self.data.keys())

    def __len__(self):
        return self.data[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        # key = self.keys[idx]
        # sample = self.data[key]
        # print(idx)
        samples = {k: self.data[k][idx] for k in self.keys}
        samples = Augmentation.normalize_intensities(
            samples, intensity_labels=["intensity"]
        )
        return samples
