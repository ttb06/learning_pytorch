import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform = None):
        #data loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) #n_samples, 1
        self.n_sample = xy.shape[0]

        self.transform = transform


    def __getitem__(self, index):
        #indexing item of dts
        #dts[index]
        sample = self.x[index], self.y[index]
        if (self.transform):
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        #return len of dts
        return self.n_sample

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

dts = WineDataset()
dataloader = DataLoader(dataset=dts, batch_size=4, shuffle=True, num_workers=2)
 