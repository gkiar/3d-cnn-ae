#!/usr/bin/env python

from torch.utils.data import Dataset
import torch
import numpy as np
import nibabel as nib


class SimulationDataset(Dataset):
    def __init__(self,
                 shape=(48, 56, 48),
                 n_samples=1000,
                 constants=(10, 3, 16)):
        """
        Args:
            shape (tuple): 3D shape of simulated data
            n_samples (int): number of samples to generate
            constants (tuple): 3-length tuple of constants to use in data gen
        """
        constants = constants[:3]  # Truncates constants to len 3 just in case
        self.data = [self._generate_sample(shape, *constants)
                     for _ in range(n_samples)]

    def _generate_sample(self, shape, c1, c2, c3):
        # Create incrementing point grids
        xx, yy, zz = np.meshgrid(np.arange(0, shape[0], 1, dtype='float64'),
                                 np.arange(0, shape[1], 1, dtype='float64'),
                                 np.arange(0, shape[2], 1, dtype='float64'))
        # Apply some slightly-random transform along each axis
        xx += np.random.randint(c1)/(c2 + c3*np.random.random())
        yy /= (np.random.randint(c2) + 1)/(c3 + c1*np.random.random())
        zz *= (np.random.randint(c3) + 1)/(c1 + c2*np.random.random())

        # Sum all the independent axis functions
        xyz = (xx+yy+zz)

        # Return the sin of these combined functions in the shape (1,X1,X2,X3)
        return torch.tensor(np.sin(xyz)).view((1), *shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ImageDataset(Dataset):
    def __init__(self, directory, random_seed=42, mode="train",
                 train_size=0.8, test_size=0.1, validate_size=0.1):
        assert(train_size + test_size + validate_size == 1.0)
        assert(mode in ["train", "test", "validate"])
        # get all files in directory
        # randomly permute files (according to seed)
        # grab first train_size if doing train, next if doing test, etc.
        # print N samples in each group
        # load files and convert to large tensor
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
