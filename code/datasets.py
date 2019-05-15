#!/usr/bin/env python

from torch.utils.data import Dataset
import torch
import numpy as np
import nibabel as nib

import os.path as op
from glob import glob


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
        self.shape = shape
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
        return np.sin(xyz)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.tensor(sample).view((1), *self.shape)
        return sample


class ImageDataset(Dataset):
    def __init__(self, dataset_directory, random_seed=42, mode="train",
                 train_size=0.8, test_size=0.1, validate_size=0.1):
        # Makes sure that train-test-validate split makes sense
        assert(train_size + test_size + validate_size == 1.0)
        # Makes sure that we're grabbing a valid subset of our dataset
        assert(mode in ["train", "test", "validate"])

        # Get all nifti files in directory
        files = glob(op.join(dataset_directory, "*", "*.nii*"),
                     recursive=True)

        # randomly permute files (according to seed)
        np.random.seed(random_seed)
        files = np.random.permutation(files)

        # grab first train_size if doing train, next if doing test, etc.
        N = len(files)
        assert(N > 0)

        subset = {"train": slice(0,
                                 int(np.floor(train_size*N))),
                  "test": slice(int(np.floor(train_size*N)),
                                int(np.floor((train_size+test_size)*N))),
                  "validate": slice(int(np.floor((train_size+test_size)*N)),
                                    N)}
        print(subset)
        relevant_files = files[subset[mode]]

        # print N samples in each group
        print("Total number of images: {0}".format(N))
        print("Train | Test | Validate: {0} | {1} | {2}".format(train_size,
                                                                test_size,
                                                                validate_size))
        print("Mode: {0}".format(mode))
        print("Number of loaded images: {0}".format(len(relevant_files)))

        # load file later to not kill memory
        self.data = relevant_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        sample = nib.load(fname).get_data().copy().astype(float)
        sample = np.nan_to_num(sample)
        sample = torch.tensor(sample).view((1), *sample.shape)
        return sample
