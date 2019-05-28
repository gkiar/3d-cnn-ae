#!/usr/bin/env python

from torch.utils.data import Dataset
from glob import glob
import os.path as op
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import json


class SimulationDataset(Dataset):
    def __init__(self,
                 shape=(48, 56, 48),
                 n_samples=1000,
                 constants=(90, 4, 87)):
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
        # xx += np.random.randint(c1)/(c2 + c3*np.random.random())
        # yy /= (np.random.randint(c2) + 1)/(c3 + c1*np.random.random())
        # zz *= (np.random.randint(c3) + 1)/(c1 + c2*np.random.random())

        # FOR DEBUGGING: Make data very predictable
        xx += c1/(c2 + c3*np.random.random())
        yy /= (c2 + 1)/(c3 + c1*np.random.random())
        zz *= (c3 + 1)/(c1 + c2*np.random.random())

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
                 train_size=0.8, test_size=0.1, validate_size=0.1,
                 cache=True, stratify=None):
        # Makes sure that train-test-validate split makes sense
        assert(train_size + test_size + validate_size == 1.0)
        # Makes sure that we're grabbing a valid subset of our dataset
        assert(mode in ["train", "test", "validate"])

        # Get all nifti files in directory, or grab list if you've
        # already built it (since a big glob can be slow)
        print("... Getting file list")
        cached_filelist = op.join(op.expanduser('~'), '.nv_cnn_filelist.txt')
        if not op.isfile(cached_filelist) or not cache:
            files = sorted(glob(op.join(dataset_directory, "*", "*.nii*"),
                                recursive=True))
            with open(cached_filelist, 'w') as fhandle:
                fhandle.write("\n".join(f for f in files))
        else:
            print("... ... Using cached list!")
            with open(cached_filelist, 'r') as fhandle:
                files = sorted(fhandle.read().split("\n"))
        self.ids = {str(f.split('/')[-2]) : f for f in files}

        # Get associated metadata from either generic metadata file
        # or cached and pruned copy
        cached_metadata = op.join(op.expanduser('~'), ".nv_cnn_summary.json")
        print("... Loading map metadata")
        if not op.isfile(cached_metadata) or not cache:
            summary_file = op.join(dataset_directory, "summary.json")
            with open(summary_file, 'r') as fhandle:
                tmpsummary = json.load(fhandle)
                self.summary = {}
                for item in tmpsummary:
                    if str(item["id"]) in self.ids.keys():
                        l = str(item["id"])
                        self.summary[l] = item
                        self.summary[l]["localfile"] = self.ids[l]
                del tmpsummary
            with open(cached_metadata, 'w') as fhandle:
                fhandle.write(json.dumps(self.summary))
        else:
            print("... ... Using cached metadata!")
            with open(cached_metadata, 'r') as fhandle:
                self.summary = json.load(fhandle)

        # Reorganize samples according to either statified classes or
        # random permutation
        if isinstance(stratify, list):
            df = pd.DataFrame.from_dict(self.summary, orient="index")
            assert(all(s in df.columns for s in stratify))
            cols = list(set(df.columns) - set(stratify) - set(["localfile"]))
            df = df.drop(columns=cols)
            df = df.fillna(value="unknown")

            grps = df.groupby(stratify)
            weight_df = pd.DataFrame(columns=['weight'])
            for grp, rows in grps:
                for r in rows.index:
                    weight_df.loc[r] = np.log(len(rows.index)+1)
            df = pd.concat([df, weight_df], axis="columns", sort=True)

            resampled_df = df.sample(frac=1, replace=False, weights="weight",
                                     random_state=random_seed, axis=0)
            files = list(resampled_df["localfile"])
        else:
            np.random.seed(random_seed)
            files = list(np.random.permutation(files))

        # grab first train_size if doing train, next if doing test, etc.
        N = len(files)
        assert(N > 0)

        subset = {"train": slice(0,
                                 int(np.floor(train_size*N))),
                  "test": slice(int(np.floor(train_size*N)),
                                int(np.floor((train_size+test_size)*N))),
                  "validate": slice(int(np.floor((train_size+test_size)*N)),
                                    N)}
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
