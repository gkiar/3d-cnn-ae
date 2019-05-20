#!/usr/bin/env python

from argparse import ArgumentParser

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import sys

from model import Autoencoder3D
from datasets import SimulationDataset, ImageDataset


def simulate(*args, **kwargs):
    data_shape = (48, 56, 48)
    training_samples = 900
    batch_size = 25
    training = SimulationDataset(shape=data_shape, n_samples=training_samples)
    training_loader = DataLoader(training, batch_size=batch_size)
    _trainer(training_loader)


def train(*args, **kwargs):
    batch_size = 16
    training = ImageDataset("/home/users/gkiar/ace_mount/ace_home/data/nv_filtered/",
                            mode="train", stratify=["map_type", "analysis_level"])
    training_loader = DataLoader(training, batch_size=batch_size)
    _trainer(training_loader)


def _trainer(dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize model, loss function and optimizer
    model = Autoencoder3D().to(device)
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Perform training
    num_epochs = 25
    outfname = './nv_params'  # simulation
    for epoch in range(num_epochs):
        for idx, data in enumerate(dataset):
            img = data
            img = img.type('torch.FloatTensor').to(device)
            # forward
            output = model(img)
            loss = distance(output, img)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not (idx % 100):
                print(idx, loss.data, flush=True)
        print('')
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,
                                                  num_epochs, loss.data))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '{0}_epoch_{1}'.format(outfname, epoch + 1))
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    torch.save(model, outfname)


def main(args=None):
    description = "Launcher for 3D CNN-AE network to be used on 4mm fMRI SPMs."
    parser = ArgumentParser(__file__, description=description)

    htext = ("simulate: Trains network on simuluated dataset. Intended to test "
             "successful construction and execution of model.\n"
             "train: Trains network on real dataset. Will save out weights "
             "incrementally and terminally.\n"
             "visualize: Creates a figure from the model. To be implemented.")
    subparsers = parser.add_subparsers(dest="mode",
                                       help=htext)

    parser_sim = subparsers.add_parser("simulate")
    parser_sim.set_defaults(func=simulate)

    parser_trn = subparsers.add_parser("train")
    parser_trn.set_defaults(func=train)

    inps = parser.parse_args(args) if args is not None else parser.parse_args()

    # If no args are provided, print help
    if len(sys.argv) < 2 and args is None:
        parser.print_help()
        sys.exit()
    else:
        inps.func(**vars(inps))
        return 0


if __name__ == "__main__":
    main()
