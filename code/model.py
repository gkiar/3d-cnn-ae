#!/usr/bin/env python

import torch.nn as nn


class Autoencoder3D(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Architecture adopted from:
        #  https://www.sciencedirect.com/science/article/pii/S1361841517301287
        self.relu = nn.ReLU(True)

        # Layer input image size: 1x48x56x48 ; Kernel size: 1
        # N channels in: 1 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x48x56x48 ; Pad: 0
        self.conv1 = nn.Conv3d(1, 32, kernel_size=1, pad=0)

        # Layer input image size: 32x48x56x48 ; Kernel size: 2
        # N channels in: 32 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x24x28x24 ; Pad: 0
        self.pool1 = nn.MaxPool3d(2, stride=1,, return_indices=True)

        # Layer input image size: 32x24x28x24 ; Kernel size: 5
        # N channels in: 32 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x24x28x24 ; Pad: 2
        self.conv2 = nn.Conv3d(32, 32, kernel_size=5, pad=2)

        # Layer input image size: 32x48x56x48 ; Kernel size: 2
        # N channels in: 32 ; N channels out: 32 ; Stride: 1
        # Layer output image size: 32x12x14x12 ; Pad: 0
        self.pool2 = nn.MaxPool3d(2, stride=1, return_indices=True)

        # Layer input image size: 32x12x14x12 ; Kernel size: 5
        # N channels in: 32 ; N channels out: 32 ;  Stride: 1
        # Layer output image size: 32x12x14x12 ; Pad: 2
        self.conv3 = nn.Conv3d(32, 32, kernel_size=4, pad=2)

        # Layer input image size: 32x12x14x12 ; Kernel size: 2
        # N channels_in: 32 ; N channels_out: 32 ;  Stride: 1
        # Layer output image size: 32x6x7x6
        self.pool3 = nn.MaxPool3d(2, stride=1, return_indices=True)

        # Layer input image size: 32x6x7x6 ; Kernel size: 1
        # N channels in: 32 ; N channels out: 1 ;  Stride: 1
        # Layer output image size: 1x6x7x6 ; Pad: 0
        self.conv4 = nn.Conv3d(32, 32, kernel_size=1, pad=0)

        def encoder(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x, ind1 = self.pool1(x)
            x = self.conv2(x)
            x = self.relu(x)
            x, ind2 = self.pool2(x)
            x = self.conv3(x)
            x = self.relu(x)
            x, ind3 = self.pool3(x)
            x = self.conv4(x)
            x = self.relu(x)
            return x, [ind1, ind2, ind3]

        def decoder(self, x, indices):
            return x
        #self.decoder = nn.Sequential(
        #        nn.ConvTranspose3d(, , kernel_size=1),
        #        nn.ReLU(True),
        #        nn.Unpool3d(),
        #        nn.ConvTranspose3d(shape_layer3, shape_layer2, kernel_size=conv_size2),
        #        nn.ReLU(True),
        #        nn.ConvTranspose3d(shape_layer2, shape_layer1, kernel_size=conv_size1),
        #        nn.ReLU(True),
        #        nn.Sigmoid())

    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.decoder(x, indices)
        return x
