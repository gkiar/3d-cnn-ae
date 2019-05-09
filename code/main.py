#!/usr/bin/env python

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from model import Autoencoder3D


# Create data generation function
def gen_3d_sample(c1=10, c2=2, c3=7, size=(48,56,48)):
    # Create incrementing point grids
    xx, yy, zz = np.meshgrid(np.arange(0, size[0], 1, dtype='float64'),
                             np.arange(0, size[1], 1, dtype='float64'),
                             np.arange(0, size[2], 1, dtype='float64'))
    # Apply some slightly-random transform along each axis
    xx += np.random.randint(c1)/(c2 + c3*np.random.random())
    yy /= (np.random.randint(c2) + 1)/(c3 + c1*np.random.random())
    zz *= (np.random.randint(c3) + 1)/(c1 + c2*np.random.random())

    # Sum all the independent axis functions
    xyz = (xx+yy+zz)

    # Return the sin of these combined functions in the shape (1, X1, X2, X3)
    return torch.tensor(np.sin(xyz)).view((1), *size)

# Set epochs, batch size, and N samples
num_epochs = 5
batch_size = 16
training_samples = 1000
data_shape = (48, 56, 48)

# Generate training data
training_data = []
for _ in range(0, training_samples+batch_size, batch_size):
    t_samples = tuple(gen_3d_sample() for idx in range(batch_size))
    t_samples = torch.cat(t_samples, 0)
    t_samples = t_samples.view((batch_size), (1), *data_shape)

    training_data += [t_samples]

# Initialize model, loss function and optimizer
model = Autoencoder3D().cpu()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

# Perform training
for epoch in range(num_epochs):
    print("Training sample (of {0}): ".format(len(training_data)),
          end='', flush=True)
    for idx, data in enumerate(training_data):
        if idx == 0:
            print(idx+1, end='', flush=True)
        else:
            print(", {0}".format(idx + 1), end='', flush=True)
        img = data
        img = img.type('torch.FloatTensor').cpu()
        # forward
        output = model(img)
        loss = distance(output, img)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('')
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))

