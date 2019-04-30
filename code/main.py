#!/usr/bin/env python

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.utils import save_image

from model import Autoencoder


# Loading and Transforming data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4466),
                                                     (0.247, 0.243, 0.261))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(),
                                         tv.transforms.Normalize((0.4914, 0.4822, 0.4466),
                                                                 (0.247, 0.243, 0.261))])

trainset = tv.datasets.CIFAR10(root='../data',  train=True,
                               download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                         shuffle=False, num_workers=4)

testset = tv.datasets.CIFAR10(root='../data', train=False,
                              download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

num_epochs = 5
batch_size = 128

model = Autoencoder().cpu()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np

def imshow(img, output):
    img = img.data.numpy()[0].T
    output = output.data.numpy()[0].T

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(output)
    plt.subplot(1,3,3)
    plt.imshow(np.abs(img-output))
    plt.show()

for epoch in range(num_epochs):
    for data in dataloader:
        img, _  = data
        img = Variable(img).cpu()
        # forward
        output = model(img)
        loss = distance(output, img)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log
    imshow(img, output)
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))



