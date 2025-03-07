import numpy as np
import torch
import torch.nn as nn


def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 315875385


def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000


class CNN(nn.Module):
    def __init__(self):  # Do NOT change the signature of this function
        super(CNN, self).__init__()
        self.n = 10
        self.kernel_size = 7
        self.padding = (self.kernel_size - 1) // 2

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2 * self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(in_channels=2 * self.n, out_channels=4 * self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv4 = nn.Conv2d(in_channels=4 * self.n, out_channels=8 * self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.fc1 = nn.Linear(8 * self.n * 28 * 14, 100)
        self.fc2 = nn.Linear(100, 2)
        self.relu = nn.ReLU()

        # TODO: complete this method

    def forward(self, inp):  # Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # First Conv
        inp = self.conv1(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Second Conv
        inp = self.conv2(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Third Conv
        inp = self.conv3(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Fourth Conv
        inp = self.conv4(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Spread
        inp = inp.contiguous().view(-1, 8 * self.n * 28 * 14)

        # First FC
        inp = self.fc1(inp)
        inp = self.relu(inp)

        # Second FC
        out = self.fc2(inp)
        # TODO: complete this function
        return out


class CNNChannel(nn.Module):
    def __init__(self):  # Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        self.n = 40
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2 * self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(in_channels=2 * self.n, out_channels=4 * self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv4 = nn.Conv2d(in_channels=4 * self.n, out_channels=8 * self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.fc1 = nn.Linear(8 * self.n * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 2)
        self.relu = nn.ReLU()

        # TODO: complete this method

    # TODO: complete this class
    def forward(self, inp):  # Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        left_shoes = inp[:, :, :224, :]
        right_shoes = inp[:, :, 224:, :]
        inp = torch.cat((left_shoes, right_shoes), dim=1)

        # First Conv
        inp = self.conv1(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Second Conv
        inp = self.conv2(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Third Conv
        inp = self.conv3(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Fourth Conv
        inp = self.conv4(inp)
        inp = self.relu(inp)
        inp = self.pool(inp)

        # Spread
        inp = inp.contiguous().view(-1, 8 * self.n * 14 * 14)

        # First FC
        inp = self.fc1(inp)
        inp = self.relu(inp)

        # Second FC
        out = self.fc2(inp)
        # TODO: complete this function
        return out
