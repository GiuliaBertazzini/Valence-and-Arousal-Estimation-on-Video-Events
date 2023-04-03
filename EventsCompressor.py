import torch
from torch import nn
from torch.nn import Module


class EventsCompressor(nn.Module):
    def __init__(self):
        super(EventsCompressor, self).__init__()

        self.conv1 = nn.Conv3d(1, 3, kernel_size=(5, 5, 5), stride=(5, 1, 1), padding=(2, 2, 2), bias=True)
        self.activation = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(3, 3, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2), bias=True)

        self.compression_ratio = self.conv1.kernel_size[0] * self.conv2.kernel_size[0]

    def forward(self, VET):  # batch x M_cappuccio x H x W
        return self.conv2(self.activation(self.conv1(VET.unsqueeze(1))))  # add channels dimension --> batch x 1 x M_cappuccio x H x W
