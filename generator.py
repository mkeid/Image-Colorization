import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_size, image_size):
        super(Generator, self).__init__()

        # Keep params
        self.noise_size = noise_size
        self.image_size = image_size

        # Noise projection
        self.project = nn.Linear(self.noise_size, self.image_size * self.image_size)
        self.norm0 = nn.BatchNorm2d(1)

        # Activation for each block
        self.relu = nn.ReLU()


        # Block 1
        self.conv1 = nn.Conv2d(3, 130, 1, 1)
        self.norm1 = nn.BatchNorm2d(130)

        # Block 2
        self.conv2 = nn.Conv2d(131, 66, 1, 1)
        self.norm2 = nn.BatchNorm2d(66)

        # Block 3
        self.conv3 = nn.Conv2d(67, 65, 1, 1)
        self.norm3 = nn.BatchNorm2d(65)

        # Block 4
        self.conv4 = nn.Conv2d(66, 65, 1, 1)
        self.norm4 = nn.BatchNorm2d(65)

        # Block 5
        self.conv5 = nn.Conv2d(66, 33, 1, 1)
        self.norm5 = nn.BatchNorm2d(33)

        # Output
        self.conv6 = nn.Conv2d(34, 2, 1, 1)

        # Last activation
        self.tanh = nn.Tanh()

    def forward(self, noise, y):
        noise = self.project(noise).view(-1, 1, self.image_size, self.image_size)
        noise = self.norm0(noise)
        noise = self.relu(noise)
        x = torch.cat([y, noise], 1)

        for i in range(1, 7):
            x = self.conv_block(y, x, i, (i == 6))

        return torch.cat([y, x], 1)

    def conv_block(self, y, x, block, last=False):
        x = torch.cat([y, x], 1)
        conv = getattr(self, "conv{}".format(block))
        x = conv(x)

        if not last:
            norm = getattr(self, "norm{}".format(block))
            x = norm(x)
            return self.relu(x)
        else:
            return self.tanh(x)
