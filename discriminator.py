import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Activation for each block
        self.relu = nn.LeakyReLU()

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, 2, 1)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, 2, 1)
        self.norm2 = nn.BatchNorm2d(128)

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, 2, 1)
        self.norm3 = nn.BatchNorm2d(256)

        # Block 4
        self.conv4 = nn.Conv2d(256, 512, 2, 1)
        self.norm4 = nn.BatchNorm2d(512)

        # Prediction
        self.prediction = nn.Linear(512, 1)

    def forward(self, x):
        for i in range(1, 5):
            x = self.conv_block(x, i, (i > 1))

        x = x.view(-1, 512)
        return self.prediction(x)

    def conv_block(self, x, block, norm=True):
        conv = getattr(self, "conv{}".format(block))
        x = conv(x)

        if norm:
            norm = getattr(self, "norm{}".format(block))
            x = norm(x)

        return self.relu(x)
