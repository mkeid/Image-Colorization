import torch.nn as nn


class Discriminator(nn.Module):
    """
    
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Activation for each block
        self.relu = nn.LeakyReLU()

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, 5, 2, padding=2)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.norm2 = nn.BatchNorm2d(128)

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, 5, 2, padding=2)
        self.norm3 = nn.BatchNorm2d(256)

        # Block 4
        self.conv4 = nn.Conv2d(256, 512, 5, 2, padding=2)
        self.norm4 = nn.BatchNorm2d(512)

        # Prediction
        self.project = nn.Linear(512, 64)
        self.predict = nn.Linear(64, 1)

    def forward(self, x):
        """
        
        :param x: 
        :return: 
        """

        for i in range(1, 5):
            x = self.conv_block(x, i, (i > 1))

        x = x.view(-1, 512)
        x = self.project(x)
        return self.predict(x)

    def conv_block(self, x, block, norm=True):
        """
        
        :param x: 
        :param block: 
        :param norm: 
        :return: 
        """

        conv = getattr(self, "conv{}".format(block))
        x = conv(x)

        if norm:
            norm = getattr(self, "norm{}".format(block))
            x = norm(x)

        return self.relu(x)
