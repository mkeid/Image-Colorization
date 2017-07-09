import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    
    """

    def __init__(self):
        super(Generator, self).__init__()

        # Conv blocks
        self.conv1 = nn.Conv2d(3, 130, 7, padding=3)
        self.norm1 = nn.BatchNorm2d(130)

        self.conv2 = nn.Conv2d(131, 66, 5, padding=2)
        self.norm2 = nn.BatchNorm2d(66)

        self.conv3 = nn.Conv2d(67, 65, 5, padding=2)
        self.norm3 = nn.BatchNorm2d(65)

        self.conv4 = nn.Conv2d(66, 65, 5, padding=2)
        self.norm4 = nn.BatchNorm2d(65)

        self.conv5 = nn.Conv2d(66, 33, 5, padding=2)
        self.norm5 = nn.BatchNorm2d(33)

        self.relu = nn.ReLU()

        # Out
        self.conv6 = nn.Conv2d(34, 2, 5, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, noise, y):
        """
        
        :param noise: 
        :param y: 
        :return: 
        """

        x = torch.cat([y, noise], 1)

        for i in range(1, 7):
            x = self.conv_block(y, x, i, (i == 6))

        return torch.cat([x, y], 1)

    def conv_block(self, y, x, block, last=False):
        """
        
        :param y: 
        :param x: 
        :param block: 
        :param last: 
        :return: 
        """

        x = torch.cat([x, y], 1)
        conv = getattr(self, "conv{}".format(block))
        x = conv(x)

        if not last:
            norm = getattr(self, "norm{}".format(block))
            x = norm(x)
            return self.relu(x)
        else:
            return self.tanh(x)
