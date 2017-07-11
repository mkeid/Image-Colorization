import torch.nn as nn
import torch.nn.init as init


class Discriminator(nn.Module):
    """
    Discriminative model that learns to score a given image, assigning
    higher values to encodings that are naturally colorized and lower
    values to encodings that were produced by the generator.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Conv blocks
        self.conv1 = nn.Conv2d(3, 64, 5, 2, padding=2)
        init.normal(self.conv1.weight, 0., .02)

        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        init.normal(self.conv2.weight, 0., .02)
        self.norm2 = nn.BatchNorm2d(128, momentum=.9)

        self.conv3 = nn.Conv2d(128, 256, 5, 2, padding=2)
        init.normal(self.conv3.weight, 0., .02)
        self.norm3 = nn.BatchNorm2d(256, momentum=.9)

        self.conv4 = nn.Conv2d(256, 512, 5, 2, padding=2)
        init.normal(self.conv4.weight, 0., .02)
        self.norm4 = nn.BatchNorm2d(512, momentum=.9)

        self.relu = nn.LeakyReLU(.2)

        # Prediction
        self.project = nn.Linear(512, 64)
        init.normal(self.project.weight, 0., .02)
        self.predict = nn.Linear(64, 1)
        init.normal(self.predict.weight, 0., .02)

    def forward(self, x):
        """
        Given a three-channel colorized image encoding, runs through all conv
        blocks and outputs a scoring for how confident the network is that 
        images are naturally colorized vs. come from the generator.
        
        Args:
            x (Tensor)  :   Three-channel image encoding represented in HSV.
        Returns:
            Tensor      :   Scores for each image. 
        """

        for i in range(1, 5):
            x = self._conv_block(x, i, (i > 1))

        x = x.view(-1, 512)
        x = self.project(x)
        return self.predict(x)

    def _conv_block(self, x, block, norm=True):
        """
        Multi-operation layer comprised of convolution and normalization (except
        for first layer).
         
        Args:
            x (Tensor)  :   Encoding to apply conv -> normalization (possibly) on.
            block (int) :   Block number corresponding to the layer.
            norm (bool) :   Whether or not apply batch normalization.
        Returns:
            Tensor      :   Layer output that has been conved and possibly normalized.
        """

        conv = getattr(self, "conv{}".format(block))
        x = conv(x)

        if norm:
            norm = getattr(self, "norm{}".format(block))
            x = norm(x)

        return self.relu(x)
