import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import tee
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader


class ToPILImage(object):
    """
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving the value range.
    """

    def __call__(self, pic):
        """
        Args:
            pic (Tensor) :   Image to be converted to PIL.Image.
        Returns:
            PIL.Image    :   Image converted to PIL.Image.
        """

        npimg = pic

        if isinstance(pic, torch.FloatTensor):
            pic = pic.mul(255).byte()

        if torch.is_tensor(pic):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'

        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

        return Image.fromarray(npimg, mode='HSV')


class ETL:
    """
    Responsible for data extraction, transformation and loading.
    """

    toPIL = ToPILImage()
    toTensor = transforms.ToTensor()

    def __init__(self, batch_size, image_size, training_dir):
        self.batch_size = batch_size
        self.image_size = image_size

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

        self.dataset = dset.ImageFolder(training_dir, transform=transform)
        self.loader = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=True))
        self.loader, self.loader_backup = tee(self.loader)

    def next_batch(self):
        """
        Retrieves next batch of examples.
        
        Returns: 
            Tensor, Tensor  :   (grayscale examples, color labels)
        """

        try:
            images = next(self.loader)[0]
        except StopIteration:
            self.loader, self.loader_backup = tee(self.loader_backup)
            images = next(self.loader)[0]
        except OSError:
            images = next(self.loader)[0]

        for i in range(self.batch_size):
            image = self.toPIL(images[i])
            image = image.convert('HSV')
            image = (self.toTensor(image) - .5) * 2
            images[i] = image

        images = torch.Tensor(images)
        y = images[:, -1].unsqueeze(1)
        return Variable(y).cuda(), Variable(images).cuda()

    @staticmethod
    def save_models(g_net, d_net):
        """
        Saves states of generator and discriminator.
        
        Args:
            g_net (nn.Module)   :   The generative model.
            d_net (nn.Module)   :   The discriminative model.
        """

        print("Saving models..")
        torch.save(g_net.state_dict(), 'data/generator_state')
        torch.save(d_net.state_dict(), 'data/discriminator_state')
        print("Model states have been saved to the data directory.")



