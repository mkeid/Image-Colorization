import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import tee
from torch.autograd import Variable
from torch.utils.data import DataLoader


class ETL:
    toPIL = transforms.ToPILImage()
    toTensor = transforms.ToTensor()

    def __init__(self, batch_size, image_size, training_dir):
        # Keep params
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
            images[i] = (self.toTensor(image) - .5) * 2

        images = torch.Tensor(images)
        y = images[:, -1].unsqueeze(1)
        return Variable(y).cuda(), Variable(images).cuda()

    @staticmethod
    def save_models(g_net, d_net):
        print("Saving models..")
        torch.save(g_net.state_dict(), 'data/generator_state')
        torch.save(d_net.state_dict(), 'data/discriminator_state')
        print("Model states have been saved to the data directory.")
