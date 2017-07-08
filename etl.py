import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Pool
from torch.autograd import Variable
from torch.utils.data import DataLoader


class ETL:
    def __init__(self, batch_size, image_size):
        # Keep params
        self.batch_size = batch_size
        self.image_size = image_size

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])
        self.dataset = dset.ImageFolder('/home/mo/Datasets/coco/', transform=transform)

        self.loader = iter(DataLoader(self.dataset, self.batch_size))
        self.n_examples = len(self.dataset)

        self.toPIL = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()

    def next_batch(self):
        images = next(self.loader)[0]

        for i in range(self.batch_size):
            image = self.toPIL(images[i])
            image = image.convert('HSV')
            images[i] = self.toTensor(image)

        images = torch.Tensor(images)
        y = images[:, -1].unsqueeze(1)
        return Variable(y).cuda(), Variable(images).cuda()

    @staticmethod
    def rgb2hsv(image):
        image = Image.fromarray(np.uint8(image))
        image = image.convert('HSV')
        image = np.array(image)
        return image / 127.5 - 1.

    @staticmethod
    def hsv2rgb(image):
        image = Image.fromarray(np.uint8((image + 1) * 127.5))
        image = image.convert('RGB')
        image = np.array(image)
        return image

    @staticmethod
    def rgb2yuv(image):
        cvt_matrix = np.array([[0.299, -0.169, 0.5],
                               [0.587, -0.331, -0.419],
                               [0.114, 0.5, -0.081]], dtype=np.float32)
        image = image.dot(cvt_matrix)
        image += [0, 127.5, 127.5]
        return image / 127.5 - 1.

    @staticmethod
    def yuv2rgb(image):
        cvt_matrix = np.array([[1, 1, 1],
                               [-0.00093, -0.3437, 1.77216],
                               [1.401687, -0.71417, 0.00099]], dtype=np.float32)
        image -= [0, 127.5, 127.5]
        return image.dot(cvt_matrix).clip(min=0, max=255)
