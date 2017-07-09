import argparse
import os
import torch
from etl import ETL
from generator import Generator
from discriminator import Discriminator
from PIL import Image
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('--output-dir', default='data/')
args = parser.parse_args()

g_net = Generator().cuda()
d_net = Discriminator().cuda()

img = Image.open(args.test_image)
img = img.convert('HSV')
img = (ETL.toTensor(img) - .5) * 2.
img_shape = img.size()
img = Variable(img).cuda()
img = img[-1].unsqueeze(0).unsqueeze(0)

z = torch.Tensor(1, 1, img_shape[1], img_shape[2]).uniform_(-1., 1.)
z = Variable(z).cuda()

g_net.eval()
sample = g_net(z, img)
sample = (sample + 1.) / 2.
sample = ETL.toPIL(sample.squeeze(0).data.cpu())
sample = sample.convert('RGB')

save_path = args.output_dir + os.path.basename(args.input)
sample.save(save_path)
print("A colorized version of the given image has been rendered to %s" % save_path)
