import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
from etl import ETL
from torch.autograd import Variable
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Training images directory path.')

parser.add_argument('--adam-beta1', default=.5)
parser.add_argument('--batch-size', default=64)
parser.add_argument('--epochs', default=50000)
parser.add_argument('--grad-clip', default=.01)
parser.add_argument('--image-size', default=64, help='Length to resize training images to (size x size).')
parser.add_argument('--k-discriminator', default=5, help='Number of times to train discriminator per iteration.')
parser.add_argument('--k-generator', default=1, help='Number of times to train generator per iteration.')
parser.add_argument('--learning-rate-d', default=.0002, help='Learning rate for discriminator optimizer.')
parser.add_argument('--learning-rate-g', default=.0001, help='Learning rate for generator optimizer.')
parser.add_argument('--noise-size', default=100)
parser.add_argument('--optimizer', default='RMSProp', help='[RMSProp, Adam]')
parser.add_argument('--rmsprop-decay', default=.9)
parser.add_argument('--test-image', default=None)

parser.add_argument('--nlog', default=10)
parser.add_argument('--nrender', default=10)
parser.add_argument('--nsave', default=1000)
parser.add_argument('--retrain', action='store_true')
args = parser.parse_args()


g_net = Generator().cuda()
if args.optimizer == 'RMSProp':
    g_opt = optim.RMSprop(g_net.parameters(), args.learning_rate_d, weight_decay=args.rmsprop_decay)
elif args.optimizer == 'Adam':
    g_opt = optim.RMSprop(g_net.parameters(), args.learning_rate_d, (args.adam_beta1, .999))
g_losses = np.empty(0)

d_net = Discriminator().cuda()
if args.optimizer == 'RMSProp':
    d_opt = optim.RMSprop(d_net.parameters(), args.learning_rate_d, weight_decay=args.rmsprop_decay)
elif args.optimizer == 'Adam':
    d_opt = optim.RMSprop(d_net.parameters(), args.learning_rate_d, (args.adam_beta1, .999))
d_losses = np.empty(0)

if args.retrain:
    g_net.load_state_dict(torch.load('data/generator_state'))
    d_net.load_state_dict(torch.load('data/discriminator_state'))

print("Beginning training..")
loader = ETL(args.batch_size, args.image_size, args.path)

for epoch in range(args.epochs):

    # Train discriminator
    for _ in range(args.k_discriminator):
        d_opt.zero_grad()

        d_examples, d_targets = loader.next_batch()
        d_noise = torch.Tensor(args.batch_size, 1, args.image_size, args.image_size).uniform_(-1., 1.)
        d_noise = Variable(d_noise).cuda()
        d_samples = g_net(d_noise, d_examples)
        d_real_pred = d_net(d_targets)
        d_fake_pred = d_net(d_samples)

        d_loss = -torch.mean(d_real_pred - d_fake_pred)
        d_loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm(d_net.parameters(), args.grad_clip)
        d_opt.step()

    # Train generator
    for _ in range(args.k_generator):
        g_opt.zero_grad()

        g_examples, _ = loader.next_batch()
        g_noise = torch.Tensor(args.batch_size, 1, args.image_size, args.image_size).uniform_(-1., 1.)
        g_noise = Variable(g_noise).cuda()
        g_samples = g_net(g_noise, g_examples)
        g_pred = d_net(g_samples)

        g_loss = -torch.mean(g_pred)
        g_loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm(d_net.parameters(), args.grad_clip)
        g_opt.step()

    # Keep track of moving averages for losses
    g_losses = np.append(g_losses, g_loss.data.cpu().numpy())
    d_losses = np.append(d_losses, d_loss.data.cpu().numpy())

    if not epoch: continue

    if not epoch % args.nlog:
        print("Epoch %d | Generator Loss: %.3f | Discriminator Loss: %.3f" % (epoch, g_losses.mean(), d_losses.mean()))
        g_losses = np.empty(0)
        n_losses = np.empty(0)

    if args.test_image and not epoch % args.nrender:
        img = Image.open(args.test_image)
        img = img.convert('HSV')
        img = (loader.toTensor(img) - .5) * 2.
        img_shape = img.size()
        img = Variable(img).cuda()
        img = img[-1].unsqueeze(0).unsqueeze(0)

        z = torch.Tensor(1, 1, img_shape[1], img_shape[2]).uniform_(-1., 1.)
        z = Variable(z).cuda()

        g_net.eval()
        sample = g_net(z, img)
        sample = (sample + 1.) / 2.
        sample = loader.toPIL(sample.squeeze(0).data.cpu())
        sample = sample.convert('RGB')
        sample.save('data/sample.jpg')
        g_net.train()

    if not epoch % args.nsave:
        loader.save_models(g_net, d_net)

print("Training complete.")
loader.save_models(g_net, d_net)
