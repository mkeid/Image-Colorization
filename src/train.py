#!/usr/bin/python3

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from etl import ETL
from generator import Generator
from torch.autograd import Variable
from discriminator import Discriminator

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('path', help='Training images directory path.')

parser.add_argument('--batch-size', default=64, help='Training batch size.')
parser.add_argument('--iterations', default=500000, help='Number of iterations to train.')
parser.add_argument('--grad-clip', default=.01, help='Bound of which to clip the gradients.')
parser.add_argument('--image-size', default=64, help='Length to resize training images to (size x size).')
parser.add_argument('--k-discriminator', default=5, help='Number of times to train discriminator per iteration.')
parser.add_argument('--k-generator', default=1, help='Number of times to train generator per iteration.')
parser.add_argument('--learning-rate-d', default=.0002, help='Learning rate for discriminator optimizer.')
parser.add_argument('--learning-rate-g', default=.0001, help='Learning rate for generator optimizer.')
parser.add_argument('--rmsprop-decay', default=.9, help='Weight decay parameter value of RMSProp optimizer.')
parser.add_argument('--test-image', default=None, help='Path of image to render while training.')
parser.add_argument('--test-image-out', default=dir_path + '/../data/sample.jpg', help='Path to render test image.')

parser.add_argument('--nlog', default=10, help='Log error every nlog iterations.')
parser.add_argument('--nrender', default=10, help='Render test image every nrender iterations.')
parser.add_argument('--nsave', default=100, help='Save states of models every nsave iterations.')
parser.add_argument('--retrain', action='store_true', help='Whether or not to start training from a previous state.')
args = parser.parse_args()


print("Initializing generator model and optimizer.")
g_net = Generator().cuda()
g_opt = optim.RMSprop(g_net.parameters(), args.learning_rate_d, weight_decay=args.rmsprop_decay)
g_losses = np.empty(0)

print("Initializing discriminator model and optimizer.")
d_net = Discriminator().cuda()
d_opt = optim.RMSprop(d_net.parameters(), args.learning_rate_d, weight_decay=args.rmsprop_decay)
d_losses = np.empty(0)

if args.retrain:
    g_net.load_state_dict(torch.load('../data/generator_state'))
    d_net.load_state_dict(torch.load('../data/discriminator_state'))

print("Beginning training..")
loader = ETL(args.batch_size, args.image_size, args.path)

for iteration in range(args.iterations):

    # Train discriminator
    for _ in range(args.k_discriminator):
        d_opt.zero_grad()

        d_examples, d_targets = loader.next_batch()
        d_noise = torch.Tensor(args.batch_size, 1, args.image_size, args.image_size).uniform_(-1., 1.)
        d_noise = Variable(d_noise).cuda()
        d_samples = g_net(d_noise, d_examples).detach()

        d_real_pred = d_net(d_targets)
        d_fake_pred = d_net(d_samples)
        d_loss = -torch.mean(d_real_pred - d_fake_pred)
        d_loss.backward()

        for param in d_net.parameters():
            param.grad.data.clamp(-args.grad_clip, args.grad_clip)
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
        g_opt.step()

    # Keep track of moving averages for losses
    g_losses = np.append(g_losses, g_loss.data.cpu().numpy())
    d_losses = np.append(d_losses, d_loss.data.cpu().numpy())

    if not iteration: continue

    if not iteration % args.nlog:
        g_losses_mean = g_losses.mean()
        d_losses_mean = d_losses.mean()
        total_loss = g_losses_mean + d_losses_mean
        percent_done = iteration / args.iterations * 100
        print("Iteration [%05d/%d] (%02d%%) | Total Loss: %.3f | G Loss: %.3f | D Loss: %.3f"
              % (iteration, args.iterations, percent_done, total_loss, g_losses_mean, d_losses_mean))
        g_losses = np.empty(0)
        n_losses = np.empty(0)

    if args.test_image and not iteration % args.nrender:
        g_net.render(args.test_image, args.test_image_out)
        g_net.train()

    if not iteration % args.nsave:
        loader.save_models(g_net, d_net)

print("Training complete.")
loader.save_models(g_net, d_net)
