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
parser.add_argument('--batch-size', default=64)
parser.add_argument('--epochs', default=100)
parser.add_argument('--grad-clip', default=.01)
parser.add_argument('--image-size', default=64)
parser.add_argument('--learning-rate-d', default=.0002)
parser.add_argument('--learning-rate-g', default=.0001)
parser.add_argument('--adam-beta1', default=.5)
parser.add_argument('--noise-size', default=100)
parser.add_argument('--test-image', default='data/bird.jpg')

parser.add_argument('--nlog', default=100)
parser.add_argument('--nrender', default=1)
parser.add_argument('--nsave', default=1000)
args = parser.parse_args()


# Initialize models
g_net = Generator(args.noise_size, args.image_size).cuda()
d_net = Discriminator().cuda()

# Initialize optimization ops
g_opt = optim.Adam(g_net.parameters(), args.learning_rate_g, (args.adam_beta1, .999))
d_opt = optim.Adam(d_net.parameters(), args.learning_rate_d, (args.adam_beta1, .999))

loader = ETL(args.batch_size, args.image_size)


def save_models():
    global g_net, d_net
    print("Saving models..")
    torch.save(g_net.state_dict(), 'data/generator_state')
    torch.save(d_net.state_dict(), 'data/discriminator_state')
    print("Model states have been saved to the data directory.")


# Begin training cycle
g_losses = np.empty(0)
d_losses = np.empty(0)
print("Beginning training..")

for epoch in range(args.epochs):
    # Train discriminator
    d_opt.zero_grad()

    d_examples, d_targets = loader.next_batch()
    d_noise = torch.Tensor(args.batch_size, 1, args.image_size, args.image_size).uniform_(-1., 1.)
    d_noise = Variable(d_noise).cuda()
    d_samples = g_net(d_noise, d_examples)
    d_real_pred = d_net(d_targets)
    d_fake_pred = d_net(d_samples)

    d_loss = -torch.mean(d_real_pred - d_fake_pred)
    d_loss.backward()
    nn.utils.clip_grad_norm(d_net.parameters(), args.grad_clip)
    d_opt.step()

    # Train generator
    g_opt.zero_grad()

    g_examples, _ = loader.next_batch()
    g_noise = torch.Tensor(args.batch_size, 1, args.image_size, args.image_size).uniform_(-1., 1.)
    g_noise = Variable(g_noise).cuda()
    g_samples = g_net(g_noise, g_examples)
    g_pred = d_net(g_samples)

    g_loss = -torch.mean(g_pred)
    g_loss.backward()
    nn.utils.clip_grad_norm(d_net.parameters(), args.grad_clip)
    g_opt.step()

    # Keep track of moving averages for losses
    g_losses = np.append(g_losses, g_loss)
    d_losses = np.append(d_losses, d_loss)

    if not epoch:
        continue

    if not epoch % args.nlog:
        print("Epoch {} | Generator Loss: {}| Discriminator Loss: {}".format(epoch, g_losses.mean(), d_losses.mean()))
        g_losses = np.empty(0)
        n_losses = np.empty(0)

    if not epoch % args.nrender:
        img = Image.open(args.test_image)
        img = img.convert('HSV')
        img = loader.toTensor(img)
        img_shape = img.size()
        img = Variable(img).cuda()
        img = img[-1].unsqueeze(0).unsqueeze(0)

        z = torch.Tensor(1, 1, img_shape[1], img_shape[2]).uniform_(-1., 1.)
        z = Variable(z).cuda()

        sample = g_net(z, img)
        sample = loader.toPIL(sample.squeeze(0).data.cpu())
        sample.save('output/sample.jpg')

    if not epoch % args.nsave:
        save_models()

print("Training complete.")
save_models()
