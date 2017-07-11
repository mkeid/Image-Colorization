#!/usr/bin/python3

import argparse
import os
from generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('--output-dir', default='data/')
args = parser.parse_args()

save_path = args.output_dir + os.path.basename(args.input)
g_net = Generator().cuda()
g_net.render(args.test_image, save_path)

print("A colorized version of the given image has been rendered to %s" % save_path)
