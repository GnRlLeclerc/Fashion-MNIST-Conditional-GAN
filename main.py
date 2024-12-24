"""
Train and showcase the Conditional GAN
"""

import torch

from display import showcase, showcase_gradcam
from networks import Discriminator, Generator, Parameters
from train_cgan import train_cgan
from utils import weights_init

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
params = Parameters()

generator = Generator(params)
generator.apply(weights_init)

discriminator = Discriminator(params)
discriminator.apply(weights_init)

# If they exist, load the model weights
try:
    generator.load_state_dict(torch.load("generator.pth", weights_only=True))
    discriminator.load_state_dict(torch.load("discriminator.pth", weights_only=True))
except FileNotFoundError:
    # If the files don't exist, train the models
    train_cgan(generator, discriminator, params, device)

# Showcase the generator images
showcase(generator, device)
showcase_gradcam(generator, discriminator, device)
