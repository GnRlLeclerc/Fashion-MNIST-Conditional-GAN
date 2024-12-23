"""Train a Conditional GAN on the Fashion MNIST dataset."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fashion_mnist import load_fashion_mnist_datasets
from networks import Discriminator, Generator, Parameters


def train_cgan(generator: Generator, discriminator: Discriminator, params: Parameters, device: str):
    """Train a Conditional GAN."""
    generator.to(device)
    discriminator.to(device)

    train_dataset, _ = load_fashion_mnist_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=params.lr, betas=(params.beta1, 0.999))

    # Keep track of the accuracies
    fake_image_detection_rates: list[float] = []

    for _ in tqdm(range(params.epochs)):
        fake_image_detection_rate = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            ###################################################################
            #                       TRAIN DISCRIMINATOR                       #
            ###################################################################

            disc_optimizer.zero_grad()

            # Real images
            output = discriminator.forward(images, labels)
            error_real = F.binary_cross_entropy(output, torch.full_like(output, 1))
            error_real.backward()

            # Fake images
            noise = torch.randn(params.batch_size, params.z_size).to(device)
            fake_labels = torch.randint(0, params.n_classes, (params.batch_size,)).to(device)
            fake_images = generator.forward(noise, fake_labels)

            output = discriminator.forward(fake_images.detach(), fake_labels)
            error_fake = F.binary_cross_entropy(output, torch.full_like(output, 0))
            error_fake.backward()
            detection_rate = (output < 0.5).float().mean().item()
            fake_image_detection_rate += detection_rate

            if detection_rate > 0.9:
                # Stop training the discriminator if it's too good
                disc_optimizer.zero_grad()
            else:
                disc_optimizer.step()

            ###################################################################
            #                         TRAIN GENERATOR                         #
            ###################################################################

            gen_optimizer.zero_grad()

            # Redo a forward pass for the discriminator
            output = discriminator.forward(fake_images, fake_labels)
            error_gen = F.binary_cross_entropy(output, torch.full_like(output, 1))
            error_gen.backward()

            gen_optimizer.step()

        fake_image_detection_rate /= len(train_dataloader)
        fake_image_detection_rates.append(fake_image_detection_rate)

    # Save the models and results
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    np.save("fake_image_detection_rates.npy", fake_image_detection_rates)
