"""Display utils"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fashion_mnist import index_to_label
from gradcam import GradCAM
from networks import Discriminator, Generator


def plot_tensor(x: Tensor) -> None:
    """Plot a PyTorch tensor as a grayscale image."""

    assert len(x.shape) == 3, f"Expected a 3D tensor, got {x.shape}"

    np_image = x.squeeze().detach().numpy()

    plt.imshow(np_image, cmap="gray")
    plt.show()


def showcase(generator: Generator, device: str):
    """Display a grid of generated images with their labels to showcase a trained generator."""

    n = generator.params.n_classes
    generator.to(device)

    fig, axes = plt.subplots(n, n, figsize=(n, n + 1), sharex=True, sharey=True)
    for row in range(n):
        # Generate n images with the same label for this row
        noise = torch.randn(n, generator.params.z_size, device=device)
        labels = torch.full((n,), row, device=device)
        images = generator.forward(noise, labels)

        for col in range(n):
            ax = axes[row, col]
            ax.imshow(images[col].squeeze().detach().cpu().numpy(), cmap="gray")

            # Hide the ticks
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row, 0].set_ylabel(index_to_label[row], rotation=45, size="large", labelpad=20)

    fig.suptitle("Conditional GAN generated images for all labels")
    plt.show()


def showcase_gradcam(generator: Generator, discriminator: Discriminator, device: str):
    """Display a grid of generated images with their labels and discriminator heatmaps"""

    n = generator.params.n_classes
    generator.to(device)
    discriminator.to(device)
    gradcam = GradCAM(discriminator.conv1)

    fig, axes = plt.subplots(n, n, figsize=(n, n + 1), sharex=True, sharey=True)
    for row in range(n):
        # Generate n images with the same label for this row
        noise = torch.randn(n, generator.params.z_size, device=device)
        labels = torch.full((n,), row, device=device)
        images = generator.forward(noise, labels)

        output = discriminator.forward(images, labels)
        error = F.binary_cross_entropy(output, torch.full_like(output, 0))
        error.backward()  # Backpropagate the error to compute gradients
        heatmaps = gradcam.compute()
        upscaled = np.repeat(np.repeat(heatmaps, 2, axis=1), 2, axis=2)

        for col in range(n):
            ax = axes[row, col]
            ax.imshow(images[col].squeeze().detach().cpu().numpy(), cmap="gray")
            ax.imshow(upscaled[col], cmap="jet", alpha=0.4)

            # Hide the ticks
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row, 0].set_ylabel(index_to_label[row], rotation=45, size="large", labelpad=20)

    fig.suptitle("Conditional GAN generated images for all labels, with heatmaps")
    plt.show()
