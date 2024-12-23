"""
Implementation inspired from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Adapted for conditional GANs & 1-channel, 28x28 images.

Conditional integration:
- Generator: labels are 1-hot encoded and concatenated with the noise vector
- Discriminator: labels are integrated as a channel in the input images (1, 28, 28) -> (10, 28, 28)
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from utils import to_one_hot_batched


@dataclass
class Parameters:
    """GAN parameters"""

    z_size: int = 100  # Size of the input noise vector
    n_classes: int = 10  # Number of classes in the dataset (for conditional GAN)
    feature_map_size: int = 64  # Size of the feature maps in the generator & discriminator
    channels: int = 1  # Number of channels in the input images
    epochs: int = 5  # Amount of training epochs
    lr: float = 0.0002  # Learning rate
    beta1: float = 0.5  # Beta1 for Adam optimizer


class Generator(nn.Module):
    def __init__(self, params: Parameters) -> None:
        super().__init__()
        self.params = params

        self.main = nn.Sequential(
            # (batch_size, params.z_size, 1, 1) -> (batch_size, params.feature_map_size * 2, 7, 7)
            nn.ConvTranspose2d(params.z_size + params.n_classes, params.feature_map_size * 2, 7, 1, 0, bias=False),
            nn.BatchNorm2d(params.feature_map_size * 2),
            nn.ReLU(True),
            # (batch_size, params.feature_map_size * 2, 7, 7) -> (batch_size, params.feature_map_size, 14, 14)
            nn.ConvTranspose2d(params.feature_map_size * 2, params.feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params.feature_map_size),
            nn.ReLU(True),
            # (batch_size, params.feature_map_size, 14, 14) -> (batch_size, params.channels, 28, 28)
            nn.ConvTranspose2d(params.feature_map_size, params.channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output: (batch_size, params.channels, 28, 28)
        )

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        """Generate images from noise vectors
        Args:
            noise (batch_size, params.z_size): Noise vectors
            labels (batch_size, 1): Labels
        """
        one_hot = to_one_hot_batched(labels)
        concatenated = torch.cat((noise, one_hot), 1)  # (batch_size, params.z_size + params.n_classes)
        return self.main(concatenated.unsqueeze(-1).unsqueeze(-1))  # Add (1, 1) dimensions to the noise vector


class Discriminator(nn.Module):
    pass
