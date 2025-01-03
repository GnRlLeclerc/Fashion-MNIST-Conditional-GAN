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
    batch_size: int = 64  # Batch size
    n_classes: int = 10  # Number of classes in the dataset (for conditional GAN)
    feature_map_size: int = 64  # Size of the feature maps in the generator & discriminator
    epochs: int = 10  # Amount of training epochs
    lr: float = 0.0002  # Learning rate
    beta1: float = 0.5  # Beta1 for Adam optimizer


class Generator(nn.Module):
    def __init__(self, params: Parameters) -> None:
        super().__init__()
        self.params = params

        # (batch_size, params.z_size + params.n_channels, 1, 1) -> (batch_size, params.feature_map_size * 2, 7, 7)
        self.conv1 = nn.ConvTranspose2d(
            params.z_size + params.n_classes, params.feature_map_size * 2, 7, 1, 0, bias=False
        )
        self.norm1 = nn.BatchNorm2d(params.feature_map_size * 2)
        self.activ = nn.ReLU()
        # (batch_size, params.feature_map_size * 2, 7, 7) -> (batch_size, params.feature_map_size, 14, 14)
        self.conv2 = nn.ConvTranspose2d(params.feature_map_size * 2, params.feature_map_size, 4, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(params.feature_map_size)
        # (batch_size, params.feature_map_size, 14, 14) -> (batch_size, channels=1, 28, 28)
        self.conv3 = nn.ConvTranspose2d(params.feature_map_size, 1, 4, 2, 1, bias=False)
        self.activ_final = nn.Tanh()
        # output: (batch_size, 1, 28, 28)

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        """Generate images from noise vectors
        Args:
            noise (batch_size, params.z_size): Noise vectors
            labels (batch_size): Labels

        Returns:
            (batch_size, 1, 28, 28): Generated images
        """
        one_hot = to_one_hot_batched(labels)
        # (batch_size, params.z_size + params.n_classes)
        concatenated = torch.cat((noise.cpu(), one_hot.cpu()), 1).to(noise.device)

        # Add (1, 1) dimensions to the noise vector
        x = self.activ(self.norm1(self.conv1(concatenated.unsqueeze(-1).unsqueeze(-1))))
        x = self.activ(self.norm2(self.conv2(x)))
        x = self.activ_final(self.conv3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, params: Parameters) -> None:
        super().__init__()
        self.params = params

        # (batch_size, params.n_classes, 28, 28) -> (batch_size, params.feature_map_size, 14, 14)
        self.conv1 = nn.Conv2d(params.n_classes, params.feature_map_size, 4, 2, 1, bias=False)
        self.activ = nn.LeakyReLU(0.2)
        # (batch_size, params.feature_map_size, 14, 14) -> (batch_size, params.feature_map_size * 2, 7, 7)
        self.conv2 = nn.Conv2d(params.feature_map_size, params.feature_map_size * 2, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(params.feature_map_size * 2)
        # (batch_size, params.feature_map_size * 2, 7, 7) -> (batch_size, 1, 1, 1)
        self.conv3 = nn.Conv2d(params.feature_map_size * 2, 1, 7, 1, 0, bias=False)
        self.activ_final = nn.Sigmoid()
        # output: (batch_size, 1, 1, 1)

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        """Discriminate real from fake images
        Args:
            images (batch_size, 1, 28, 28): Images
            labels (batch_size, 1): Labels

        Returns:
            (batch_size, 1): Predictions of the discriminator
        """
        # Create the (batch_size, params.n_classes, 28, 28) "one-hot" image tensor
        encoded = torch.zeros((labels.size(0), self.params.n_classes, 28, 28), device=images.device)

        for i, (image, label) in enumerate(zip(images, labels)):
            encoded[i, int(label.item())] = image.squeeze()

        x = self.activ(self.conv1(encoded))
        x = self.activ(self.norm(self.conv2(x)))
        x = self.activ_final(self.conv3(x))

        return x.squeeze(-1).squeeze(-1)  # Remove the last 2 dimensions
