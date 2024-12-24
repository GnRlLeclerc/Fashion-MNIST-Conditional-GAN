"""Utility functions."""

import numpy as np
import torch
from torch import Tensor, nn

from fashion_mnist import index_to_label


def to_one_hot(label: int) -> Tensor:
    """Convert a label to a one-hot tensor."""
    encoded_label = torch.zeros(len(index_to_label))
    encoded_label[label] = 1
    return encoded_label


def to_one_hot_batched(labels: Tensor) -> Tensor:
    """Convert a batch of labels to one-hot tensors.

    Args:
        labels (batch_size): Labels

    Returns:
        (batch_size, n_classes): One-hot tensors"""
    assert len(labels.shape) == 1, f"Expected a 1D tensor, got {labels.shape}"

    return torch.stack([to_one_hot(int(label.item())) for label in labels])


def weights_init(module: nn.Module):
    """Initialize model weights.

    Usage:
    ```
    model = Generator(params)
    model.apply(weights_init)
    ```
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
