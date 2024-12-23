"""
Fashion MNIST dataset.
"""

import datasets
import torch
from datasets import DatasetDict, load_dataset
from PIL.PngImagePlugin import PngImageFile
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

###############################################################################
#                             TYPING & CONSTANTS                              #
###############################################################################

# FashionMNIST labels from label indexes
index_to_label = {
    0: "T - shirt / top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


###############################################################################
#                                DATASET CLASS                                #
###############################################################################


class MNISTDataset(Dataset):
    """Fashion MNIST dataset."""

    def __init__(self, dataset: datasets.Dataset) -> None:
        """Instanciate a PyTorch-compatible Fashion MNIST dataset from a `datasets` dataset."""
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get a sample from the dataset.

        Returns
         - image: (1, 28, 28) tensor
         - label: (1,) label index tensor
        """
        sample = self.dataset[index]
        image: PngImageFile = sample["image"]
        label: int = sample["label"]

        # Convert the image to [-1, 1], float32, like the GAN output
        processed = pil_to_tensor(image) / 255 - 0.5
        return processed, torch.tensor([label])


def load_fashion_mnist_datasets() -> tuple[MNISTDataset, MNISTDataset]:
    """Load Fashion MNIST train & test datasets.

    Returns:
        A tuple containing the train and test datasets, in that order.
    """
    mnist: DatasetDict = load_dataset("zalando-datasets/fashion_mnist")  # type: ignore
    return MNISTDataset(mnist["train"]), MNISTDataset(mnist["test"])
