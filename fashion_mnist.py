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


def to_one_hot(label: int) -> Tensor:
    """Convert a label to a one-hot tensor."""
    encoded_label = torch.zeros(len(index_to_label))
    encoded_label[label] = 1
    return encoded_label


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

        return pil_to_tensor(image), torch.tensor([label])


def load_fashion_mnist_datasets() -> tuple[MNISTDataset, MNISTDataset]:
    """Load Fashion MNIST train & test datasets.

    Returns:
        A tuple containing the train and test datasets, in that order.
    """
    mnist: DatasetDict = load_dataset("zalando-datasets/fashion_mnist")  # type: ignore
    return MNISTDataset(mnist["train"]), MNISTDataset(mnist["test"])
