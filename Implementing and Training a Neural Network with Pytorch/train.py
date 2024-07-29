import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128

# 1 - download dataset
def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data


if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST Datasets Downloaded")

    # 2 - create data loader for the train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

# 3 - build model



# 4 - train
# 5 - save trained model

