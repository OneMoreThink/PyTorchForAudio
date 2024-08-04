import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


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

# 3 - build model
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        flatten_data = self.flatten(input_data)
        logits = self.dense(flatten_data)
        predictions = self.softmax(logits)
        return  predictions

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        predictions = model(input)
        loss = loss_fn(predictions, target)

        # backpropagation loss update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss.item():.4f}')

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}')
        train_one_epoch(model,data_loader,loss_fn,optimizer,device)
        print("------------------")
print("Training is Done")


if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST Datasets Downloaded")

    # 2 - create data loader for the train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # 3 build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    feedforward_net = FeedForwardNet().to(device)

    # instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feedforward_net.parameters(), lr=LEARNING_RATE)

    # 4 train model
    train(feedforward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    # 5 - save trained model
    torch.save(feedforward_net.state_dict(), "feedforward_net.pth")
    print("Model trained and stored at feedforward_net.pth")
