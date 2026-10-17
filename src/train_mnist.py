from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_model import MNISTClassifier


RANDOM_SEED = 42

BATCH_SIZE = 64

LEARNING_RATE = 0.001

EPOCHS = 5


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"



def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_data_loaders():

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        running_loss /
        len(data_loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


def evaluate(
    model,
    data_loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        running_loss /
        len(data_loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


def main():

    torch.manual_seed(RANDOM_SEED)

    device = get_device()

    print(f"Using device: {device}")

    train_loader, test_loader = (
        create_data_loaders()
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    model = MNISTClassifier().to(device)

    print("\nModel Architecture")
    print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    if device.type == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        test_loss, test_accuracy = (
            evaluate(
                model,
                test_loader,
                criterion,
                device
            )
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_accuracies.append(
            train_accuracy
        )

        test_accuracies.append(
            test_accuracy
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy:.2%}"
        )

        print(
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: "
            f"{test_accuracy:.2%}"
        )

    if device.type == "mps":
        torch.mps.synchronize()

    training_time = time.perf_counter() - start_time
    
    print(
        f"\nTotal Training Time: "
        f"{training_time:.2f} seconds"
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODELS_DIR /
        "mnist_classifier.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"\nModel saved to: "
        f"{model_path}"
    )


    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    epochs = range(
        1,
        EPOCHS + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        test_losses,
        label="Test Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "MNIST Training and Test Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "mnist_loss_curve.png"
    )

    plt.close()

if __name__ == "__main__":
    main()