from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_model import MNISTClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

MODEL_PATH = MODELS_DIR / "mnist_classifier.pth"

BATCH_SIZE = 64


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_test_data():

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


def load_model(device):

    model = MNISTClassifier().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def evaluate_model(model, test_loader, device):

    all_predictions = []
    all_labels = []

    misclassified_images = []
    misclassified_predictions = []
    misclassified_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            incorrect = predictions != labels

            incorrect_images = images[incorrect]
            incorrect_predictions = predictions[incorrect]
            incorrect_labels = labels[incorrect]

            for image, prediction, label in zip(
                incorrect_images,
                incorrect_predictions,
                incorrect_labels
            ):
                if len(misclassified_images) < 12:

                    misclassified_images.append(
                        image.cpu()
                    )

                    misclassified_predictions.append(
                        prediction.cpu().item()
                    )

                    misclassified_labels.append(
                        label.cpu().item()
                    )

    return (
        all_predictions,
        all_labels,
        misclassified_images,
        misclassified_predictions,
        misclassified_labels
    )


def plot_confusion_matrix(
    labels,
    predictions
):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=range(10),
        cmap="Blues"
    )

    plt.title(
        "MNIST Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "mnist_confusion_matrix.png"
    )

    plt.close()


def plot_misclassified_examples(
    images,
    predictions,
    labels
):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(8, 7)
    )

    for axis, image, prediction, label in zip(
        axes.flat,
        images,
        predictions,
        labels
    ):

        axis.imshow(
            image.squeeze(),
            cmap="gray"
        )

        axis.set_title(
            f"True: {label} | Pred: {prediction}"
        )

        axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "mnist_misclassified_examples.png"
    )

    plt.close()


def main():

    device = get_device()

    print(
        f"Using device: {device}"
    )

    test_loader = load_test_data()

    model = load_model(device)

    (
        predictions,
        labels,
        incorrect_images,
        incorrect_predictions,
        incorrect_labels
    ) = evaluate_model(
        model,
        test_loader,
        device
    )

    correct = sum(
        prediction == label
        for prediction, label
        in zip(predictions, labels)
    )

    accuracy = correct / len(labels)

    print(
        f"Test Accuracy: {accuracy:.2%}"
    )

    print(
        f"Correct Predictions: {correct}"
    )

    print(
        f"Incorrect Predictions: "
        f"{len(labels) - correct}"
    )

    plot_confusion_matrix(
        labels,
        predictions
    )

    plot_misclassified_examples(
        incorrect_images,
        incorrect_predictions,
        incorrect_labels
    )

    print(
        "\nEvaluation figures saved."
    )


if __name__ == "__main__":
    main()