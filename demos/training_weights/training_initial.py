"""Train and evaluate a DenseNet201 image-classification model on CPU."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
SCHEDULER_PATIENCE = 3
LEARNING_RATE = 1e-3

TRAIN_DIRECTORY = Path("material/train")
TEST_DIRECTORY = Path("material/test")
MODEL_SAVE_PATH = Path("densenet201_material.pt")

DEVICE = torch.device("cpu")

Loader: TypeAlias = DataLoader
History: TypeAlias = tuple[list[float], list[float]]


def create_transform() -> transforms.Compose:
    """Create the preprocessing pipeline used by DenseNet201."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_data_loaders(
    transform: transforms.Compose,
) -> tuple[Loader, Loader, list[str]]:
    """Load the training and test datasets and create data loaders."""
    train_dataset = datasets.ImageFolder(
        TRAIN_DIRECTORY,
        transform=transform,
    )
    test_dataset = datasets.ImageFolder(
        TEST_DIRECTORY,
        transform=transform,
    )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "The training and test datasets must contain the same classes."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, test_loader, train_dataset.classes


def create_model(number_of_classes: int) -> nn.Module:
    """Create a pretrained DenseNet201 with a trainable classifier."""
    weights = models.DenseNet201_Weights.IMAGENET1K_V1
    model = models.densenet201(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    number_of_features = model.classifier.in_features
    model.classifier = nn.Linear(number_of_features, number_of_classes)

    return model.to(DEVICE)


def train_model(
    model: nn.Module,
    loader: Loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    number_of_epochs: int,
    patience: int,
) -> History:
    """Train the classifier and save the weights with the lowest loss."""
    best_loss = float("inf")
    epochs_without_improvement = 0
    training_losses: list[float] = []
    training_accuracies: list[float] = []

    for epoch_index in range(number_of_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        start_time = time.perf_counter()

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        scheduler.step(epoch_loss)

        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        elapsed_time = time.perf_counter() - start_time
        print(
            f"Epoch {epoch_index + 1}/{number_of_epochs} - "
            f"Loss: {epoch_loss:.4f} - "
            f"Accuracy: {epoch_accuracy:.4f} - "
            f"Time: {elapsed_time:.2f} s"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return training_losses, training_accuracies


def plot_training_metrics(
    training_losses: list[float],
    training_accuracies: list[float],
) -> None:
    """Plot training loss and accuracy in separate figures."""
    epochs = range(1, len(training_losses) + 1)

    plt.figure()
    plt.plot(epochs, training_losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs, training_accuracies, label="Training accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Loader,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the true and predicted class labels for a dataset."""
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())

    return np.asarray(true_labels), np.asarray(predicted_labels)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    normalize: bool = False,
) -> None:
    """Plot a raw or row-normalized confusion matrix."""
    displayed_matrix = matrix.copy()

    if normalize:
        row_totals = displayed_matrix.sum(axis=1, keepdims=True)
        displayed_matrix = np.divide(
            displayed_matrix.astype(float),
            row_totals,
            out=np.zeros_like(displayed_matrix, dtype=float),
            where=row_totals != 0,
        )

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        displayed_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12},
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the complete training and evaluation workflow."""
    transform = create_transform()
    train_loader, test_loader, class_names = create_data_loaders(transform)
    model = create_model(len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=SCHEDULER_PATIENCE,
    )

    training_losses, training_accuracies = train_model(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        number_of_epochs=NUM_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
    )

    plot_training_metrics(training_losses, training_accuracies)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    true_labels, predicted_labels = evaluate_model(model, test_loader)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    )
    plot_confusion_matrix(matrix, class_names)
    plot_confusion_matrix(matrix, class_names, normalize=True)


if __name__ == "__main__":
    main()
