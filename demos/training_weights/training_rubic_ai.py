"""Train and evaluate a DenseNet201 classifier using frozen features.

The script loads DenseNet201 feature weights from an existing checkpoint,
freezes the convolutional backbone, trains a new classification layer, plots
training metrics, evaluates the model on a test dataset, and saves the best
model state dictionary.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Paths
FEATURES_PATH = Path("densenet201_n_stories.pt")
TRAIN_DIR = Path("stories") / "n_stories_train"
TEST_DIR = Path("stories") / "n_stories_test"
MODEL_SAVE_PATH = Path("stories") / "densenet201_example_feature.pt"

# Training configuration
NUM_CLASSES = 6
IMAGE_SIZE = (256, 180)  # Height, width
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
MAX_EPOCHS = 5
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
MIN_LOSS_IMPROVEMENT = 1e-6

# CPU configuration
DEVICE = torch.device("cpu")
NUM_WORKERS = 0
PIN_MEMORY = False
PERSISTENT_WORKERS = NUM_WORKERS > 0
MAX_CPU_THREADS = 8

# ImageNet normalization values
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_data_loaders(
    train_dir: Path,
    test_dir: Path,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create training and test data loaders.

    Args:
        train_dir: Directory containing the training class folders.
        test_dir: Directory containing the test class folders.

    Returns
    -------
        The training loader, test loader, and ordered class names.

    Raises
    ------
        FileNotFoundError: If either dataset directory does not exist.
        ValueError: If the training and test datasets have different classes.
    """
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory was not found: {train_dir}"
        )

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory was not found: {test_dir}")

    image_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=image_transform,
    )
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=image_transform,
    )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "The training and test datasets must contain the same class "
            "folders in the same order."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )

    return train_loader, test_loader, train_dataset.classes


def _unwrap_state_dict(checkpoint: Any) -> Any:
    """Extract a state dictionary from common checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model"):
        wrapped_state = checkpoint.get(key)
        if isinstance(wrapped_state, dict):
            return wrapped_state

    return checkpoint


def _strip_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Remove a prefix from state-dictionary keys when present."""
    prefix_length = len(prefix)
    return {
        (
            key[prefix_length:]
            if key.startswith(prefix)
            else key
        ): value
        for key, value in state_dict.items()
    }


def _load_feature_weights(
    feature_module: nn.Module,
    checkpoint_path: Path,
) -> None:
    """Load compatible DenseNet feature weights from a checkpoint.

    The loader supports full-model state dictionaries, feature-only state
    dictionaries, DataParallel prefixes, and checkpoints wrapped under the
    ``state_dict`` or ``model`` keys.

    Args:
        feature_module: DenseNet feature module receiving the weights.
        checkpoint_path: Path to the saved checkpoint.

    Raises
    ------
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If no compatible feature weights can be identified.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Feature checkpoint was not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = _unwrap_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "The checkpoint does not contain a valid state dictionary."
        )

    if any(key.startswith("module.") for key in state_dict):
        state_dict = _strip_prefix(state_dict, "module.")

    if any(key.startswith("features.") for key in state_dict):
        state_dict = {
            key.removeprefix("features."): value
            for key, value in state_dict.items()
            if key.startswith("features.")
        }

    expected_keys = set(feature_module.state_dict())
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in expected_keys
    }

    if not compatible_state:
        sample_keys = list(state_dict)[:10]
        raise RuntimeError(
            "No checkpoint parameters could be mapped to "
            "DenseNet201.features. Sample checkpoint keys: "
            f"{sample_keys}"
        )

    missing_keys, unexpected_keys = feature_module.load_state_dict(
        compatible_state,
        strict=False,
    )

    if missing_keys:
        preview = missing_keys[:10]
        suffix = " ..." if len(missing_keys) > 10 else ""
        print(f"Warning: missing feature keys: {preview}{suffix}")

    if unexpected_keys:
        preview = unexpected_keys[:10]
        suffix = " ..." if len(unexpected_keys) > 10 else ""
        print(f"Warning: unexpected feature keys: {preview}{suffix}")


def build_model(
    number_of_classes: int,
    checkpoint_path: Path,
) -> nn.Module:
    """Build a DenseNet201 model with a frozen feature extractor."""
    model = models.densenet201(weights=None)
    _load_feature_weights(model.features, checkpoint_path)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier.in_features
    model.classifier = nn.Linear(input_features, number_of_classes)

    return model.to(DEVICE)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    max_epochs: int,
    patience: int,
) -> tuple[list[float], list[float]]:
    """Train the classifier with early stopping based on training loss."""
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        sample_count = 0

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
            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("The training data loader contains no images.")

        epoch_loss = running_loss / sample_count
        epoch_accuracy = correct_predictions / sample_count
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

        if epoch_loss < best_loss - MIN_LOSS_IMPROVEMENT:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            print(
                "No loss improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return loss_history, accuracy_history


def plot_training_metrics(
    loss_history: list[float],
    accuracy_history: list[float],
) -> None:
    """Plot the training loss and accuracy histories."""
    plt.figure()
    plt.plot(loss_history, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(accuracy_history, label="Training accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
) -> np.ndarray:
    """Evaluate the model and display a raw confusion matrix."""
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    if not all_labels:
        raise RuntimeError("The test data loader contains no images.")

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=labels,
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return matrix


def save_model(model: nn.Module, save_path: Path) -> None:
    """Save the trained model state dictionary."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


def main() -> None:
    """Run the complete training and evaluation workflow."""
    torch.set_num_threads(min(MAX_CPU_THREADS, os.cpu_count() or 1))
    set_seed(SEED)

    train_loader, test_loader, class_names = create_data_loaders(
        TRAIN_DIR,
        TEST_DIR,
    )

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"NUM_CLASSES is {NUM_CLASSES}, but the training dataset contains "
            f"{len(class_names)} classes: {class_names}"
        )

    model = build_model(NUM_CLASSES, FEATURES_PATH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_history, accuracy_history = train_model(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )

    plot_training_metrics(loss_history, accuracy_history)
    evaluate_model(model, test_loader, class_names)
    save_model(model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
