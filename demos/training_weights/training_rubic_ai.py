import os
import platform
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ======================
# CONFIGURATION (edit)
# ======================
features_path = "densenet201_n_stories.pt"
train_dir = os.path.join("stories", "n_stories_train")
test_dir  = os.path.join("stories", "n_stories_test")
save_path = os.path.join("stories", "densenet201_example_feature.pt")

num_classes = 6
IMAGE_SIZE  = (256, 180)   # (H, W)
BATCH_SIZE_TRAIN = 32      # smaller batch helps CPU
BATCH_SIZE_TEST  = 64
PATIENCE   = 3
MAX_EPOCHS = 5

# ======================
# FORCE CPU + SPEED HINTS
# ======================
LR         = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
device = torch.device("cpu")  # force CPU
torch.set_num_threads(min(8, os.cpu_count() or 1))  # avoid oversubscription

IS_WINDOWS = platform.system().lower().startswith("win")
NUM_WORKERS = 0  # safest on Windows/Spyder
PIN_MEMORY = False
PERSISTENT_WORKERS = False

# ======================
# REPRODUCIBILITY
# ======================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# ======================
# TRANSFORMS & DATA
# ======================
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform=transform)
class_names = train_dataset.classes

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_TEST,  # batched eval is faster on CPU
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS
)

# ======================
# CHECKPOINT LOADER (robust to prefixes/wrappers)
# ======================
def _unwrap_state_dict(raw):
    """Unwrap common checkpoint wrappers."""
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
    return raw

def _strip_prefix(state, prefix):
    plen = len(prefix)
    return { (k[plen:] if k.startswith(prefix) else k): v for k, v in state.items() }

def _load_features_state(features_module: nn.Module, ckpt_path: str):
    """
    Loads a variety of DenseNet checkpoints into features_module:
    - full model state_dict with 'features.' prefix
    - DataParallel with 'module.' / 'module.features.' prefixes
    - features-only dict (no prefix)
    - checkpoints wrapped as {'state_dict': ...} or {'model': ...}
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(raw)

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict.")

    # Strip DataParallel first
    if any(k.startswith("module.") for k in state.keys()):
        state = _strip_prefix(state, "module.")

    # If it's a full model, keep only 'features.' keys and strip the prefix
    if any(k.startswith("features.") for k in state.keys()):
        state = { k.replace("features.", "", 1): v
                  for k, v in state.items() if k.startswith("features.") }

    # If there are still unexpected prefixes, try to intersect with expected keys
    expected_keys = set(features_module.state_dict().keys())
    # Filter to only keys present in features
    filtered = { k: v for k, v in state.items() if k in expected_keys }

    # If filtering killed everything, try last-resort remap for common case
    if not filtered:
        # Sometimes checkpoints have full-model keys like 'conv0.weight' missing because they used different arch
        # Fail fast with a clearer message
        sample_keys = list(state.keys())[:10]
        raise RuntimeError(
            "Could not map checkpoint keys to DenseNet201.features. "
            f"Sample ckpt keys: {sample_keys}"
        )

    missing, unexpected = features_module.load_state_dict(filtered, strict=False)
    if missing:
        print("[WARN] Missing feature keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] Unexpected feature keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

# ======================
# MODEL
# ======================
def build_model(num_classes: int, features_path: str):
    model = models.densenet201(weights=None)

    if not (features_path and os.path.exists(features_path)):
        raise FileNotFoundError(f"Feature weights not found at: {features_path}")

    # Load backbone features robustly
    _load_features_state(model.features, features_path)

    # Freeze backbone
    for p in model.features.parameters():
        p.requires_grad = False

    # Replace classifier: DenseNet -> Linear(in_features -> num_classes)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)

    return model.to(device)

model_4class = build_model(num_classes=num_classes, features_path=features_path)

# ======================
# TRAINING SETUP
# ======================
criterion = nn.CrossEntropyLoss()  # expects raw logits
optimizer = optim.Adam(
    model_4class.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)

# ======================
# TRAIN WITH EARLY STOPPING (on train loss)
# ======================
def train_model_with_early_stopping(
    model, loader, criterion, optimizer, max_epochs=5, patience=3
):
    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    loss_hist, acc_hist = [], []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_acc)

        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # Early stopping on lowest loss
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            epochs_no_improve = 0
            # keep a CPU copy
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            print(f"No improvement in loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return loss_hist, acc_hist

loss_list, acc_list = train_model_with_early_stopping(
    model_4class,
    train_loader,
    criterion,
    optimizer,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE
)

# ======================
# PLOTS
# ======================
plt.figure()
plt.plot(loss_list, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(acc_list, label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ======================
# EVALUATION + CONFUSION MATRIX
# ======================
def evaluate_model(model, loader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=False)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(list(preds))
            all_labels.extend(list(labels.numpy()))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

evaluate_model(model_4class, test_loader, class_names)

# ======================
# SAVE (state dict)
# ======================
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model_4class.state_dict(), save_path)
print(f"Model saved to: {save_path}")
