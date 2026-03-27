# -*- coding: utf-8 -*-
"""
DenseNet201 training with METADATA fusion (country/city) - Windows friendly.

This is the FULL adjusted script that removes sklearn OneHotEncoder/ColumnTransformer
(to avoid the np.isnan TypeError) and replaces it with a robust manual one-hot encoder.

Folder structure (ImageFolder style):
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg

Metadata CSV (train/val/test) minimal columns:
  ID,country,city

- ID must match the image filename basename (e.g., IMG_0001.jpg).
"""

import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix  # safe; does not cause your error


# ============================================================
#        LIMIT CPU USAGE (tune for your machine)
# ============================================================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
torch.set_num_threads(6)
torch.set_num_interop_threads(2)


# ============================================================
#                  CONFIG
# ============================================================

@dataclass
class Config:
    # Image folders (ImageFolder)
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"

    # Metadata CSVs
    meta_train_csv: Optional[str] = "train_data.csv"
    meta_val_csv: Optional[str] = "val_data.csv"
    meta_test_csv: Optional[str] = "test_data.csv"

    # Column in CSV that matches image basename (filename)
    meta_id_col: str = "ID"

    # Metadata columns used
    meta_cols: Tuple[str, ...] = ("country", "city")

    # Training params
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 32
    num_workers: int = 0            # Windows-safe default
    max_epochs: int = 3
    patience: int = 20
    lr: float = 1e-4

    # Output folders
    weights_dir: str = "New_weights_metadata"
    perf_dir: str = "model_performance_metadata"

    # Fusion behavior
    fusion_mode: str = "concat"     # "concat" (recommended) or "add_logits"
    meta_hidden: int = 128
    dropout: float = 0.5

    # Missing metadata handling
    allow_missing_metadata: bool = True


CFG = Config()


# ============================================================
#                  PLOTTING / EVAL UTILITIES
# ============================================================

def save_training_curves(train_loss, train_acc, val_loss, val_acc, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_vs_epoch_densenet201_metadata.png"))
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_epoch_densenet201_metadata.png"))
    plt.close()


def save_confusion_matrix(cm, labels, normalize, output_path):
    if normalize:
        cm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
        cm = np.round(cm, 2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt="g",
        cmap="Blues",
        annot_kws={"size": 10}
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for inputs, labels, meta in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)

        outputs = model(inputs, meta)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


# ============================================================
#               METADATA ENCODING (NO SKLEARN)
# ============================================================

def _clean_meta_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize ID to basename
    df[CFG.meta_id_col] = df[CFG.meta_id_col].astype(str).apply(lambda x: os.path.basename(x))

    # Force meta columns to clean strings (prevents mixed types and NaN issues)
    for c in CFG.meta_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in metadata CSV")
        df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()

    return df


def _read_meta_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _clean_meta_df(df)


def make_meta_lookup(df_meta: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for _, row in df_meta.iterrows():
        lookup[str(row[CFG.meta_id_col])] = row.to_dict()
    return lookup


class ManualOneHotMeta:
    """
    Manual one-hot encoder fitted on TRAIN only.
    - Each column gets its own vocabulary + an UNKNOWN bucket (always present).
    - encode(row_dict) -> np.float32 vector of shape (meta_dim,)
    """
    def __init__(self, cols: Tuple[str, ...]):
        self.cols = cols
        self.vocab: Dict[str, List[str]] = {}
        self.index: Dict[str, Dict[str, int]] = {}
        self.offsets: Dict[str, int] = {}
        self.meta_dim: int = 0

    def fit(self, df_train: pd.DataFrame):
        offset = 0
        for col in self.cols:
            cats = sorted(df_train[col].unique().tolist())
            if "UNKNOWN" not in cats:
                cats.append("UNKNOWN")

            self.vocab[col] = cats
            self.index[col] = {v: i for i, v in enumerate(cats)}
            self.offsets[col] = offset
            offset += len(cats)

        self.meta_dim = offset

    def encode(self, row: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros((self.meta_dim,), dtype=np.float32)
        for col in self.cols:
            v = row.get(col, "UNKNOWN")
            if v is None:
                v = "UNKNOWN"
            v = str(v).strip()
            i = self.index[col].get(v, self.index[col]["UNKNOWN"])
            vec[self.offsets[col] + i] = 1.0
        return vec


class ImageFolderWithMetadata(torch.utils.data.Dataset):
    """
    Wrap ImageFolder so each item returns: image, label, meta_vector
    """
    def __init__(
        self,
        imagefolder_ds: datasets.ImageFolder,
        meta_lookup: Optional[Dict[str, Dict[str, Any]]],
        meta_encoder: Optional[ManualOneHotMeta],
        allow_missing: bool = True,
    ):
        self.ds = imagefolder_ds
        self.meta_lookup = meta_lookup
        self.meta_encoder = meta_encoder
        self.allow_missing = allow_missing
        self.meta_dim = meta_encoder.meta_dim if meta_encoder is not None else 1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]  # ImageFolder returns (image_tensor, class_index)

        # No metadata -> return zeros
        if self.meta_lookup is None or self.meta_encoder is None:
            meta_vec = torch.zeros(self.meta_dim, dtype=torch.float32)
            return x, y, meta_vec

        path, _ = self.ds.samples[idx]
        fname = os.path.basename(path)

        row = self.meta_lookup.get(fname, None)
        if row is None:
            if not self.allow_missing:
                raise KeyError(f"Metadata not found for image: {fname}")
            meta_vec = torch.zeros(self.meta_dim, dtype=torch.float32)
            return x, y, meta_vec

        enc = self.meta_encoder.encode(row)       # np.float32
        meta_vec = torch.from_numpy(enc)          # torch.float32
        return x, y, meta_vec


# ============================================================
#                  MODEL DEFINITION
# ============================================================

class DenseNet201WithMetadata(nn.Module):
    """
    Two fusion modes:
      - concat (recommended): logits = classifier([img_feat, meta_feat])
      - add_logits: logits = img_logits + meta_logits
    """
    def __init__(
        self,
        num_classes: int,
        meta_dim: int,
        fusion_mode: str = "concat",
        meta_hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode.lower().strip()

        # Base DenseNet201
        self.backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

        # Freeze all
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last block (fine-tune)
        for p in self.backbone.features.denseblock4.parameters():
            p.requires_grad = True
        for p in self.backbone.features.norm5.parameters():
            p.requires_grad = True

        img_feat_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        # Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.fusion_mode == "concat":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(img_feat_dim + meta_hidden, num_classes),
            )
        elif self.fusion_mode == "add_logits":
            self.img_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(img_feat_dim, num_classes),
            )
            self.meta_head = nn.Sequential(
                nn.Linear(meta_hidden, num_classes),
            )
        else:
            raise ValueError("fusion_mode must be 'concat' or 'add_logits'")

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(x)   # (B, img_feat_dim)
        meta_feat = self.meta_mlp(meta)

        if self.fusion_mode == "concat":
            feat = torch.cat([img_feat, meta_feat], dim=1)
            return self.classifier(feat)

        # add_logits
        return self.img_head(img_feat) + self.meta_head(meta_feat)


# ============================================================
#                   TRAINING LOOP
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, max_epochs, patience):

    os.makedirs(CFG.weights_dir, exist_ok=True)
    os.makedirs(CFG.perf_dir, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(max_epochs):

        # ---------------- TRAIN ----------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels, meta in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, meta)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs, meta)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels, meta in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True)

                outputs = model(inputs, meta)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"\nEpoch [{epoch+1}/{max_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        save_training_curves(train_losses, train_accs, val_losses, val_accs, CFG.perf_dir)

        if val_loss < best_val_loss:
            print("✅ Validation improved — Saving model")
            torch.save(model.state_dict(), os.path.join(CFG.weights_dir, "densenet201_with_metadata.pt"))
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹️ Early stopping triggered")
                break


# ============================================================
#                        MAIN
# ============================================================

def main():
    start_time = time.time()

    os.makedirs(CFG.perf_dir, exist_ok=True)
    os.makedirs(CFG.weights_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------- Transforms -------------------
    train_transform = transforms.Compose([
        transforms.Resize(280),
        transforms.RandomResizedCrop(CFG.image_size[0], scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(CFG.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ------------------- ImageFolder datasets -------------------
    train_base = datasets.ImageFolder(CFG.train_dir, transform=train_transform)
    val_base = datasets.ImageFolder(CFG.val_dir, transform=test_transform)
    test_base = datasets.ImageFolder(CFG.test_dir, transform=test_transform)

    class_names = train_base.classes
    print("Classes:", class_names)

    # ------------------- Class weights -------------------
    targets = [label for _, label in train_base.samples]
    class_count = Counter(targets)
    weights = torch.tensor([1.0 / class_count[i] for i in range(len(class_names))], device=device)

    # ------------------- Metadata (fit on train only) -------------------
    meta_encoder: Optional[ManualOneHotMeta] = None
    meta_train_lookup = meta_val_lookup = meta_test_lookup = None

    if CFG.meta_train_csv is not None and os.path.exists(CFG.meta_train_csv):
        meta_train_df = _read_meta_csv(CFG.meta_train_csv)

        meta_encoder = ManualOneHotMeta(CFG.meta_cols)
        meta_encoder.fit(meta_train_df)
        meta_train_lookup = make_meta_lookup(meta_train_df)

        if CFG.meta_val_csv is not None and os.path.exists(CFG.meta_val_csv):
            meta_val_df = _read_meta_csv(CFG.meta_val_csv)
            meta_val_lookup = make_meta_lookup(meta_val_df)

        if CFG.meta_test_csv is not None and os.path.exists(CFG.meta_test_csv):
            meta_test_df = _read_meta_csv(CFG.meta_test_csv)
            meta_test_lookup = make_meta_lookup(meta_test_df)

        print(f"✅ Metadata enabled. meta_dim = {meta_encoder.meta_dim} | meta_cols = {CFG.meta_cols}")
    else:
        print("⚠️ Metadata CSV not found (or disabled). Running with zero metadata vectors (all zeros).")

    # ------------------- Wrap datasets -------------------
    train_dataset = ImageFolderWithMetadata(
        train_base, meta_train_lookup, meta_encoder, allow_missing=CFG.allow_missing_metadata
    )
    val_dataset = ImageFolderWithMetadata(
        val_base, meta_val_lookup, meta_encoder, allow_missing=CFG.allow_missing_metadata
    )
    test_dataset = ImageFolderWithMetadata(
        test_base, meta_test_lookup, meta_encoder, allow_missing=CFG.allow_missing_metadata
    )

    # ------------------- DataLoaders -------------------
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=pin_mem
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=pin_mem
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=pin_mem
    )

    # ------------------- Model -------------------
    meta_dim = meta_encoder.meta_dim if meta_encoder is not None else 1

    model = DenseNet201WithMetadata(
        num_classes=len(class_names),
        meta_dim=meta_dim,
        fusion_mode=CFG.fusion_mode,
        meta_hidden=CFG.meta_hidden,
        dropout=CFG.dropout
    ).to(device)

    print("Model loaded on:", next(model.parameters()).device)

    # Train only trainable params
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=3
    )

    criterion = nn.CrossEntropyLoss(weight=weights)

    # ------------------- Train -------------------
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        CFG.max_epochs,
        CFG.patience
    )

    # ------------------- Evaluate on test -------------------
    print("\n🔍 Evaluating model on test set...")
    true_labels, pred_labels = evaluate_model(model, test_loader, device)
    cm = confusion_matrix(true_labels, pred_labels)

    save_confusion_matrix(
        cm, class_names, False,
        os.path.join(CFG.perf_dir, "confusion_matrix_raw_densenet201_metadata.png")
    )
    save_confusion_matrix(
        cm, class_names, True,
        os.path.join(CFG.perf_dir, "confusion_matrix_normalized_densenet201_metadata.png")
    )

    total_time = time.time() - start_time
    print(f"\n✅ Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"✅ Best weights saved to: {os.path.join(CFG.weights_dir, 'densenet201_with_metadata.pt')}")


if __name__ == "__main__":
    # Windows multiprocessing safety
    torch.multiprocessing.freeze_support()
    main()
