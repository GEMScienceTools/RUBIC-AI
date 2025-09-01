import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configuration
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 16
NUM_WORKERS = 0  # CPU systems often do better with 0 or 2 workers
EPOCHS = 10
PATIENCE = 3
MODEL_SAVE_PATH = "densenet201_material.pt"

device = torch.device("cpu")  # Force CPU usage

# Data transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder("train", transform=transform)
test_dataset = datasets.ImageFolder("test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)

class_names = train_dataset.classes

# Load pretrained model
model = models.densenet201(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_features, len(class_names)),
    nn.LogSoftmax(dim=1)
)
model.to(device)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Training loop
def train_model(model, loader, criterion, optimizer, scheduler, num_epochs=10, patience=3):
    best_loss = float('inf')
    epochs_no_improve = 0
    train_loss, train_acc = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        scheduler.step(epoch_loss)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Time: {time.time()-start_time:.2f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    return train_loss, train_acc

# Train
train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler,
                                    num_epochs=EPOCHS, patience=PATIENCE)

# Plotting
def plot_metrics(train_loss, train_acc):
    # Plot Train Loss
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Train Accuracy
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Call function
plot_metrics(train_loss, train_acc)

# Evaluation
def evaluate(model, loader):
    model.eval()
    true, preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, pred = torch.max(output, 1)
            preds.extend(pred.cpu().numpy())
            true.extend(y.cpu().numpy())

    return np.array(true), np.array(preds)

y_true, y_pred = evaluate(model, test_loader)

# Confusion matrix
def show_confusion_matrix(cm, labels, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.show()

cm = confusion_matrix(y_true, y_pred)
show_confusion_matrix(cm, class_names)
show_confusion_matrix(cm, class_names, normalize=True)
