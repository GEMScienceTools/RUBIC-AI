# Training for Image Classification

The demos in this folder are designed for training a Convolutional Neural Network (CNN) model for image classification using PyTorch and transfer learning with DenseNet-201. The model is designed to classify building materials from images.

## Overview

The script implements a transfer learning approach using a pre-trained DenseNet-201 model from ImageNet, fine-tuned for six models:
 1. Material (predominant building material)
 2. Lateral load resisting system (LLRS)
 3. Heights (number of storeys)
 4. Block position
 5. Occupancy
 6. Code level
 7. Roof shape
 8. Roof material

## Dataset Structure

For any of the trained models, the training script expects the following directory structure:

```text
├── material_example.py
├── train/
│   ├── CR/     # Concrete Reinforced
│   ├── MCF/    # Masonry Confined
│   ├── MR/     # Masonry Reinforced
│   └── MUR/    # Masonry Unreinforced
└── test/
    ├── CR/
    ├── MCF/
    ├── MR/
    └── MUR/
```

Each class folder should contain the corresponding training/testing images.


## Configuration Parameters

### Model Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `IMAGE_SIZE` | `(96, 96)` | Input image dimensions (width, height) in pixels |
| `BATCH_SIZE` | `16` | Number of images processed in each batch |
| `NUM_WORKERS` | `0` | Number of worker processes for data loading (0 for CPU systems) |
| `EPOCHS` | `10` | Maximum number of training epochs |
| `PATIENCE` | `3` | Early stopping patience (epochs without improvement) |
| `MODEL_SAVE_PATH` | `"densenet201_llrs.pt"` | Path to save the trained model weights |


### Data Preprocessing

The script uses the following image transformations:

| Transform | Parameters | Description |
|-----------|------------|-------------|
| `Resize` | `IMAGE_SIZE` | Resizes images to specified dimensions |
| `ToTensor` | - | Converts PIL Image to PyTorch tensor |
| `Normalize` | `mean=[0.485, 0.456, 0.406]` `std=[0.229, 0.224, 0.225]` | ImageNet normalization values |

### Training Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Learning Rate | `1e-3` | Initial learning rate for Adam optimizer |
| Loss Function | `NLLLoss` | Negative Log Likelihood Loss |
| Optimizer | `Adam` | Adam optimizer for classifier parameters |
| Scheduler | `ReduceLROnPlateau` | Reduces learning rate when loss plateaus |
| Scheduler Factor | `0.1` | Factor by which learning rate is reduced |
| Scheduler Patience | `3` | Epochs to wait before reducing learning rate |

## Model Architecture

- **Base Model**: DenseNet-201 pre-trained on ImageNet
- **Transfer Learning**: All pre-trained layers are frozen (`requires_grad=False`)
- **Custom Classifier**:
  - Flatten layer
  - Linear layer (input: DenseNet features, output: number of classes)
  - LogSoftmax activation

## Usage

### Prerequisites

Install required dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

### Running the Training

1. **Prepare your dataset**: Organize images in the train/test folder structure shown above
2. **Modify parameters** (optional): Edit the configuration section at the top of the script
3. **Run training**:

   ```bash
   python material_example.py
   ```

### Customizing Parameters

You can modify the following parameters in the script:

```python
# Example customization
IMAGE_SIZE = (128, 128)  # Larger input images
BATCH_SIZE = 32          # Larger batch size
EPOCHS = 20              # More training epochs
PATIENCE = 5             # More patience for early stopping
```

## Output

The script generates:

1. **Model weights**: Saved to `MODEL_SAVE_PATH` (best model based on validation loss)
2. **Training plots**:
   - Training loss over epochs
   - Training accuracy over epochs
3. **Confusion matrices**:
   - Raw confusion matrix
   - Normalized confusion matrix
4. **Console output**: Epoch-by-epoch training progress

## Model Performance Monitoring

The script includes several monitoring features:

- **Early Stopping**: Prevents overfitting by stopping when validation loss doesn't improve
- **Learning Rate Scheduling**: Automatically reduces learning rate when loss plateaus
- **Real-time Progress**: Shows loss, accuracy, and time per epoch
- **Model Checkpointing**: Saves best model automatically

## Building Material Classes

The model is trained to classify four types of building materials:

- **CR**: Concrete Reinforced
- **MCF**: Masonry Confined
- **MR**: Masonry Reinforced
- **MUR**: Masonry Unreinforced

## Tips for Better Performance

1. **Increase image size** for better feature extraction (trade-off with computation time)
2. **Adjust batch size** based on available memory
3. **Use GPU** if available by changing `device = torch.device("cuda")`
4. **Increase epochs** for more complex datasets
5. **Data augmentation** can be added to transforms for better generalization

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce `BATCH_SIZE` or `IMAGE_SIZE`
2. **Slow training**: Increase `NUM_WORKERS` (if using CPU) or use GPU
3. **Poor performance**: Check data quality, increase training data, or adjust learning rate
4. **Early stopping too early**: Increase `PATIENCE` parameter

### System Requirements

- **RAM**: Minimum 8GB recommended
- **Storage**: Depends on dataset size
- **Python**: 3.7+ with PyTorch installed
