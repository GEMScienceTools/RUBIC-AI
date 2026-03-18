"""
dl_prediction_models.py
=======================
Building attribute prediction using pre-trained DenseNet201 and ConvNeXt-Tiny models.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from PyQt5.QtWidgets import QMessageBox

# ─────────────────────────────────────────────────────────────────────────────
# Shared infrastructure
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache: weights_path -> {"model", "device", "transform"}
_MODEL_CACHE: dict = {}

# Single shared transform (all models use identical pre-processing)
_SHARED_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _resolve_image(image_path, insp_method: int, box_id):
    """
    Convert image_path to a PIL RGB image based on inspection method.
    Centralises the branching logic shared by every prediction function.
    """
    if insp_method != 2:
        # image_path is a NumPy array
        return Image.fromarray(image_path).convert("RGB")
    else:
        if box_id is not None:
            # Cropped bounding-box NumPy array
            return Image.fromarray(image_path).convert("RGB")
        else:
            # File path on disk
            return Image.open(image_path).convert("RGB")


def _run_inference(model, device, image, return_probs: bool = False):
    """
    Pre-process a PIL image and run a forward pass.
    Returns argmax class index (int) or softmax probabilities (list of float).
    """
    tensor = _SHARED_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type,
                            enabled=(device.type == "cuda")):
            output = model(tensor)

    if return_probs:
        return torch.softmax(output, dim=1).squeeze().tolist()
    return torch.argmax(output, dim=1).item()


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders  (one per architecture / classifier shape)
# ─────────────────────────────────────────────────────────────────────────────

def _get_densenet_bundle(weights_path: str, num_classes: int) -> dict:
    """
    Load (or return cached) a DenseNet201 model with `num_classes` outputs.
    Architecture:  classifier = Sequential(Dropout(0.2), Linear(1920, num_classes))
    """
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet201(weights=None)
    num_features = model.classifier.in_features  # 1920
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, num_classes),
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # Uncomment for PyTorch >= 2.0 (adds ~20-40 % throughput after warm-up):
    # model = torch.compile(model)

    bundle = {"model": model, "device": device}
    _MODEL_CACHE[weights_path] = bundle
    return bundle


def _get_convnext_bundle(weights_path: str, num_classes: int) -> dict:
    """
    Load (or return cached) a ConvNeXt-Tiny model with `num_classes` outputs.
    Architecture:  classifier[2] = Sequential(Dropout(0.2), Linear(768, num_classes))
    """
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features  # 768
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # model = torch.compile(model)  # PyTorch >= 2.0

    bundle = {"model": model, "device": device}
    _MODEL_CACHE[weights_path] = bundle
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Public warm-up helper  (call once at app start to pre-load all models)
# ─────────────────────────────────────────────────────────────────────────────

def warm_up_all_models() -> None:
    """
    Pre-load every model into the cache so the first real prediction call
    is not penalised by disk I/O + weight loading.
    Safe to call at application start-up (e.g. after the main window opens).
    """
    print("[dl_prediction_models] Pre-loading all models …")
    _get_densenet_bundle("dl_weights/densenet201_material.pt",    num_classes=8)
    _get_densenet_bundle("dl_weights/densenet201_llrs.pt",        num_classes=6)
    _get_densenet_bundle("dl_weights/densenet201_block_position.pt", num_classes=4)
    _get_densenet_bundle("dl_weights/densenet201_roof_shape.pt",  num_classes=5)
    _get_densenet_bundle("dl_weights/densenet201_roof_material.pt", num_classes=3)
    _get_convnext_bundle("dl_weights/convnext_tiny_code_level.pt", num_classes=4)
    _get_convnext_bundle("dl_weights/convnext_tiny_n_stories.pt", num_classes=9)
    _get_convnext_bundle("dl_weights/convnext_tiny_occupancy.pt", num_classes=4)
    print("[dl_prediction_models] All models loaded and ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction functions
# ─────────────────────────────────────────────────────────────────────────────

############ Material prediction ################
def predict_material_img(image_path, insp_method: int, box_id, self,
                         return_probs: bool = False):
    """
    Predict the wall construction material of a building.
    Model : DenseNet201 — 8 classes
    Weights: dl_weights/densenet201_material.pt
    """
    bundle = _get_densenet_bundle("dl_weights/densenet201_material.pt",
                                  num_classes=8)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_material_img] Error: {e}")
        QMessageBox.warning(
            self,
            "Image Error",
            f"No Street View image found or no building detected for ID: {box_id}",
        )
        return None


############ LLRS prediction ################
def predict_llrs_img(image_path, insp_method: int, box_id, self,
                     return_probs: bool = False):
    """
    Predict the lateral load-resisting system (LLRS) of a building.
    Model : DenseNet201 — 6 classes
    Weights: dl_weights/densenet201_llrs.pt
    """
    bundle = _get_densenet_bundle("dl_weights/densenet201_llrs.pt",
                                  num_classes=6)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_llrs_img] Error: {e}")
        return None


############ Block Position prediction ################
def predict_block_position_img(image_path, insp_method: int, box_id, self,
                               return_probs: bool = False):
    """
    Predict the block position of a building.
    Model : DenseNet201 — 4 classes
    Weights: dl_weights/densenet201_block_position.pt
    """
    bundle = _get_densenet_bundle("dl_weights/densenet201_block_position.pt",
                                  num_classes=4)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_block_position_img] Error: {e}")
        return None


############ Roof Shape prediction ################
def predict_roof_shape_img(image_path, insp_method: int, box_id, self,
                           return_probs: bool = False):
    """
    Predict the roof shape of a building.
    Model : DenseNet201 — 5 classes
    Weights: dl_weights/densenet201_roof_shape.pt
    """
    bundle = _get_densenet_bundle("dl_weights/densenet201_roof_shape.pt",
                                  num_classes=5)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_roof_shape_img] Error: {e}")
        return None


############ Roof Material prediction ################
def predict_roof_material_img(image_path, insp_method: int, box_id, self,
                              return_probs: bool = False):
    """
    Predict the roof material of a building.
    Model : DenseNet201 — 3 classes
    Weights: dl_weights/densenet201_roof_material.pt
    """
    bundle = _get_densenet_bundle("dl_weights/densenet201_roof_material.pt",
                                  num_classes=3)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_roof_material_img] Error: {e}")
        return None


############ Code Level prediction ################
def predict_code_img(image_path, insp_method: int, box_id, self,
                     return_probs: bool = False):
    """
    Predict the design code level of a building.
    Model : ConvNeXt-Tiny — 4 classes
    Weights: dl_weights/convnext_tiny_code_level.pt
    """
    bundle = _get_convnext_bundle("dl_weights/convnext_tiny_code_level.pt",
                                  num_classes=4)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_code_img] Error: {e}")
        return None


############ Number of Stories prediction ################
def predict_n_stories_img(image_path, insp_method: int, box_id, self,
                          return_probs: bool = False):
    """
    Predict the number of stories of a building.
    Model : ConvNeXt-Tiny — 9 classes
    Weights: dl_weights/convnext_tiny_n_stories.pt
    """
    bundle = _get_convnext_bundle("dl_weights/convnext_tiny_n_stories.pt",
                                  num_classes=9)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_n_stories_img] Error: {e}")
        return None


############ Occupancy prediction ################
def predict_occupancy_img(image_path, insp_method: int, box_id, self,
                          return_probs: bool = False):
    """
    Predict the occupancy type of a building.
    Model : ConvNeXt-Tiny — 4 classes
    Weights: dl_weights/convnext_tiny_occupancy.pt
    """
    bundle = _get_convnext_bundle("dl_weights/convnext_tiny_occupancy.pt",
                                  num_classes=4)
    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(bundle["model"], bundle["device"],
                              image, return_probs)
    except Exception as e:
        print(f"[predict_occupancy_img] Error: {e}")
        return None
