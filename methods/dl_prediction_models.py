"""Predict building attributes with pretrained deep-learning models.

This module loads and caches ConvNeXt-Tiny and Swin Transformer Tiny models,
prepares input images, and exposes prediction functions used by RUBIC-AI.
"""

from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import QMessageBox
from torchvision import models

# -----------------------------------------------------------------------------
# Shared infrastructure
# -----------------------------------------------------------------------------

# Module-level cache: weights_path -> {"model", "device"}
_MODEL_CACHE: dict[str, dict[str, Any]] = {}

# All models use the same image preprocessing pipeline.
_SHARED_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _resolve_image(image_path, insp_method: int, box_id):
    """Convert an image input to a PIL RGB image.

    The input is interpreted as a NumPy array or a file path according to the
    inspection method and bounding-box identifier.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier. A non-``None`` value indicates that
        ``image_path`` contains an in-memory cropped image.

    Returns
    -------
    PIL.Image.Image
        Image converted to RGB format.
    """
    if insp_method != 2 or box_id is not None:
        return Image.fromarray(image_path).convert("RGB")

    return Image.open(image_path).convert("RGB")


def _run_inference(model, device, image, return_probs: bool = False):
    """Preprocess an image and run a model inference pass.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    device : torch.device
        Device on which inference is performed.
    image : PIL.Image.Image
        RGB image to classify.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int or list[float]
        Predicted class index or softmax probabilities.
    """
    tensor = _SHARED_TRANSFORM(image).unsqueeze(0).to(device)

    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ),
    ):
        output = model(tensor)

    if return_probs:
        return torch.softmax(output, dim=1).squeeze().tolist()

    return torch.argmax(output, dim=1).item()


# -----------------------------------------------------------------------------
# Model loaders
# -----------------------------------------------------------------------------


def _get_swin_tiny_bundle(weights_path: str, num_classes: int) -> dict[str, Any]:
    """Load or retrieve a cached Swin Transformer Tiny model.

    The classifier head contains a dropout layer followed by a linear layer
    with ``num_classes`` outputs.

    Parameters
    ----------
    weights_path : str
        Path to the trained model weights.
    num_classes : int
        Number of output classes.

    Returns
    -------
    dict[str, object]
        Dictionary containing the model and inference device.
    """
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.swin_t(weights=None)
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, num_classes),
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    bundle = {"model": model, "device": device}
    _MODEL_CACHE[weights_path] = bundle
    return bundle


def _get_convnext_bundle(weights_path: str, num_classes: int) -> dict[str, Any]:
    """Load or retrieve a cached ConvNeXt-Tiny model.

    The final classifier layer is replaced by a dropout layer followed by a
    linear layer with ``num_classes`` outputs.

    Parameters
    ----------
    weights_path : str
        Path to the trained model weights.
    num_classes : int
        Number of output classes.

    Returns
    -------
    dict[str, object]
        Dictionary containing the model and inference device.
    """
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    bundle = {"model": model, "device": device}
    _MODEL_CACHE[weights_path] = bundle
    return bundle


# -----------------------------------------------------------------------------
# Public warm-up helper
# -----------------------------------------------------------------------------


def warm_up_all_models() -> None:
    """Preload all prediction models into the shared cache.

    Call this function once during application startup to avoid loading model
    weights during the first user prediction.
    """
    print("[dl_prediction_models] Preloading all models...")
    _get_convnext_bundle(
        "dl_weights/convnext_tiny_material.pt",
        num_classes=3,
    )
    _get_convnext_bundle(
        "dl_weights/convnext_tiny_llrs.pt",
        num_classes=5,
    )
    _get_swin_tiny_bundle(
        "dl_weights/swin_t_b_position.pt",
        num_classes=4,
    )
    _get_swin_tiny_bundle(
        "dl_weights/swin_t_roof_shape.pt",
        num_classes=4,
    )
    _get_swin_tiny_bundle(
        "dl_weights/swin_t_roof_material.pt",
        num_classes=3,
    )
    _get_convnext_bundle(
        "dl_weights/convnext_tiny_code.pt",
        num_classes=4,
    )
    _get_convnext_bundle(
        "dl_weights/convnext_tiny_n_stories.pt",
        num_classes=9,
    )
    _get_convnext_bundle(
        "dl_weights/convnext_tiny_occupancy.pt",
        num_classes=4,
    )
    print("[dl_prediction_models] All models are loaded and ready.")


# -----------------------------------------------------------------------------
# Prediction functions
# -----------------------------------------------------------------------------


def predict_material_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the wall construction material of a building.

    The function uses a three-class ConvNeXt-Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget used to display warning messages.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_convnext_bundle(
        "dl_weights/convnext_tiny_material.pt",
        num_classes=3,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_material_img] Error: {error}")
        QMessageBox.warning(
            self,
            "Image Error",
            (
                "No Street View image was found, or no building was "
                f"detected for ID: {box_id}"
            ),
        )
        return None


def predict_llrs_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the lateral load-resisting system of a building.

    The function uses a five-class ConvNeXt-Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget retained for compatibility with the GUI interface.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_convnext_bundle(
        "dl_weights/convnext_tiny_llrs.pt",
        num_classes=5,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_llrs_img] Error: {error}")
        return None


def predict_block_position_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the block position of a building.

    The function uses a four-class Swin Transformer Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget retained for compatibility with the GUI interface.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_swin_tiny_bundle(
        "dl_weights/swin_t_b_position.pt",
        num_classes=4,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_block_position_img] Error: {error}")
        return None


def predict_roof_shape_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the roof shape of a building.

    The function uses a four-class Swin Transformer Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget retained for compatibility with the GUI interface.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_swin_tiny_bundle(
        "dl_weights/swin_t_roof_shape.pt",
        num_classes=4,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_roof_shape_img] Error: {error}")
        return None


def predict_roof_material_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the roof material of a building.

    The function uses a three-class Swin Transformer Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget retained for compatibility with the GUI interface.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_swin_tiny_bundle(
        "dl_weights/swin_t_roof_material.pt",
        num_classes=3,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_roof_material_img] Error: {error}")
        return None


def predict_code_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the design code level of a building.

    The function uses a four-class ConvNeXt-Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget retained for compatibility with the GUI interface.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_convnext_bundle(
        "dl_weights/convnext_tiny_code.pt",
        num_classes=4,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_code_img] Error: {error}")
        return None


def predict_n_stories_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the number of stories of a building.

    The function uses a nine-class ConvNeXt-Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget retained for compatibility with the GUI interface.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_convnext_bundle(
        "dl_weights/convnext_tiny_n_stories.pt",
        num_classes=9,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_n_stories_img] Error: {error}")
        return None


def predict_occupancy_img(
    image_path,
    insp_method: int,
    box_id,
    self,
    return_probs: bool = False,
):
    """Predict the occupancy type of a building.

    The function uses a four-class ConvNeXt-Tiny model.

    Parameters
    ----------
    image_path : str or numpy.ndarray
        Image file path or in-memory image array.
    insp_method : int
        Inspection-method identifier.
    box_id : object or None
        Bounding-box identifier.
    self : QWidget
        Parent widget retained for compatibility with the GUI interface.
    return_probs : bool, optional
        Return class probabilities instead of the predicted class index.

    Returns
    -------
    int, list[float], or None
        Prediction result, or ``None`` if inference fails.
    """
    bundle = _get_convnext_bundle(
        "dl_weights/convnext_tiny_occupancy.pt",
        num_classes=4,
    )

    try:
        image = _resolve_image(image_path, insp_method, box_id)
        return _run_inference(
            bundle["model"],
            bundle["device"],
            image,
            return_probs,
        )
    except Exception as error:
        print(f"[predict_occupancy_img] Error: {error}")
        return None
