"""Run building-attribute inference on locally stored images.

The script reads building coordinates and image filenames from a CSV file,
detects the main building in each image, predicts its attributes, validates
the resulting taxonomy, and writes the results to a CSV file.

Model architecture and output dimensions are validated against each
checkpoint before inference.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from PIL import Image
from torch import Tensor
from torchvision import models
from ultralytics import YOLO

RUBICAI_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPT_DIR = Path(__file__).parent.resolve()
DL_DIR = RUBICAI_ROOT / "dl_weights"

if str(RUBICAI_ROOT) not in sys.path:
    sys.path.append(str(RUBICAI_ROOT))

from methods.taxonomy import check_taxonomy  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASS = "building-xzyh"
CONFIDENCE_THRESHOLD = 0.5
GEOCODER_TIMEOUT_SECONDS = 3

DATABASE_COLUMNS = [
    "id",
    "latitude",
    "longitude",
    "country",
    "city",
    "material",
    "llrs",
    "code_level",
    "n_stories",
    "occupancy",
    "block_position",
    "roof_shape",
    "roof_material",
    "taxonomy",
    "image filename or link",
]

CLASS_NAMES = {
    "material": [
        "CR",
        "MCF",
        "MUR",
    ],
    "llrs": [
        "LDUAL",
        "LFINF",
        "LFM",
        "LWAL",
        "LWAL",
    ],
    "code_level": [
        "CDH",
        "CDL",
        "CDM",
        "CDN",
    ],
    "n_stories": [
        "10-12",
        "13+",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6-7",
        "8-9",
    ],
    "occupancy": [
        "COM",
        "IND",
        "MIX(RES;COM)",
        "RES",
    ],
    "block_position": [
        "BP1",
        "BP2",
        "BP3",
        "BPD",
    ],
    "roof_shape": [
        "RSH1",
        "RSH2",
        "RSH3",
        "RSH7",
    ],
    "roof_material": [
        "RMN",
        "RMT1",
        "RMT6",
    ],
}

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@dataclass
class PredictionModels:
    """Store all loaded building-attribute models."""

    material: nn.Module
    llrs: nn.Module
    code_level: nn.Module
    n_stories: nn.Module
    occupancy: nn.Module
    block_position: nn.Module
    roof_shape: nn.Module
    roof_material: nn.Module


@dataclass(frozen=True)
class ModelSpecification:
    """Describe one classifier and its checkpoint."""

    attribute: str
    architecture: str
    weight_path: Path


MODEL_SPECIFICATIONS = [
    ModelSpecification(
        attribute="material",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_material.pt",
    ),
    ModelSpecification(
        attribute="llrs",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_llrs.pt",
    ),
    ModelSpecification(
        attribute="code_level",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_code.pt",
    ),
    ModelSpecification(
        attribute="n_stories",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_n_stories.pt",
    ),
    ModelSpecification(
        attribute="occupancy",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_occupancy.pt",
    ),
    ModelSpecification(
        attribute="block_position",
        architecture="swin_t",
        weight_path=DL_DIR / "swin_t_b_position.pt",
    ),
    ModelSpecification(
        attribute="roof_shape",
        architecture="swin_t",
        weight_path=DL_DIR / "swin_t_roof_shape.pt",
    ),
    ModelSpecification(
        attribute="roof_material",
        architecture="swin_t",
        weight_path=DL_DIR / "swin_t_roof_material.pt",
    ),
]


def create_database(
    local_building_info: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load building information and create an empty output database."""
    footprint_data = pd.read_csv(local_building_info)
    footprint_data.columns = footprint_data.columns.str.lower().str.strip()

    required_columns = {"id", "latitude", "longitude"}
    missing_columns = required_columns.difference(footprint_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required CSV columns: {missing}")

    inspection_data = pd.DataFrame(
        np.full(
            (len(footprint_data), len(DATABASE_COLUMNS)),
            None,
            dtype=object,
        ),
        columns=DATABASE_COLUMNS,
    )

    return inspection_data, footprint_data


def load_building_detector() -> YOLO:
    """Load the YOLO building detector once."""
    weight_path = DL_DIR / "building_detector.pt"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Building-detector weights not found: {weight_path}"
        )

    return YOLO(weight_path)


def detect_building(
    image_path: Path,
    detector: YOLO,
) -> np.ndarray | None:
    """Return the highest-confidence building crop from a local image."""
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Unable to read image: {image_path}")
        return None

    try:
        result = detector.predict(
            source=str(image_path),
            device=str(DEVICE),
            verbose=False,
        )[0]
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"Object detection failed for {image_path.name}: {error}")
        return None

    best_box: np.ndarray | None = None
    best_score = 0.0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = detector.names[class_id]
            score = float(box.conf[0])

            if (
                class_name == TARGET_CLASS
                and score > CONFIDENCE_THRESHOLD
                and score > best_score
            ):
                best_score = score
                best_box = box.xyxy[0].cpu().numpy().astype(int)

    if best_box is None:
        print(f"No building detected in {image_path.name}.")
        return None

    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = best_box

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)

    cropped_image = image[y_min:y_max, x_min:x_max]
    if cropped_image.size == 0:
        print(f"Empty building crop for {image_path.name}.")
        return None

    return cropped_image


def get_city_name(
    latitude: float,
    longitude: float,
) -> tuple[str, str]:
    """Return the city and country associated with coordinates."""
    geolocator = Nominatim(
        user_agent="rubic_ai_city_name_locator",
        timeout=GEOCODER_TIMEOUT_SECONDS,
    )

    try:
        location = geolocator.reverse(
            (latitude, longitude),
            exactly_one=True,
            language="en",
        )
    except (GeocoderServiceError, GeocoderTimedOut) as error:
        print(f"Reverse geocoding failed: {error}")
        return "Unknown", "Unknown"

    if location is None:
        return "Unknown", "Unknown"

    address = location.raw.get("address", {})
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or address.get("state_district")
        or "Unknown"
    )
    country = address.get("country", "Unknown")
    return city, country


def extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    """Extract and normalize a model state dictionary."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            nested_state = checkpoint.get(key)
            if isinstance(nested_state, dict):
                checkpoint = nested_state
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint must contain a PyTorch state dictionary."
        )

    state_dict = {}
    for key, value in checkpoint.items():
        if not isinstance(value, Tensor):
            continue

        normalized_key = key.removeprefix("module.")
        state_dict[normalized_key] = value

    if not state_dict:
        raise ValueError("No tensor parameters were found in the checkpoint.")

    return state_dict


def read_checkpoint(weight_path: Path) -> dict[str, Tensor]:
    """Read a checkpoint and return its normalized state dictionary."""
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    checkpoint = torch.load(
        weight_path,
        map_location=DEVICE,
    )
    return extract_state_dict(checkpoint)


def find_output_layer(
    state_dict: dict[str, Tensor],
    architecture: str,
) -> tuple[str, int, bool]:
    """Find the output layer key, class count, and head structure."""
    if architecture == "convnext_tiny":
        candidate_keys = (
            "classifier.2.weight",
            "classifier.2.1.weight",
        )
    elif architecture == "swin_t":
        candidate_keys = (
            "head.weight",
            "head.1.weight",
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    for key in candidate_keys:
        weight = state_dict.get(key)
        if weight is not None and weight.ndim == 2:
            uses_sequential_head = ".1.weight" in key
            return key, int(weight.shape[0]), uses_sequential_head

    expected_keys = ", ".join(candidate_keys)
    raise KeyError(
        "Unable to locate the classifier output layer. "
        f"Expected one of: {expected_keys}"
    )


def create_classifier_model(
    specification: ModelSpecification,
) -> nn.Module:
    """Create a model whose output head matches its checkpoint."""
    state_dict = read_checkpoint(specification.weight_path)
    output_key, checkpoint_classes, uses_sequential_head = (
        find_output_layer(
            state_dict,
            specification.architecture,
        )
    )

    labels = CLASS_NAMES[specification.attribute]
    expected_classes = len(labels)

    if checkpoint_classes != expected_classes:
        raise ValueError(
            f"{specification.attribute}: checkpoint output layer "
            f"'{output_key}' has {checkpoint_classes} classes, but "
            f"{expected_classes} labels are configured: {labels}"
        )

    if specification.architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        input_features = model.classifier[2].in_features

        if uses_sequential_head:
            model.classifier[2] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(input_features, checkpoint_classes),
            )
        else:
            model.classifier[2] = nn.Linear(
                input_features,
                checkpoint_classes,
            )

    elif specification.architecture == "swin_t":
        model = models.swin_t(weights=None)
        input_features = model.head.in_features

        if uses_sequential_head:
            model.head = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(input_features, checkpoint_classes),
            )
        else:
            model.head = nn.Linear(
                input_features,
                checkpoint_classes,
            )

    else:
        raise ValueError(
            f"Unsupported architecture: {specification.architecture}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded {specification.attribute}: "
        f"{specification.architecture}, "
        f"{checkpoint_classes} classes"
    )
    return model


def load_prediction_models() -> PredictionModels:
    """Load and validate all building-attribute prediction models."""
    print(f"Loading deep-learning models on {DEVICE}...")

    loaded_models = {
        specification.attribute: create_classifier_model(specification)
        for specification in MODEL_SPECIFICATIONS
    }

    prediction_models = PredictionModels(
        material=loaded_models["material"],
        llrs=loaded_models["llrs"],
        code_level=loaded_models["code_level"],
        n_stories=loaded_models["n_stories"],
        occupancy=loaded_models["occupancy"],
        block_position=loaded_models["block_position"],
        roof_shape=loaded_models["roof_shape"],
        roof_material=loaded_models["roof_material"],
    )

    print("Deep-learning models loaded and validated successfully.")
    return prediction_models


def prepare_image(image_array: np.ndarray) -> Tensor:
    """Convert an OpenCV BGR image into a normalized model tensor."""
    rgb_image = cv2.cvtColor(
        image_array.astype(np.uint8),
        cv2.COLOR_BGR2RGB,
    )
    image = Image.fromarray(rgb_image)
    return IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)


def predict_class(
    image_array: np.ndarray,
    model: nn.Module,
    class_names: list[str],
) -> str:
    """Predict one class label for an image."""
    image_tensor = prepare_image(image_array)

    with torch.inference_mode():
        output = model(image_tensor)

    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    if output.shape[1] != len(class_names):
        raise ValueError(
            f"Model returned {output.shape[1]} outputs, but "
            f"{len(class_names)} labels were provided."
        )

    prediction = int(torch.argmax(output, dim=1).item())
    return class_names[prediction]


def normalize_llrs_label(label: str) -> str:
    """Map internal wall subclasses to the common LWAL taxonomy class."""
    if label in {"LWAL(LR)", "LWAL(HR)"}:
        return "LWAL"
    return label


def build_taxonomy(row: pd.Series) -> str:
    """Build a taxonomy string from the predicted attributes."""
    return (
        f"{row['material']}/{row['llrs']}/{row['code_level']}"
        f"/H:{row['n_stories']}/{row['block_position']}"
        f"/{row['roof_shape']}+{row['roof_material']}"
        f"/{row['occupancy']}"
    )


def validate_taxonomy(taxonomy: str) -> str | None:
    """Validate a taxonomy and return its canonical value when available."""
    taxonomy_data = pd.DataFrame({"TAXONOMY": [taxonomy]})

    try:
        check_taxonomy(
            taxonomy_data,
            taxo_col="TAXONOMY",
        )
        return taxonomy
    except ValueError as error:
        print(f"Invalid taxonomy: {taxonomy}")

        match = re.search(
            r"'canonical': '([^']+)'",
            str(error),
        )
        if match:
            canonical_taxonomy = match.group(1)
            print(f"Canonical taxonomy: {canonical_taxonomy}")
            return canonical_taxonomy

        print("No canonical taxonomy was found.")
        return None


def inspect_buildings(
    inspection_data: pd.DataFrame,
    footprint_data: pd.DataFrame,
    image_folder: Path,
    detector: YOLO,
    prediction_models: PredictionModels,
) -> pd.DataFrame:
    """Inspect local images and populate building-attribute predictions."""
    for row_index, footprint in footprint_data.iterrows():
        building_id = footprint["id"]
        latitude = float(footprint["latitude"])
        longitude = float(footprint["longitude"])
        image_path = image_folder / str(building_id)

        inspection_data.loc[
            row_index,
            ["id", "latitude", "longitude"],
        ] = [
            building_id,
            latitude,
            longitude,
        ]

        try:
            building_image = detect_building(
                image_path,
                detector,
            )
            if building_image is None:
                continue

            city, country = get_city_name(
                latitude,
                longitude,
            )
            inspection_data.loc[
                row_index,
                ["country", "city"],
            ] = [
                country,
                city,
            ]

            inspection_data.loc[row_index, "material"] = predict_class(
                building_image,
                prediction_models.material,
                CLASS_NAMES["material"],
            )

            llrs_label = predict_class(
                building_image,
                prediction_models.llrs,
                CLASS_NAMES["llrs"],
            )
            inspection_data.loc[row_index, "llrs"] = (
                normalize_llrs_label(llrs_label)
            )

            inspection_data.loc[row_index, "code_level"] = predict_class(
                building_image,
                prediction_models.code_level,
                CLASS_NAMES["code_level"],
            )
            inspection_data.loc[row_index, "n_stories"] = predict_class(
                building_image,
                prediction_models.n_stories,
                CLASS_NAMES["n_stories"],
            )
            inspection_data.loc[row_index, "occupancy"] = predict_class(
                building_image,
                prediction_models.occupancy,
                CLASS_NAMES["occupancy"],
            )
            inspection_data.loc[row_index, "block_position"] = (
                predict_class(
                    building_image,
                    prediction_models.block_position,
                    CLASS_NAMES["block_position"],
                )
            )
            inspection_data.loc[row_index, "roof_shape"] = predict_class(
                building_image,
                prediction_models.roof_shape,
                CLASS_NAMES["roof_shape"],
            )
            inspection_data.loc[row_index, "roof_material"] = (
                predict_class(
                    building_image,
                    prediction_models.roof_material,
                    CLASS_NAMES["roof_material"],
                )
            )

            taxonomy = build_taxonomy(inspection_data.loc[row_index])
            inspection_data.loc[row_index, "taxonomy"] = (
                validate_taxonomy(taxonomy) or taxonomy
            )
            inspection_data.loc[
                row_index,
                "image filename or link",
            ] = str(image_path)

        except (
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Error in building ID {building_id}: {error}")

        print(f"Inspection: {row_index + 1}/{len(inspection_data)}")

    return inspection_data


def main() -> None:
    """Run local-image building inspection."""
    local_building_info = SCRIPT_DIR / "data_ex1.csv"
    image_folder = SCRIPT_DIR / "images_ex1"
    saved_path = SCRIPT_DIR / "console_mode/local_results_ex1.csv"

    inspection_data, footprint_data = create_database(
        local_building_info
    )
    detector = load_building_detector()
    prediction_models = load_prediction_models()

    inspection_data = inspect_buildings(
        inspection_data,
        footprint_data,
        image_folder,
        detector,
        prediction_models,
    )

    saved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    inspection_data.to_csv(
        saved_path,
        index=False,
    )

    print(f"Results saved to: {saved_path}")


if __name__ == "__main__":
    main()
