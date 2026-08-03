"""Run building-attribute inference for specific coordinates.

The workflow reads a CSV file containing building identifiers and geographic
coordinates, retrieves Google Street View imagery, detects the most likely
building, predicts its attributes, validates the resulting taxonomy, and
exports the completed inspection database.

All classification models are validated against their configured class
labels before inference.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import requests
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
DL_DIR = RUBICAI_ROOT / "dl_weights"

if str(RUBICAI_ROOT) not in sys.path:
    sys.path.append(str(RUBICAI_ROOT))

from methods.taxonomy import check_taxonomy  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GSV_API_FILE = RUBICAI_ROOT / "methods/gsv_api_key.txt"
ROADS_API_FILE = RUBICAI_ROOT / "methods/roads_api_key.txt"

TARGET_CLASS = "building-xzyh"
CONFIDENCE_THRESHOLD = 0.5
REQUEST_TIMEOUT_SECONDS = 30
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
    "material": ["CR", "MCF", "MUR"],
    "llrs": ["LDUAL", "LFINF", "LFM", "LWAL", "LWAL"],
    "code_level": ["CDH", "CDL", "CDM", "CDN"],
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
    "occupancy": ["COM", "IND", "MIX(RES;COM)", "RES"],
    "block_position": ["BP1", "BP2", "BP3", "BPD"],
    "roof_shape": ["RSH1", "RSH2", "RSH3", "RSH7"],
    "roof_material": ["RMN", "RMT1", "RMT6"],
}

CONVNEXT_TRANSFORM_NS = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

CONVNEXT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

SWIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@dataclass
class LoadedPredictionModel:
    """Store a classifier together with its preprocessing transform."""

    model: nn.Module
    image_transform: transforms.Compose


@dataclass
class PredictionModels:
    """Store all loaded building-attribute model bundles."""

    material: LoadedPredictionModel
    llrs: LoadedPredictionModel
    code_level: LoadedPredictionModel
    n_stories: LoadedPredictionModel
    occupancy: LoadedPredictionModel
    block_position: LoadedPredictionModel
    roof_shape: LoadedPredictionModel
    roof_material: LoadedPredictionModel


@dataclass(frozen=True)
class ModelSpecification:
    """Describe one classifier checkpoint and inference configuration."""

    attribute: str
    architecture: str
    weight_path: Path
    image_transform: transforms.Compose


MODEL_SPECIFICATIONS = [
    ModelSpecification(
        attribute="material",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_material.pt",
        image_transform=CONVNEXT_TRANSFORM,
    ),
    ModelSpecification(
        attribute="llrs",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_llrs.pt",
        image_transform=CONVNEXT_TRANSFORM,
    ),
    ModelSpecification(
        attribute="code_level",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_code.pt",
        image_transform=CONVNEXT_TRANSFORM,
    ),
    ModelSpecification(
        attribute="n_stories",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_n_stories.pt",
        image_transform=CONVNEXT_TRANSFORM_NS,
    ),
    ModelSpecification(
        attribute="occupancy",
        architecture="convnext_tiny",
        weight_path=DL_DIR / "convnext_tiny_occupancy.pt",
        image_transform=CONVNEXT_TRANSFORM,
    ),
    ModelSpecification(
        attribute="block_position",
        architecture="swin_t",
        weight_path=DL_DIR / "swin_t_b_position.pt",
        image_transform=SWIN_TRANSFORM,
    ),
    ModelSpecification(
        attribute="roof_shape",
        architecture="swin_t",
        weight_path=DL_DIR / "swin_t_roof_shape.pt",
        image_transform=SWIN_TRANSFORM,
    ),
    ModelSpecification(
        attribute="roof_material",
        architecture="swin_t",
        weight_path=DL_DIR / "swin_t_roof_material.pt",
        image_transform=SWIN_TRANSFORM,
    ),
]


def read_api_key(path: Path) -> str:
    """Read and validate an API key file."""
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")

    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {path}")

    return api_key


def create_database(
    coordinate_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an empty inspection table and load building coordinates."""
    footprint_data = pd.read_csv(coordinate_file)
    footprint_data.columns = footprint_data.columns.str.lower().str.strip()

    required_columns = {"id", "latitude", "longitude"}
    missing_columns = required_columns.difference(footprint_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required building columns: {missing}"
        )

    inspection_data = pd.DataFrame(
        np.full(
            (len(footprint_data), len(DATABASE_COLUMNS)),
            None,
            dtype=object,
        ),
        columns=DATABASE_COLUMNS,
    )
    return inspection_data, footprint_data


def compute_azimuth(
    point_1: tuple[float, float],
    point_2: tuple[float, float],
) -> float:
    """Compute the azimuth between two geographic points."""
    latitude_1, longitude_1 = map(math.radians, point_1)
    latitude_2, longitude_2 = map(math.radians, point_2)

    longitude_difference = longitude_2 - longitude_1
    x_value = (
        math.sin(longitude_difference) * math.cos(latitude_2)
    )
    y_value = (
        math.cos(latitude_1) * math.sin(latitude_2)
        - math.sin(latitude_1)
        * math.cos(latitude_2)
        * math.cos(longitude_difference)
    )

    azimuth = math.degrees(math.atan2(x_value, y_value))
    return (azimuth + 360) % 360


def get_road_orientation(
    location: tuple[float, float],
    roads_api_key: str,
) -> float | None:
    """Get the road orientation near a geographic location."""
    endpoint = "https://roads.googleapis.com/v1/nearestRoads"
    params = {
        "points": f"{location[0]},{location[1]}",
        "key": roads_api_key,
    }

    try:
        response = requests.get(
            endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Roads API request failed: {error}")
        return None

    snapped_points = response.json().get("snappedPoints", [])
    if not snapped_points:
        print("No road was found near the building.")
        return None

    snapped_location = snapped_points[0]["location"]
    road_point = (
        snapped_location["latitude"],
        snapped_location["longitude"],
    )
    return compute_azimuth(location, road_point)


def request_street_view_metadata(
    location: tuple[float, float],
    api_key: str,
    radius: int | None = None,
) -> dict[str, Any]:
    """Request Google Street View metadata."""
    endpoint = (
        "https://maps.googleapis.com/maps/api/streetview/metadata"
    )
    params: dict[str, Any] = {
        "location": f"{location[0]},{location[1]}",
        "source": "outdoor",
        "key": api_key,
    }

    if radius is not None:
        params["radius"] = radius

    response = requests.get(
        endpoint,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def download_street_view_image(
    params: dict[str, Any],
) -> np.ndarray | None:
    """Download and decode one Street View image."""
    endpoint = "https://maps.googleapis.com/maps/api/streetview"

    response = requests.get(
        endpoint,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        print("Street View response did not contain an image.")
        return None

    image_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )


def get_street_view_image(
    location: tuple[float, float],
    gsv_api_key: str,
    roads_api_key: str,
    angle: float = 0,
    pitch: float = 5,
    field_of_view: float = 120,
) -> tuple[str | None, np.ndarray | None, str | None]:
    """Fetch an outdoor Street View image and its Maps URL."""
    metadata = None

    for radius in (None, 5, 10, 15, 20):
        try:
            metadata = request_street_view_metadata(
                location,
                gsv_api_key,
                radius=radius,
            )
        except requests.RequestException as error:
            print(f"Street View metadata request failed: {error}")
            continue

        if metadata.get("status") == "OK":
            break

    if metadata is None or metadata.get("status") != "OK":
        print(f"No outdoor Street View panorama near {location}.")
        return None, None, None

    pano_id = metadata.get("pano_id") or metadata.get("panoId")
    panorama_location = metadata.get("location", {})
    panorama_latitude = panorama_location.get("lat")
    panorama_longitude = panorama_location.get("lng")
    year = metadata.get("date", "").split("-")[0] or None

    if panorama_longitude is not None:
        angle = 180 if panorama_longitude > location[1] else 0

    road_orientation = get_road_orientation(
        location,
        roads_api_key,
    )
    heading = ((road_orientation or 0) + angle + 180) % 360

    image_params: dict[str, Any] = {
        "size": "640x480",
        "heading": heading,
        "pitch": pitch,
        "fov": field_of_view,
        "source": "outdoor",
        "key": gsv_api_key,
    }

    if pano_id:
        image_params["pano"] = pano_id
    else:
        image_params["location"] = (
            f"{location[0]},{location[1]}"
        )
        image_params["scale"] = 2

    try:
        image = download_street_view_image(image_params)
    except requests.RequestException as error:
        print(f"Street View image request failed: {error}")
        return None, None, year

    viewpoint_latitude = panorama_latitude or location[0]
    viewpoint_longitude = panorama_longitude or location[1]
    maps_url = (
        "https://www.google.com/maps/@?api=1&map_action=pano"
        f"&viewpoint={viewpoint_latitude},{viewpoint_longitude}"
        f"&heading={heading}&pitch={pitch}&fov={field_of_view}"
    )
    return maps_url, image, year


def load_building_detector() -> YOLO:
    """Load the YOLO building detector once."""
    weight_path = DL_DIR / "building_detector.pt"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Building-detector weights not found: {weight_path}"
        )

    return YOLO(weight_path)


def detect_building(
    latitude: float,
    longitude: float,
    detector: YOLO,
    gsv_api_key: str,
    roads_api_key: str,
) -> tuple[np.ndarray | None, str | None]:
    """Retrieve Street View imagery and return the best building crop."""
    maps_url, image, _ = get_street_view_image(
        (latitude, longitude),
        gsv_api_key,
        roads_api_key,
    )

    if image is None:
        return None, maps_url

    try:
        result = detector.predict(
            source=image,
            device=str(DEVICE),
            verbose=False,
        )[0]
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"Building detection failed: {error}")
        return None, maps_url

    best_box: np.ndarray | None = None
    best_confidence = 0.0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = detector.names[class_id]
            confidence = float(box.conf[0])

            if (
                label == TARGET_CLASS
                and confidence > CONFIDENCE_THRESHOLD
                and confidence > best_confidence
            ):
                best_confidence = confidence
                best_box = box.xyxy[0].cpu().numpy().astype(int)

    if best_box is None:
        print("No building was detected in the Street View image.")
        return None, maps_url

    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = best_box

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)

    cropped_image = image[y_min:y_max, x_min:x_max]
    if cropped_image.size == 0:
        print("The detected building crop is empty.")
        return None, maps_url

    return cropped_image, maps_url


def get_city_name(
    latitude: float,
    longitude: float,
) -> tuple[str, str]:
    """Return the city and country corresponding to coordinates."""
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


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, Tensor]:
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
        if isinstance(value, Tensor):
            state_dict[key.removeprefix("module.")] = value

    if not state_dict:
        raise ValueError(
            "No tensor parameters were found in the checkpoint."
        )

    return state_dict


def read_checkpoint(
    weight_path: Path,
) -> dict[str, Tensor]:
    """Read and normalize a model checkpoint."""
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weight_path}"
        )

    checkpoint = torch.load(
        weight_path,
        map_location=DEVICE,
    )
    return extract_state_dict(checkpoint)


def create_classifier_model(
    specification: ModelSpecification,
) -> LoadedPredictionModel:
    """Create a classifier using the confirmed RUBIC-AI configuration."""
    state_dict = read_checkpoint(specification.weight_path)
    labels = CLASS_NAMES[specification.attribute]
    num_classes = len(labels)

    if specification.architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        input_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(input_features, num_classes),
        )
    elif specification.architecture == "swin_t":
        model = models.swin_t(weights=None)
        input_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_features, num_classes),
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
        f"{specification.architecture}, {num_classes} classes"
    )
    return LoadedPredictionModel(
        model=model,
        image_transform=specification.image_transform,
    )


def load_prediction_models() -> PredictionModels:
    """Load and validate every attribute-prediction model."""
    print(f"Loading deep-learning models on {DEVICE}...")

    loaded_models = {
        specification.attribute: create_classifier_model(
            specification
        )
        for specification in MODEL_SPECIFICATIONS
    }

    model_bundle = PredictionModels(
        material=loaded_models["material"],
        llrs=loaded_models["llrs"],
        code_level=loaded_models["code_level"],
        n_stories=loaded_models["n_stories"],
        occupancy=loaded_models["occupancy"],
        block_position=loaded_models["block_position"],
        roof_shape=loaded_models["roof_shape"],
        roof_material=loaded_models["roof_material"],
    )

    print("All deep-learning models loaded and validated.")
    return model_bundle


def prepare_image(
    image_array: np.ndarray,
    image_transform: transforms.Compose,
) -> Tensor:
    """Convert an OpenCV BGR image using the model-specific transform."""
    rgb_image = cv2.cvtColor(
        image_array.astype(np.uint8),
        cv2.COLOR_BGR2RGB,
    )
    image = Image.fromarray(rgb_image)
    return image_transform(image).unsqueeze(0).to(DEVICE)


def predict_class(
    image_array: np.ndarray,
    model_bundle: LoadedPredictionModel,
    class_names: list[str],
) -> str:
    """Predict one class label using its training-time configuration."""
    image_tensor = prepare_image(
        image_array,
        model_bundle.image_transform,
    )

    with torch.no_grad():
        output = model_bundle.model(image_tensor)

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


def normalize_llrs_label(
    label: str,
) -> str:
    """Map internal wall subclasses to the common LWAL class."""
    if label in {"LWAL(LR)", "LWAL(HR)"}:
        return "LWAL"
    return label


def build_taxonomy(
    row: pd.Series,
) -> str:
    """Build a taxonomy string from predicted attributes."""
    return (
        f"{row['material']}/{row['llrs']}/{row['code_level']}"
        f"/H:{row['n_stories']}/{row['block_position']}"
        f"/{row['roof_shape']}+{row['roof_material']}"
        f"/{row['occupancy']}"
    )


def validate_taxonomy(
    taxonomy: str,
) -> str | None:
    """Validate a taxonomy and return its canonical value."""
    taxonomy_data = pd.DataFrame(
        {"TAXONOMY": [taxonomy]}
    )

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
    detector: YOLO,
    prediction_models: PredictionModels,
    gsv_api_key: str,
    roads_api_key: str,
) -> pd.DataFrame:
    """Inspect sampled buildings and populate their attributes."""
    for row_index, footprint in footprint_data.iterrows():
        try:
            building_id = footprint["id"]
            latitude = float(footprint["latitude"])
            longitude = float(footprint["longitude"])

            inspection_data.loc[
                row_index,
                ["id", "latitude", "longitude"],
            ] = [
                building_id,
                latitude,
                longitude,
            ]

            try:
                building_image, maps_url = detect_building(
                    latitude,
                    longitude,
                    detector,
                    gsv_api_key,
                    roads_api_key,
                )

                if building_image is None:
                    print(
                        f"No usable building image for ID {building_id}."
                    )
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

                inspection_data.loc[row_index, "material"] = (
                    predict_class(
                        building_image,
                        prediction_models.material,
                        CLASS_NAMES["material"],
                    )
                )

                llrs_label = predict_class(
                    building_image,
                    prediction_models.llrs,
                    CLASS_NAMES["llrs"],
                )
                inspection_data.loc[row_index, "llrs"] = (
                    normalize_llrs_label(llrs_label)
                )

                inspection_data.loc[row_index, "code_level"] = (
                    predict_class(
                        building_image,
                        prediction_models.code_level,
                        CLASS_NAMES["code_level"],
                    )
                )
                inspection_data.loc[row_index, "n_stories"] = (
                    predict_class(
                        building_image,
                        prediction_models.n_stories,
                        CLASS_NAMES["n_stories"],
                    )
                )
                inspection_data.loc[row_index, "occupancy"] = (
                    predict_class(
                        building_image,
                        prediction_models.occupancy,
                        CLASS_NAMES["occupancy"],
                    )
                )
                inspection_data.loc[row_index, "block_position"] = (
                    predict_class(
                        building_image,
                        prediction_models.block_position,
                        CLASS_NAMES["block_position"],
                    )
                )
                inspection_data.loc[row_index, "roof_shape"] = (
                    predict_class(
                        building_image,
                        prediction_models.roof_shape,
                        CLASS_NAMES["roof_shape"],
                    )
                )
                inspection_data.loc[row_index, "roof_material"] = (
                    predict_class(
                        building_image,
                        prediction_models.roof_material,
                        CLASS_NAMES["roof_material"],
                    )
                )

                taxonomy = build_taxonomy(
                    inspection_data.loc[row_index]
                )
                inspection_data.loc[row_index, "taxonomy"] = (
                    validate_taxonomy(taxonomy) or taxonomy
                )
                inspection_data.loc[
                    row_index,
                    "image filename or link",
                ] = maps_url

            except (
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                print(
                    f"Error while processing building ID "
                    f"{building_id}: {error}"
                )

            print(
                f"Inspection: {row_index + 1}/"
                f"{len(inspection_data)}"
            )
        except ValueError:
            building_id = footprint["id"]
            latitude = footprint["latitude"]
            longitude = footprint["longitude"]
            inspection_data.loc[
                row_index,
                ["id", "latitude", "longitude"],
            ] = [
                building_id,
                latitude,
                longitude,
            ]

            print(
                f"Inspection: {row_index + 1}/"
                f"{len(inspection_data)}"
                + " has failed"
            )

    return inspection_data


def main() -> None:
    """Run inference for the buildings listed in the input CSV file."""
    coordinate_file = (
        RUBICAI_ROOT
        / "demos/specific_coordinates/specific_coordinates_example_data.csv"
    )
    saved_path = (
        RUBICAI_ROOT
        / "demos/specific_coordinates/console_mode"
        / "console_example_prediction_result.csv"
    )

    gsv_api_key = read_api_key(GSV_API_FILE)
    roads_api_key = read_api_key(ROADS_API_FILE)

    inspection_data, footprint_data = create_database(coordinate_file)
    detector = load_building_detector()
    prediction_models = load_prediction_models()

    inspection_data = inspect_buildings(
        inspection_data,
        footprint_data,
        detector,
        prediction_models,
        gsv_api_key,
        roads_api_key,
    )

    saved_path.parent.mkdir(parents=True, exist_ok=True)
    inspection_data.to_csv(saved_path, index=False)

    print(f"Results saved to: {saved_path}")


if __name__ == "__main__":
    main()