from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
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
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from PIL import Image
from torch import Tensor
from torchvision import models
from ultralytics import YOLO

RUBICAI_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(RUBICAI_ROOT) not in sys.path:
    sys.path.append(str(RUBICAI_ROOT))

from methods.taxonomy import check_taxonomy  # noqa: E402

GSV_API_FILE = RUBICAI_ROOT / "methods/gsv_api_key.txt"
ROADS_API_FILE = RUBICAI_ROOT / "methods/roads_api_key.txt"
DL_DIR = RUBICAI_ROOT / "dl_weights"

REQUEST_TIMEOUT_SECONDS = 30
TARGET_CLASS = "building-xzyh"
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "Image filename or link",
]


@dataclass
class PredictionModels:
    """Store all loaded deep-learning models."""

    material: nn.Module
    llrs: nn.Module
    code: nn.Module
    n_stories: nn.Module
    occupancy: nn.Module
    block_position: nn.Module
    roof_shape: nn.Module
    roof_material: nn.Module


def read_api_key(path: Path) -> str:
    """Read an API key from a text file."""
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")

    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {path}")

    return api_key


def geodesic_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Calculate the geodesic distance between two points in kilometres."""
    return geodesic(
        (latitude_1, longitude_1),
        (latitude_2, longitude_2),
    ).km


def find_nearest_neighbors_geodesic(
    input_row: pd.Series,
    reference_data: pd.DataFrame,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """Return the nearest reference buildings using geodesic distance."""
    distances = reference_data.apply(
        lambda row: geodesic_distance(
            input_row["latitude"],
            input_row["longitude"],
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )

    nearest_indices = distances.nsmallest(n_neighbors).index
    neighbors = reference_data.loc[nearest_indices].copy()
    neighbors["distance_km"] = distances.loc[nearest_indices].values
    return neighbors


def compute_taxonomy_distribution(
    nearest_neighbors: pd.DataFrame,
    input_row: pd.Series,
) -> list[dict[str, Any]]:
    """Compute inverse-distance-weighted taxonomy probabilities."""
    class_weights: defaultdict[str, float] = defaultdict(float)

    for _, row in nearest_neighbors.iterrows():
        distance = float(row["distance_km"])
        taxonomy = str(row["taxonomy"])
        class_weights[taxonomy] += 1.0 / (distance + 1e-6)

    total_weight = sum(class_weights.values())
    if total_weight <= 0:
        return []

    probabilities = {
        taxonomy: weight / total_weight for taxonomy, weight in class_weights.items()
    }

    distribution_rows = []
    for taxonomy, probability in probabilities.items():
        representative = nearest_neighbors.loc[
            nearest_neighbors["taxonomy"] == taxonomy
        ].iloc[0]

        distribution_rows.append(
            {
                "id": input_row["id"],
                "latitude": input_row["latitude"],
                "longitude": input_row["longitude"],
                "country": representative["country"],
                "city": representative["city"],
                "material": representative["material"],
                "llrs": representative["llrs"],
                "code_level": representative["code_level"],
                "n_stories": representative["n_stories"],
                "occupancy": representative["occupancy"],
                "block_position": representative["block_position"],
                "taxonomy": taxonomy,
                "probability": probability,
            }
        )

    return distribution_rows


def extrapolate_existing_reference(
    reference_data: pd.DataFrame,
    target_data: pd.DataFrame,
    output_path: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extrapolate taxonomies and save their probability distributions."""
    distribution_rows: list[dict[str, Any]] = []

    for position, (_, input_row) in enumerate(
        target_data.iterrows(),
        start=1,
    ):
        nearest_neighbors = find_nearest_neighbors_geodesic(
            input_row,
            reference_data,
        )
        distribution_rows.extend(
            compute_taxonomy_distribution(nearest_neighbors, input_row)
        )

        if show_progress:
            print(f"Inspection: {position}/{len(target_data)}")

    result = pd.DataFrame(distribution_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def create_database(
    local_building_info: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load building coordinates and create an empty inspection database."""
    footprint_data = pd.read_csv(local_building_info)
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
    x_value = math.sin(longitude_difference) * math.cos(latitude_2)
    y_value = math.cos(latitude_1) * math.sin(latitude_2) - math.sin(
        latitude_1
    ) * math.cos(latitude_2) * math.cos(longitude_difference)

    azimuth = math.degrees(math.atan2(x_value, y_value))
    return (azimuth + 360) % 360


def get_road_orientation(
    location: tuple[float, float],
    roads_api_key: str,
) -> float | None:
    """Determine the road orientation near a geographic location."""
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

    data = response.json()
    snapped_points = data.get("snappedPoints", [])
    if not snapped_points:
        print("No road found near the location.")
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
    endpoint = "https://maps.googleapis.com/maps/api/streetview/metadata"
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
    """Download and decode a Google Street View image."""
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
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def get_street_view_image(
    location: tuple[float, float],
    gsv_api_key: str,
    roads_api_key: str,
    angle: float,
    pitch: float,
    field_of_view: float,
) -> tuple[str | None, np.ndarray | None, str | None]:
    """Fetch an outdoor Street View image and its Google Maps URL."""
    try:
        metadata = request_street_view_metadata(location, gsv_api_key)
    except requests.RequestException as error:
        print(f"Street View metadata request failed: {error}")
        return None, None, None

    pano_id = metadata.get("pano_id") or metadata.get("panoId")
    panorama_location = metadata.get("location", {})
    panorama_latitude = panorama_location.get("lat")
    panorama_longitude = panorama_location.get("lng")
    year = metadata.get("date", "").split("-")[0] or None

    if metadata.get("status") != "OK":
        pano_id = None
        for radius in range(5, 25, 5):
            try:
                metadata = request_street_view_metadata(
                    location,
                    gsv_api_key,
                    radius=radius,
                )
            except requests.RequestException as error:
                print(f"Metadata request failed at {radius} m: {error}")
                continue

            if metadata.get("status") != "OK":
                continue

            pano_id = metadata.get("pano_id") or metadata.get("panoId")
            panorama_location = metadata.get("location", {})
            panorama_latitude = panorama_location.get("lat")
            panorama_longitude = panorama_location.get("lng")
            year = metadata.get("date", "").split("-")[0] or None
            print(f"Outdoor panorama found within {radius} m.")
            break

        if pano_id is None:
            print(f"No outdoor panorama found near {location}.")
            return None, None, None

    if panorama_longitude is not None:
        angle = 180 if panorama_longitude > location[1] else 0

    road_orientation = get_road_orientation(location, roads_api_key)
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
        image_params["location"] = f"{location[0]},{location[1]}"
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


def check_street_view(
    latitude: float,
    longitude: float,
    api_key: str,
) -> bool:
    """Return whether outdoor Street View imagery is available."""
    try:
        metadata = request_street_view_metadata(
            (float(latitude), float(longitude)),
            api_key,
        )
    except requests.RequestException as error:
        print(f"Street View availability check failed: {error}")
        return False

    return metadata.get("status") == "OK"


def fetch_building_view(
    latitude: float,
    longitude: float,
    gsv_api_key: str,
    roads_api_key: str,
) -> tuple[np.ndarray | None, str | None]:
    """Fetch one Street View image for a building."""
    if not check_street_view(latitude, longitude, gsv_api_key):
        print("Street View is not available at the requested location.")
        return None, None

    maps_url, image, _ = get_street_view_image(
        (float(latitude), float(longitude)),
        gsv_api_key,
        roads_api_key,
        angle=0,
        pitch=5,
        field_of_view=120,
    )
    return image, maps_url


def load_building_detector() -> YOLO:
    """Load the YOLO building detector."""
    weight_path = DL_DIR / "building_detector.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {weight_path}")
    return YOLO(weight_path)


def detect_building(
    latitude: float,
    longitude: float,
    detector: YOLO,
    gsv_api_key: str,
    roads_api_key: str,
) -> tuple[np.ndarray | None, str | None]:
    """Fetch an image and return the highest-confidence building crop."""
    image, maps_url = fetch_building_view(
        latitude,
        longitude,
        gsv_api_key,
        roads_api_key,
    )
    if image is None:
        return None, maps_url

    try:
        prediction = detector.predict(
            image,
            device=str(DEVICE),
            verbose=False,
        )[0]
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"Building detection failed: {error}")
        return None, maps_url

    image_height, image_width = image.shape[:2]
    best_box: np.ndarray | None = None
    best_confidence = 0.0

    if prediction.boxes is not None:
        for box in prediction.boxes:
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
        print("No building was detected in the image.")
        return None, maps_url

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
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    try:
        location = geolocator.reverse(
            (latitude, longitude),
            exactly_one=True,
            language="en",
        )
    except Exception as error:
        print(f"Reverse geocoding failed: {error}")
        return "Unknown", "Unknown"

    if location is None or "address" not in location.raw:
        return "Unknown", "Unknown"

    address = location.raw["address"]
    city = address.get(
        "city",
        address.get(
            "town",
            address.get("village", "Unknown"),
        ),
    )
    country = address.get("country", "Unknown")
    return city, country


def prepare_model(model: nn.Module, weight_path: Path) -> nn.Module:
    """Load model weights and prepare the model for inference."""
    state_dict = torch.load(
        weight_path,
        map_location=DEVICE,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def create_convnext_model(
    number_of_classes: int,
    weight_path: Path,
) -> nn.Module:
    """Create and load a ConvNeXt Tiny classifier."""
    model = models.convnext_tiny(weights=None)
    input_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        input_features,
        number_of_classes,
    )
    return prepare_model(model, weight_path)


def create_swin_model(
    number_of_classes: int,
    weight_path: Path,
) -> nn.Module:
    """Create and load a Swin Tiny classifier."""
    model = models.swin_t(weights=None)
    input_features = model.head.in_features
    model.head = nn.Linear(
        input_features,
        number_of_classes,
    )
    return prepare_model(model, weight_path)


def load_prediction_models() -> PredictionModels:
    """Load all building-attribute prediction models."""
    print("Loading deep-learning models...")

    model_bundle = PredictionModels(
        material=create_convnext_model(
            3,
            DL_DIR / "convnext_tiny_material.pt",
        ),
        llrs=create_convnext_model(
            5,
            DL_DIR / "convnext_tiny_llrs.pt",
        ),
        code=create_convnext_model(
            4,
            DL_DIR / "convnext_tiny_code.pt",
        ),
        n_stories=create_convnext_model(
            9,
            DL_DIR / "convnext_tiny_n_stories.pt",
        ),
        occupancy=create_convnext_model(
            4,
            DL_DIR / "convnext_tiny_occupancy.pt",
        ),
        block_position=create_swin_model(
            4,
            DL_DIR / "swin_t_b_position.pt",
        ),
        roof_shape=create_swin_model(
            4,
            DL_DIR / "swin_t_roof_shape.pt",
        ),
        roof_material=create_swin_model(
            3,
            DL_DIR / "swin_t_roof_material.pt",
        ),
    )

    print("Deep-learning models loaded successfully.")
    return model_bundle


def prepare_image(image_array: np.ndarray) -> Tensor:
    """Convert an OpenCV image to a normalized model tensor."""
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
        prediction = int(torch.argmax(output, dim=1).item())

    return class_names[prediction]


def build_taxonomy(row: pd.Series) -> str:
    """Build a GEM-style taxonomy string from predicted attributes."""
    return (
        f"{row['material']}/{row['llrs']}/{row['code_level']}"
        f"/H:{row['n_stories']}/{row['block_position']}"
        f"/{row['roof_shape']}+{row['roof_material']}"
        f"/{row['occupancy']}"
    )


def check_taxonomy_value(taxonomy: str) -> str | None:
    """Validate a taxonomy and return a canonical replacement when available."""
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

    return None


def inspect_buildings(
    inspection_data: pd.DataFrame,
    footprint_data: pd.DataFrame,
    prediction_models: PredictionModels,
    detector: YOLO,
    gsv_api_key: str,
    roads_api_key: str,
) -> pd.DataFrame:
    """Inspect each building and populate its predicted attributes."""
    class_names = {
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

    for row_index, footprint in footprint_data.iterrows():
        building_id = footprint["id"]
        latitude = float(footprint["latitude"])
        longitude = float(footprint["longitude"])

        inspection_data.loc[row_index, ["id", "latitude", "longitude"]] = [
            building_id,
            latitude,
            longitude,
        ]

        try:
            image, maps_url = detect_building(
                latitude,
                longitude,
                detector,
                gsv_api_key,
                roads_api_key,
            )
            if image is None:
                print(f"No usable image for building ID {building_id}.")
                continue

            city, country = get_city_name(latitude, longitude)
            inspection_data.loc[row_index, ["country", "city"]] = [
                country,
                city,
            ]

            inspection_data.loc[row_index, "material"] = predict_class(
                image,
                prediction_models.material,
                class_names["material"],
            )
            inspection_data.loc[row_index, "llrs"] = predict_class(
                image,
                prediction_models.llrs,
                class_names["llrs"],
            )
            inspection_data.loc[row_index, "code_level"] = predict_class(
                image,
                prediction_models.code,
                class_names["code_level"],
            )
            inspection_data.loc[row_index, "n_stories"] = predict_class(
                image,
                prediction_models.n_stories,
                class_names["n_stories"],
            )
            inspection_data.loc[row_index, "occupancy"] = predict_class(
                image,
                prediction_models.occupancy,
                class_names["occupancy"],
            )
            inspection_data.loc[row_index, "block_position"] = predict_class(
                image,
                prediction_models.block_position,
                class_names["block_position"],
            )
            inspection_data.loc[row_index, "roof_shape"] = predict_class(
                image,
                prediction_models.roof_shape,
                class_names["roof_shape"],
            )
            inspection_data.loc[row_index, "roof_material"] = predict_class(
                image,
                prediction_models.roof_material,
                class_names["roof_material"],
            )

            taxonomy = build_taxonomy(inspection_data.loc[row_index])
            inspection_data.loc[row_index, "taxonomy"] = (
                check_taxonomy_value(taxonomy) or taxonomy
            )
            inspection_data.loc[
                row_index,
                "Image filename or link",
            ] = maps_url

        except (
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Error in building ID {building_id}: {error}")

        print(f"Inspection: {row_index + 1}/{len(inspection_data)}")

    return inspection_data


def run_existing_reference_mode() -> None:
    """Run extrapolation from an existing classified reference dataset."""
    reference_data = pd.read_csv(
        RUBICAI_ROOT / "demos/extrapolation/knn/neighbor_building_info.csv"
    )
    target_data = pd.read_csv(
        RUBICAI_ROOT / "demos/extrapolation/knn/unclassified_building_coord.csv"
    )
    output_path = (
        RUBICAI_ROOT / "demos/extrapolation/console_mode/extrapolation_data_example.csv"
    )

    extrapolate_existing_reference(
        reference_data,
        target_data,
        output_path,
        show_progress=True,
    )


def run_ai_reference_mode() -> None:
    """Create an AI-labelled reference dataset and then extrapolate."""
    coordinate_path = (
        RUBICAI_ROOT / "demos/extrapolation/knn/building_coordinates_example.csv"
    )
    reference_output_path = (
        RUBICAI_ROOT
        / "demos/extrapolation/console_mode"
        / "coordinates_reference_results.csv"
    )
    target_data = pd.read_csv(
        RUBICAI_ROOT / "demos/extrapolation/knn/unclassified_building_coord.csv"
    )
    extrapolation_output_path = (
        RUBICAI_ROOT
        / "demos/extrapolation/console_mode"
        / "extrapolation_data_example_using_ai.csv"
    )

    gsv_api_key = read_api_key(GSV_API_FILE)
    roads_api_key = read_api_key(ROADS_API_FILE)
    detector = load_building_detector()
    prediction_models = load_prediction_models()

    inspection_data, footprint_data = create_database(coordinate_path)
    inspection_data = inspect_buildings(
        inspection_data,
        footprint_data,
        prediction_models,
        detector,
        gsv_api_key,
        roads_api_key,
    )

    reference_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    inspection_data.to_csv(
        reference_output_path,
        index=False,
    )

    extrapolate_existing_reference(
        inspection_data,
        target_data,
        extrapolation_output_path,
        show_progress=False,
    )


def main() -> None:
    """Run the selected extrapolation workflow."""
    method = 0

    if method == 0:
        run_existing_reference_mode()
    elif method == 1:
        run_ai_reference_mode()
    else:
        raise ValueError("Method must be either 0 or 1.")


if __name__ == "__main__":
    main()
