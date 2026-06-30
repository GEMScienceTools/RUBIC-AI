"""Run polygon-based building sampling and attribute inference.

The workflow creates a polygon boundary, downloads building footprints,
extracts a random sample, creates centroid coordinates, retrieves Google
Street View imagery, detects buildings, predicts building attributes, and
exports the resulting taxonomy database.

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
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
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

GSV_API_FILE = RUBICAI_ROOT / "methods/gsv_api_key.txt"
ROADS_API_FILE = RUBICAI_ROOT / "methods/roads_api_key.txt"

TARGET_CLASS = "building-xzyh"
CONFIDENCE_THRESHOLD = 0.5
REQUEST_TIMEOUT_SECONDS = 30
GEOCODER_TIMEOUT_SECONDS = 3
MIN_BUILDING_AREA_M2 = 20
RANDOM_SEED = 10
DESIGN_CRS = "EPSG:4326"
AREA_CRS = "EPSG:3857"

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
    """Describe one classifier checkpoint."""

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


def read_api_key(path: Path) -> str:
    """Read and validate an API key file."""
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")

    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {path}")

    return api_key


def polygon_coordinates(
    file_path: Path,
    polygon_name: Path,
) -> Path:
    """Create and save a polygon boundary from CSV coordinates."""
    output_path = Path(f"{polygon_name}_boundary.gpkg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coordinate_data = pd.read_csv(file_path)
    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns.difference(coordinate_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing polygon-coordinate columns: {missing}")

    coordinates = list(
        zip(
            coordinate_data["longitude"],
            coordinate_data["latitude"],
            strict=True,
        )
    )
    polygon = Polygon(coordinates)

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    boundary = gpd.GeoDataFrame(
        {"geometry": [polygon]},
        crs=DESIGN_CRS,
    )
    boundary.to_file(
        output_path,
        driver="GPKG",
        layer="polygon_layer",
    )
    return output_path


def query_osm_buildings(
    polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Query OpenStreetMap buildings inside a polygon."""
    try:
        if hasattr(ox, "features_from_polygon"):
            return ox.features_from_polygon(
                polygon,
                tags={"building": True},
            )

        return ox.geometries_from_polygon(
            polygon,
            tags={"building": True},
        )
    except Exception as error:
        print(f"Skipping polygon because the OSM query failed: {error}")
        return gpd.GeoDataFrame()


def clean_geodataframe_columns(
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Rename columns that may be invalid in GeoPackage output."""
    reserved_names = {
        "Type",
        "FID",
        "Geometry",
        "geom",
        "geometry",
        "FIXME",
    }

    geometry_name = buildings.geometry.name
    renamed_columns = {}

    for column in buildings.columns:
        if column == geometry_name:
            continue

        if column in reserved_names or not str(column).isidentifier():
            renamed_columns[column] = f"{column}_field"

    buildings = buildings.rename(columns=renamed_columns)
    return buildings.set_geometry(geometry_name)


def add_identifier(
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add a string identifier column to building footprints."""
    fid_columns = [
        column
        for column in buildings.columns
        if str(column).lower().startswith("fid")
    ]

    if fid_columns:
        buildings["id"] = buildings[fid_columns[0]]
    elif "osmid" in buildings.columns:
        buildings["id"] = buildings["osmid"]
    elif "id" not in buildings.columns:
        buildings["id"] = range(1, len(buildings) + 1)

    buildings["id"] = buildings["id"].astype(str)
    return buildings


def download_building_footprints_polygon(
    polygon_name: Path,
) -> Path:
    """Download and save OSM building footprints for the boundary."""
    output_path = Path(f"{polygon_name}_buildings_footprint.gpkg")
    boundary_path = Path(f"{polygon_name}_boundary.gpkg")

    if output_path.exists():
        return output_path

    if not boundary_path.exists():
        raise FileNotFoundError(
            f"Polygon boundary file not found: {boundary_path}"
        )

    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        boundary = boundary.set_crs(DESIGN_CRS)
    elif boundary.crs.to_string() != DESIGN_CRS:
        boundary = boundary.to_crs(DESIGN_CRS)

    polygon = boundary.union_all()
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    ox.settings.overpass_endpoint = (
        "https://overpass-api.de/api/interpreter"
    )
    ox.settings.timeout = 180

    polygon_parts: list[Polygon] = []
    if isinstance(polygon, MultiPolygon):
        polygon_parts.extend(polygon.geoms)
    elif isinstance(polygon, Polygon):
        polygon_parts.append(polygon)
    else:
        raise TypeError(
            "Boundary geometry must be Polygon or MultiPolygon."
        )

    building_parts = []
    for index, polygon_part in enumerate(
        polygon_parts,
        start=1,
    ):
        print(
            f"Querying polygon part {index}/{len(polygon_parts)}..."
        )
        result = query_osm_buildings(polygon_part)
        if not result.empty:
            building_parts.append(result)

    if not building_parts:
        raise RuntimeError(
            "No building footprints were found in the selected polygon."
        )

    buildings = gpd.GeoDataFrame(
        pd.concat(building_parts, ignore_index=True)
    )
    buildings = buildings[
        buildings.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    buildings = clean_geodataframe_columns(buildings)
    buildings = buildings.to_crs(AREA_CRS)
    buildings["area_m2"] = buildings.geometry.area

    before_filter = len(buildings)
    buildings = buildings[
        buildings["area_m2"] > MIN_BUILDING_AREA_M2
    ].copy()
    after_filter = len(buildings)

    print(
        "Filtered buildings by area: "
        f"{before_filter} -> {after_filter} "
        f"(>{MIN_BUILDING_AREA_M2} m²)"
    )

    buildings = buildings.to_crs(DESIGN_CRS)
    buildings = add_identifier(buildings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(
        output_path,
        driver="GPKG",
    )
    return output_path


def extract_random_subset(
    polygon_name: Path,
    sample_size: float | int,
) -> Path:
    """Extract and save a reproducible random footprint sample."""
    footprint_path = Path(
        f"{polygon_name}_buildings_footprint.gpkg"
    )
    output_path = Path(f"{polygon_name}_subset_footprints.gpkg")

    if output_path.exists():
        return output_path

    buildings = gpd.read_file(footprint_path)
    population_size = len(buildings)

    if isinstance(sample_size, float) and 0 < sample_size < 1:
        number_to_sample = max(
            1,
            round(population_size * sample_size),
        )
    elif isinstance(sample_size, int) and sample_size > 0:
        number_to_sample = sample_size
    else:
        raise ValueError(
            "sample_size must be a positive integer or a fraction "
            "between 0 and 1."
        )

    number_to_sample = min(number_to_sample, population_size)
    subset = buildings.sample(
        n=number_to_sample,
        random_state=RANDOM_SEED,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_file(
        output_path,
        driver="GPKG",
        layer="random_subset",
    )
    return output_path


def create_centroid_layer(
    polygon_name: Path,
) -> Path:
    """Create centroid coordinates for the sampled footprints."""
    subset_path = Path(f"{polygon_name}_subset_footprints.gpkg")
    centroid_path = Path(f"{polygon_name}_subset_centroids.gpkg")
    database_path = Path(f"{polygon_name}_building_info.csv")

    if centroid_path.exists() and database_path.exists():
        return database_path

    buildings = gpd.read_file(subset_path)
    if buildings.crs is None:
        buildings = buildings.set_crs(DESIGN_CRS)

    projected_crs = buildings.estimate_utm_crs()
    if projected_crs is None:
        projected_crs = AREA_CRS

    projected = buildings.to_crs(projected_crs)
    centroid_geometry = projected.centroid.to_crs(DESIGN_CRS)

    centroids = gpd.GeoDataFrame(
        buildings.drop(columns=buildings.geometry.name),
        geometry=centroid_geometry,
        crs=DESIGN_CRS,
    )
    centroids["latitude"] = centroids.geometry.y
    centroids["longitude"] = centroids.geometry.x

    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_file(
        centroid_path,
        driver="GPKG",
        layer="centroids",
    )

    centroids[
        ["id", "latitude", "longitude"]
    ].to_csv(
        database_path,
        index=False,
    )
    return database_path


def create_database(
    polygon_name: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an empty inspection table and load centroid coordinates."""
    database_path = Path(f"{polygon_name}_building_info.csv")
    footprint_data = pd.read_csv(database_path)
    footprint_data.columns = (
        footprint_data.columns.str.lower().str.strip()
    )

    required_columns = {"id", "latitude", "longitude"}
    missing_columns = required_columns.difference(
        footprint_data.columns
    )
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
) -> Tensor:
    """Convert an OpenCV BGR image to a normalized tensor."""
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
    """Predict one class label and validate the output dimension."""
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

    return inspection_data


def prepare_polygon_sample(
    polygon_name: Path,
    sample_size: float | int,
    coordinate_file: Path,
) -> None:
    """Create the polygon, footprints, subset, and centroid table."""
    polygon_coordinates(
        coordinate_file,
        polygon_name,
    )
    download_building_footprints_polygon(polygon_name)
    extract_random_subset(
        polygon_name,
        sample_size,
    )
    create_centroid_layer(polygon_name)


def main() -> None:
    """Run the complete polygon-based inference workflow."""
    coordinate_file = (
        RUBICAI_ROOT
        / "demos/polygon_method/polygon_method_example.csv"
    )
    polygon_name = (
        RUBICAI_ROOT
        / "demos/polygon_method/console_mode/proof_polygon"
    )
    saved_path = (
        RUBICAI_ROOT
        / "demos/polygon_method/console_mode"
        / "example_prediction_result.csv"
    )
    sample_size = 0.15

    gsv_api_key = read_api_key(GSV_API_FILE)
    roads_api_key = read_api_key(ROADS_API_FILE)

    prepare_polygon_sample(
        polygon_name,
        sample_size,
        coordinate_file,
    )

    inspection_data, footprint_data = create_database(
        polygon_name
    )
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
