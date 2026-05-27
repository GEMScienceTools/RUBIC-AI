from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

import mapillary.interface as mly
import torch
from ultralytics import YOLO


# ============================================================
# USER SETTINGS
# ============================================================

# Search settings
MAX_OFFSET_METERS = 20.0
IMAGE_TYPE = "pano"

# Download / output
OUTPUT_DIR = Path("mapillary_building_results")
IMAGES_DIR = OUTPUT_DIR / "images"
ORTHO_DIR = OUTPUT_DIR / "orthophotos"
CROPS_DIR = OUTPUT_DIR / "building_crops"
RAW_DIR = OUTPUT_DIR / "raw_json"
RESULTS_CSV = OUTPUT_DIR / "results.csv"


# Sleep between requests/downloads
SLEEP_SECONDS = 2.0

# Thumbnail resolution requested from SDK helper
THUMBNAIL_RESOLUTION = 2048

# Orthophoto controls
ORTHO_ENABLE = True
ORTHO_FOV_DEG = 120
ORTHO_PITCH_DEG = 0.0
ORTHO_OUT_H = 1024
ORTHO_OUT_W = 1024
ORTHO_VERTICAL_FLIP = False
ORTHO_PITCH_OFFSETS = [0, -30]
ORTHO_BASE_YAW_DEG = 180
RETRY_WITH_ROTATED_YAW_IF_NO_BUILDING = True
RETRY_YAW_DELTA_DEG = -90.0
STOP_AFTER_FIRST_SUCCESS = True

# YOLO building detector settings
BUILDING_DETECTOR_WEIGHTS = r"C:\Users\user\Documents\GitHub\RUBIC-AI\dl_weights\building_detector.pt"
TARGET_CLASS = "building-xzyh"
CONF_THRESHOLD = 0.5

REQUEST_TIMEOUT = 30
PRINT_PROGRESS = False

# Runtime values set by mapillary_image_source(...)
MAPILLARY_ACCESS_TOKEN = ""
YOLO_MODEL: Optional[YOLO] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOGGING
# ============================================================

def log_info(message: str) -> None:
    """Print progress messages only when PRINT_PROGRESS is enabled."""
    if PRINT_PROGRESS:
        print(message)


def log_error(message: str) -> None:
    """Print error messages."""
    print(message)


# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for folder in [OUTPUT_DIR, IMAGES_DIR, ORTHO_DIR, CROPS_DIR, RAW_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Save a Python object as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_response(response: Any) -> Dict[str, Any]:
    """Convert Mapillary SDK responses into a plain Python dictionary."""
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return json.loads(response)
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__geo_interface__"):
        return response.__geo_interface__
    if hasattr(response, "features"):
        return {
            "type": getattr(response, "type", "FeatureCollection"),
            "features": response.features,
        }

    raise TypeError(f"Unsupported response type: {type(response)}")


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two coordinates in meters."""
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def meters_to_degree_offsets(lat_deg: float, meters: float) -> Tuple[float, float]:
    """Approximate meter-to-degree conversion around a given latitude."""
    delta_lat = meters / 111320.0
    cos_lat = math.cos(math.radians(lat_deg))
    delta_lon = meters / 111320.0 if abs(cos_lat) < 1e-12 else meters / (111320.0 * cos_lat)
    return delta_lat, delta_lon


def build_bbox(lat: float, lon: float, radius_m: float) -> Dict[str, float]:
    """Build a small bounding box around a point."""
    delta_lat, delta_lon = meters_to_degree_offsets(lat, radius_m * 2.0)
    return {
        "west": lon - delta_lon,
        "south": lat - delta_lat,
        "east": lon + delta_lon,
        "north": lat + delta_lat,
    }


# ============================================================
# CSV INPUT
# ============================================================

def load_coordinates_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Read building coordinates from a CSV file."""
    df = pd.read_csv(csv_path)

    missing = {"latitude", "longitude"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if "id" not in df.columns:
        df["id"] = [str(i + 1) for i in range(len(df))]

    return [
        {
            "id": str(row.id),
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }
        for row in df.itertuples(index=False)
    ]


# ============================================================
# MAPILLARY QUERY
# ============================================================

def query_images_in_bbox(
    building_lat: float,
    building_lon: float,
    search_radius_m: float,
    image_type: str,
) -> Dict[str, Any]:
    """Query Mapillary images in a small bounding box around the building point."""
    bbox = build_bbox(building_lat, building_lon, search_radius_m)
    bbox_image_type = "all" if image_type == "both" else image_type
    response = mly.images_in_bbox(bbox=bbox, image_type=bbox_image_type)
    return parse_response(response)


def find_nearest_image_from_building(
    building_lat: float,
    building_lon: float,
    max_offset_m: float,
    image_type: str,
) -> Optional[Dict[str, Any]]:
    """Find the nearest Mapillary image within max_offset_m from the building point."""
    data = query_images_in_bbox(building_lat, building_lon, max_offset_m, image_type)
    features = data.get("features", [])

    best_result = None
    best_distance = float("inf")

    for feature in features:
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue

        image_lon, image_lat = coords[0], coords[1]
        props = feature.get("properties", {})
        is_pano = bool(props.get("is_pano", False))

        if (image_type == "pano" and not is_pano) or (image_type == "flat" and is_pano):
            continue

        dist_m = haversine_meters(building_lat, building_lon, image_lat, image_lon)

        if dist_m <= max_offset_m and dist_m < best_distance:
            best_distance = dist_m
            best_result = {
                "image_id": str(props.get("id")),
                "sequence_id": props.get("sequence_id"),
                "captured_at": props.get("captured_at"),
                "compass_angle": props.get("compass_angle"),
                "is_pano": is_pano,
                "preview_latitude": image_lat,
                "preview_longitude": image_lon,
                "building_to_preview_distance_m": dist_m,
                "raw_feature": feature,
            }

    return best_result


def get_image_metadata(image_id: str) -> Dict[str, Any]:
    """Request image metadata with explicit fields."""
    response = mly.image_from_key(
        key=image_id,
        fields=[
            "captured_at",
            "compass_angle",
            "computed_compass_angle",
            "camera_type",
            "computed_rotation",
            "geometry",
            "computed_geometry",
            "width",
            "height",
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
        ],
    )
    return parse_response(response)


def extract_camera_position(metadata: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract original and computed image positions from Mapillary metadata."""
    props = metadata.get("properties", metadata)

    def point_to_latlon(point: Any) -> Tuple[Optional[float], Optional[float]]:
        if isinstance(point, dict) and point.get("type") == "Point":
            coords = point.get("coordinates", [])
            if len(coords) >= 2:
                return coords[1], coords[0]
        return None, None

    original_lat, original_lon = point_to_latlon(props.get("geometry"))
    computed_lat, computed_lon = point_to_latlon(props.get("computed_geometry"))

    return {
        "original_latitude": original_lat,
        "original_longitude": original_lon,
        "computed_latitude": computed_lat,
        "computed_longitude": computed_lon,
    }


# ============================================================
# IMAGE DOWNLOAD / DISPLAY
# ============================================================

def get_thumbnail_url(image_id: str, metadata: Dict[str, Any]) -> Optional[str]:
    """Prefer the SDK thumbnail helper, then fall back to metadata thumbnail fields."""
    try:
        thumb_url = mly.image_thumbnail(image_id=image_id, resolution=THUMBNAIL_RESOLUTION)
        if isinstance(thumb_url, str) and thumb_url.strip():
            return thumb_url
    except Exception as e:
        log_error(f"❌ image_thumbnail() failed for {image_id}: {e}")

    props = metadata.get("properties", metadata)
    for field in ["thumb_original_url", "thumb_2048_url", "thumb_1024_url", "thumb_256_url"]:
        value = props.get(field)
        if value:
            return value

    return None


def download_image_to_pil(url: str) -> Image.Image:
    """Download an image URL and return it as a PIL RGB image."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def plot_numpy_image(img_array: np.ndarray, title: str = "Image") -> None:
    """Plot a NumPy RGB image using matplotlib."""
    plt.figure(figsize=(8, 8))
    plt.imshow(img_array)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# ORTHOPHOTO
# ============================================================

def generate_orthophoto_from_360(
    image_path: Path,
    output_path: Path,
    yaw_deg: float,
    fov_deg: float,
    pitch_deg: float,
    out_h: int,
    out_w: int,
    vertical_flip: bool = False,
) -> np.ndarray:
    """Extract a perspective/orthophoto-like view from a 360 equirectangular image."""
    pano = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = pano.shape

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    fov = math.radians(fov_deg)

    xs = np.linspace(-1, 1, out_w)
    ys = np.linspace(-1, 1, out_h)
    xv, yv = np.meshgrid(xs, ys)

    half_width = math.tan(fov / 2.0)
    half_height = half_width * (out_h / out_w)

    x_cam = np.ones_like(xv)
    y_cam = xv * half_width
    z_cam = -yv * half_height

    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm

    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    x1 = cp * x_cam + sp * z_cam
    y1 = y_cam
    z1 = -sp * x_cam + cp * z_cam

    x2 = cy * x1 - sy * y1
    y2 = sy * x1 + cy * y1
    z2 = z1

    lon = np.arctan2(y2, x2)
    lat = np.arcsin(np.clip(z2, -1.0, 1.0))

    u = (lon / (2 * math.pi) + 0.5) * w
    v = (0.5 - lat / math.pi) * h

    u = np.mod(u, w)
    v = np.clip(v, 0, h - 1)

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = (u0 + 1) % w
    v1 = np.clip(v0 + 1, 0, h - 1)

    du = u - u0
    dv = v - v0

    Ia = pano[v0, u0]
    Ib = pano[v0, u1]
    Ic = pano[v1, u0]
    Id = pano[v1, u1]

    out = (
        Ia * (1 - du)[..., None] * (1 - dv)[..., None]
        + Ib * du[..., None] * (1 - dv)[..., None]
        + Ic * (1 - du)[..., None] * dv[..., None]
        + Id * du[..., None] * dv[..., None]
    )

    out = np.clip(out, 0, 255).astype(np.uint8)
    if vertical_flip:
        out = np.flipud(out)

    Image.fromarray(out).save(output_path)
    return out


# ============================================================
# SIDE CLASSIFICATION
# ============================================================

def azimuth_to_vector(azimuth_deg: float) -> Tuple[float, float]:
    """Convert azimuth to a 2D unit vector."""
    az_rad = math.radians(azimuth_deg)
    return math.sin(az_rad), math.cos(az_rad)


def latlon_to_local_xy(
    lat_ref: float,
    lon_ref: float,
    lat_pt: float,
    lon_pt: float,
) -> Tuple[float, float]:
    """Convert a nearby geographic point to local East-North coordinates in meters."""
    r = 6371000.0
    lat_ref_rad = math.radians(lat_ref)
    lon_ref_rad = math.radians(lon_ref)
    lat_pt_rad = math.radians(lat_pt)
    lon_pt_rad = math.radians(lon_pt)

    dlat = lat_pt_rad - lat_ref_rad
    dlon = lon_pt_rad - lon_ref_rad

    x = r * dlon * math.cos((lat_ref_rad + lat_pt_rad) / 2.0)
    y = r * dlat
    return x, y


def calculate_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the azimuth from point A to point B."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    )

    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def classify_reference_side(
    target_lat: float,
    target_lon: float,
    reference_lat: float,
    reference_lon: float,
    direction_azimuth_deg: float,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Determine whether the reference point is to the left or right of the direction vector."""
    dx, dy = azimuth_to_vector(direction_azimuth_deg)
    rx, ry = latlon_to_local_xy(target_lat, target_lon, reference_lat, reference_lon)
    cross = dx * ry - dy * rx

    if abs(cross) <= tol:
        side = "aligned"
    elif cross > 0:
        side = "left"
    else:
        side = "right"

    return {
        "side": side,
        "cross": cross,
        "azimuth_target_to_reference": calculate_azimuth(target_lat, target_lon, reference_lat, reference_lon),
        "azimuth_reference_to_target": calculate_azimuth(reference_lat, reference_lon, target_lat, target_lon),
    }


def image_side(
    building_lat: float,
    building_lon: float,
    preview_lat: Optional[float],
    preview_lon: Optional[float],
    direction_azimuth: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Determine whether the building is on the left or right side of the camera direction."""
    if preview_lat is None or preview_lon is None or direction_azimuth is None:
        return None

    return classify_reference_side(
        target_lat=preview_lat,
        target_lon=preview_lon,
        reference_lat=building_lat,
        reference_lon=building_lon,
        direction_azimuth_deg=float(direction_azimuth),
    )


# ============================================================
# BUILDING OBJECT DETECTOR
# ============================================================

def load_building_detector() -> Optional[YOLO]:
    """Load YOLO only once to avoid reloading the model for every image."""
    weights_path = Path(BUILDING_DETECTOR_WEIGHTS)
    if not weights_path.exists():
        log_error(f"❌ YOLO weights file not found: {weights_path}")
        return None

    try:
        return YOLO(str(weights_path))
    except Exception as e:
        log_error(f"❌ Could not load YOLO model: {e}")
        return None


def object_detector_building(
    img_gsv: np.ndarray,
    model: YOLO,
    print_errors: bool = True,
) -> Optional[np.ndarray]:
    """Detect the main building in an image using YOLO and return the cropped image."""
    if img_gsv is None:
        if print_errors:
            log_error("❌ Input image is None.")
        return None

    if not isinstance(img_gsv, np.ndarray) or img_gsv.ndim != 3:
        if print_errors:
            log_error(f"❌ Invalid image input. Expected RGB NumPy array, got {type(img_gsv)}.")
        return None

    try:
        results = model.predict(img_gsv, device=DEVICE, verbose=False)[0]
        h, w, _ = img_gsv.shape
        class_names = model.names
        best_box = None
        best_conf = 0.0

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                conf = float(box.conf[0])

                if label == TARGET_CLASS and conf > CONF_THRESHOLD and conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy().astype(int)

        if best_box is None:
            if print_errors:
                log_error("❌ No building detected in image.")
            return None

        x1, y1, x2, y2 = best_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            if print_errors:
                log_error("❌ Invalid crop dimensions after clipping.")
            return None

        return img_gsv[y1:y2, x1:x2]

    except Exception as e:
        if print_errors:
            log_error(f"❌ Object detector failed: {e}")
        return None


def save_cropped_building(cropped_image: np.ndarray, output_path: Path) -> None:
    """Save the cropped building image."""
    Image.fromarray(cropped_image).save(output_path)


def plot_cropped_building(cropped_image: np.ndarray, title: str = "Detected Building Crop") -> None:
    """Plot the cropped building image generated by the object detector."""
    if cropped_image is None:
        log_error("❌ No cropped image to plot.")
        return

    plot_numpy_image(cropped_image, title=title)


def get_yaw_offsets(side_result: Optional[Dict[str, Any]]) -> List[int]:
    """Choose the orthophoto yaw offsets based on the side of the building."""
    if side_result is not None and side_result.get("side") == "right":
        return [180]
    return [0]


def try_detect_crop_from_orthophoto(
    img_orto: np.ndarray,
    model: YOLO,
    point_id: str,
    image_id: str,
    yaw_used: float,
    pitch_offset: float,
    label: str,
    crop_output_paths: List[str],
) -> Optional[np.ndarray]:
    """Run the detector on one orthophoto and save the crop if detected."""
    crop = object_detector_building(img_orto, model=model, print_errors=False)
    if crop is None:
        return None

    crop_path = CROPS_DIR / (
        f"{point_id}_{image_id}"
        f"_crop_{label}"
        f"_yaw{int(round(yaw_used))}"
        f"_p{int(pitch_offset)}.jpg"
    )

    save_cropped_building(crop, crop_path)
    crop_output_paths.append(str(crop_path))
    log_info(f"  Cropped building saved: {crop_path}")

    return crop


# ============================================================
# MAIN PROCESSING
# ============================================================

def empty_summary(point_id: str, building_lat: float, building_lon: float) -> Dict[str, Any]:
    """Create the default output row for one building point."""
    return {
        "id": point_id,
        "building_latitude": building_lat,
        "building_longitude": building_lon,
        "image_found": False,
        "image_id": None,
        "sequence_id": None,
        "captured_at": None,
        "is_pano": None,
        "compass_angle": None,
        "computed_compass_angle": None,
        "camera_type": None,
        "preview_latitude": None,
        "preview_longitude": None,
        "building_to_preview_distance_m": None,
        "camera_original_latitude": None,
        "camera_original_longitude": None,
        "camera_computed_latitude": None,
        "camera_computed_longitude": None,
        "building_to_camera_distance_m": None,
        "image_path": None,
        "orthophoto_paths": None,
        "building_crop_paths": None,
        "side_of_building_from_camera_direction": None,
        "side_cross_value": None,
        "azimuth_preview_to_building": None,
        "azimuth_building_to_preview": None,
        "error": None,
    }


def process_single_building(
    point_id: str,
    building_lat: float,
    building_lon: float,
    model: YOLO,
) -> Dict[str, Any]:
    """Process one building point and return summary results."""
    summary = empty_summary(point_id, building_lat, building_lon)

    try:
        result = find_nearest_image_from_building(
            building_lat=building_lat,
            building_lon=building_lon,
            max_offset_m=MAX_OFFSET_METERS,
            image_type=IMAGE_TYPE,
        )

        if result is None:
            summary["error"] = f"No image found within {MAX_OFFSET_METERS} m"
            return summary

        save_json(result["raw_feature"], RAW_DIR / f"{point_id}_nearest_feature.json")

        image_id = result["image_id"]
        summary.update(
            {
                "image_found": True,
                "image_id": image_id,
                "sequence_id": result.get("sequence_id"),
                "captured_at": result.get("captured_at"),
                "is_pano": result.get("is_pano"),
                "compass_angle": result.get("compass_angle"),
                "preview_latitude": result.get("preview_latitude"),
                "preview_longitude": result.get("preview_longitude"),
                "building_to_preview_distance_m": result.get("building_to_preview_distance_m"),
            }
        )

        metadata = get_image_metadata(image_id)
        save_json(metadata, RAW_DIR / f"{point_id}_{image_id}_metadata.json")

        props = metadata.get("properties", metadata)
        camera_pos = extract_camera_position(metadata)

        summary.update(
            {
                "computed_compass_angle": props.get("computed_compass_angle"),
                "camera_type": props.get("camera_type"),
                "camera_original_latitude": camera_pos["original_latitude"],
                "camera_original_longitude": camera_pos["original_longitude"],
                "camera_computed_latitude": camera_pos["computed_latitude"],
                "camera_computed_longitude": camera_pos["computed_longitude"],
            }
        )

        camera_lat = camera_pos["computed_latitude"] or camera_pos["original_latitude"]
        camera_lon = camera_pos["computed_longitude"] or camera_pos["original_longitude"]

        if camera_lat is not None and camera_lon is not None:
            summary["building_to_camera_distance_m"] = haversine_meters(
                building_lat,
                building_lon,
                camera_lat,
                camera_lon,
            )

        image_url = get_thumbnail_url(image_id, metadata)
        if not image_url:
            summary["error"] = "No thumbnail URL found"
            return summary

        image_path = IMAGES_DIR / f"{point_id}_{image_id}.jpg"
        download_image_to_pil(image_url).save(image_path)
        summary["image_path"] = str(image_path)

        direction_azimuth = summary["computed_compass_angle"] or summary["compass_angle"]
        side_result = image_side(
            building_lat=building_lat,
            building_lon=building_lon,
            preview_lat=summary["preview_latitude"],
            preview_lon=summary["preview_longitude"],
            direction_azimuth=direction_azimuth,
        )

        if side_result is not None:
            summary.update(
                {
                    "side_of_building_from_camera_direction": side_result["side"],
                    "side_cross_value": side_result["cross"],
                    "azimuth_preview_to_building": side_result["azimuth_target_to_reference"],
                    "azimuth_building_to_preview": side_result["azimuth_reference_to_target"],
                }
            )

        if ORTHO_ENABLE and bool(result.get("is_pano")):
            orthophoto_paths, crop_paths = process_orthophotos_and_crops(
                point_id=point_id,
                image_id=image_id,
                image_path=image_path,
                side_result=side_result,
                model=model,
            )

            if orthophoto_paths:
                summary["orthophoto_paths"] = "; ".join(orthophoto_paths)
            if crop_paths:
                summary["building_crop_paths"] = "; ".join(crop_paths)

        return summary

    except Exception as e:
        summary["error"] = str(e)
        return summary


def process_orthophotos_and_crops(
    point_id: str,
    image_id: str,
    image_path: Path,
    side_result: Optional[Dict[str, Any]],
    model: YOLO,
) -> Tuple[List[str], List[str]]:
    """Generate orthophotos and building crops for one panorama image."""
    orthophoto_paths: List[str] = []
    crop_paths: List[str] = []
    any_building_detected = False

    for yaw_offset in get_yaw_offsets(side_result):
        if STOP_AFTER_FIRST_SUCCESS and any_building_detected:
            break

        for pitch_offset in ORTHO_PITCH_OFFSETS:
            if STOP_AFTER_FIRST_SUCCESS and any_building_detected:
                break

            try:
                current_yaw = (ORTHO_BASE_YAW_DEG + yaw_offset) % 360
                current_pitch = max(-90.0, min(90.0, ORTHO_PITCH_DEG + pitch_offset))

                orthophoto_path = ORTHO_DIR / (
                    f"{point_id}_{image_id}_ortho_y{int(yaw_offset)}_p{int(pitch_offset)}.jpg"
                )

                img_orto = generate_orthophoto_from_360(
                    image_path=image_path,
                    output_path=orthophoto_path,
                    yaw_deg=current_yaw,
                    fov_deg=ORTHO_FOV_DEG,
                    pitch_deg=current_pitch,
                    out_h=ORTHO_OUT_H,
                    out_w=ORTHO_OUT_W,
                    vertical_flip=ORTHO_VERTICAL_FLIP,
                )
                orthophoto_paths.append(str(orthophoto_path))

                crop = try_detect_crop_from_orthophoto(
                    img_orto=img_orto,
                    model=model,
                    point_id=point_id,
                    image_id=image_id,
                    yaw_used=current_yaw,
                    pitch_offset=pitch_offset,
                    label="base",
                    crop_output_paths=crop_paths,
                )

                if crop is None and RETRY_WITH_ROTATED_YAW_IF_NO_BUILDING:
                    retry_yaw = (current_yaw + RETRY_YAW_DELTA_DEG) % 360
                    retry_orthophoto_path = ORTHO_DIR / (
                        f"{point_id}_{image_id}"
                        f"_ortho_y{int(yaw_offset)}"
                        f"_retryRot{int(RETRY_YAW_DELTA_DEG)}"
                        f"_p{int(pitch_offset)}.jpg"
                    )

                    retry_img_orto = generate_orthophoto_from_360(
                        image_path=image_path,
                        output_path=retry_orthophoto_path,
                        yaw_deg=retry_yaw,
                        fov_deg=ORTHO_FOV_DEG,
                        pitch_deg=current_pitch,
                        out_h=ORTHO_OUT_H,
                        out_w=ORTHO_OUT_W,
                        vertical_flip=ORTHO_VERTICAL_FLIP,
                    )
                    orthophoto_paths.append(str(retry_orthophoto_path))

                    crop = try_detect_crop_from_orthophoto(
                        img_orto=retry_img_orto,
                        model=model,
                        point_id=point_id,
                        image_id=image_id,
                        yaw_used=retry_yaw,
                        pitch_offset=pitch_offset,
                        label="retry",
                        crop_output_paths=crop_paths,
                    )

                if crop is not None:
                    any_building_detected = True
                else:
                    log_error(
                        f"❌ No building detected for point {point_id}, "
                        f"yaw_offset {yaw_offset}, pitch_offset {pitch_offset}, even after retry."
                    )

            except Exception as e:
                log_error(
                    f"❌ Image downloaded, but orthophoto/detector failed for point {point_id}, "
                    f"yaw offset {yaw_offset}, pitch offset {pitch_offset}: {e}"
                )

    return orthophoto_paths, crop_paths


def mapillary_image_source(
    access_token: str,
    csv_coordinates_path: str,
    return_dataframe: bool = True,
) -> pd.DataFrame | List[Dict[str, Any]]:
    """
    Run the complete Mapillary image extraction workflow as a function.

    Parameters
    ----------
    access_token : str
        Mapillary access token.
    csv_coordinates_path : str
        Path to the input CSV file with columns: id, latitude, longitude.
    return_dataframe : bool, optional
        If True, return a pandas DataFrame. If False, return a list of dictionaries.

    Returns
    -------
    pandas.DataFrame | list[dict]
        Processing results for each building point. The same information is also
        saved to RESULTS_CSV.
    """
    global MAPILLARY_ACCESS_TOKEN, YOLO_MODEL

    MAPILLARY_ACCESS_TOKEN = access_token

    ensure_dirs()

    if not MAPILLARY_ACCESS_TOKEN or "YOUR_ACCESS_TOKEN_HERE" in MAPILLARY_ACCESS_TOKEN:
        raise ValueError("Please set MAPILLARY_ACCESS_TOKEN first.")

    mly.set_access_token(MAPILLARY_ACCESS_TOKEN)

    YOLO_MODEL = load_building_detector()
    if YOLO_MODEL is None:
        raise FileNotFoundError(f"YOLO weights file not found: {BUILDING_DETECTOR_WEIGHTS}")

    points = load_coordinates_from_csv(csv_coordinates_path)
    results: List[Dict[str, Any]] = []

    log_info(f"Loaded {len(points)} building points from {csv_coordinates_path}")

    for item in points:
        point_id = item["id"]

        summary = process_single_building(
            point_id=point_id,
            building_lat=item["latitude"],
            building_lon=item["longitude"],
            model=YOLO_MODEL,
        )

        results.append(summary)

        if summary.get("error"):
            log_error(f"❌ Point {point_id}: {summary['error']}")

        time.sleep(SLEEP_SECONDS)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    log_info(f"\nDone. Results saved to: {RESULTS_CSV}")

    if return_dataframe:
        return results_df

    return results
