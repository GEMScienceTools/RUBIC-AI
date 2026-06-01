from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import mapillary.interface as mly
import numpy as np
import requests
from PIL import Image


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_OUTPUT_DIR = Path("mapillary_single_result")
DEFAULT_MAX_OFFSET_METERS = 50.0
DEFAULT_IMAGE_TYPE = "pano"
DEFAULT_THUMBNAIL_RESOLUTION = 2048
DEFAULT_REQUEST_TIMEOUT = 30

DEFAULT_ORTHO_FOV_DEG = 120
DEFAULT_ORTHO_PITCH_DEG = 0.0
DEFAULT_ORTHO_OUT_H = 1024
DEFAULT_ORTHO_OUT_W = 1024
DEFAULT_ORTHO_VERTICAL_FLIP = False
DEFAULT_ORTHO_BASE_YAW_DEG = 180

PRINT_PROGRESS = False


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

def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create output folders and return their paths."""
    folders = {
        "output": output_dir,
        "images": output_dir / "images",
        "orthophotos": output_dir / "orthophotos",
        "raw": output_dir / "raw_json",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


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
    earth_radius_m = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return earth_radius_m * c


def meters_to_degree_offsets(lat_deg: float, meters: float) -> Tuple[float, float]:
    """Approximate meter-to-degree conversion around a given latitude."""
    delta_lat = meters / 111320.0
    cos_lat = math.cos(math.radians(lat_deg))

    if abs(cos_lat) < 1e-12:
        delta_lon = meters / 111320.0
    else:
        delta_lon = meters / (111320.0 * cos_lat)

    return delta_lat, delta_lon


def build_bbox(lat: float, lon: float, radius_m: float) -> Dict[str, float]:
    """Build a small bounding box around one coordinate."""
    delta_lat, delta_lon = meters_to_degree_offsets(lat, radius_m * 2.0)

    return {
        "west": lon - delta_lon,
        "south": lat - delta_lat,
        "east": lon + delta_lon,
        "north": lat + delta_lat,
    }


# ============================================================
# MAPILLARY QUERY
# ============================================================

def query_images_in_bbox(
    latitude: float,
    longitude: float,
    search_radius_m: float,
    image_type: str,
) -> Dict[str, Any]:
    """Query Mapillary images around one coordinate."""
    bbox = build_bbox(latitude, longitude, search_radius_m)
    bbox_image_type = "all" if image_type == "both" else image_type

    response = mly.images_in_bbox(
        bbox=bbox,
        image_type=bbox_image_type,
    )

    return parse_response(response)


def find_nearest_image_from_coordinate(
    latitude: float,
    longitude: float,
    max_offset_m: float,
    image_type: str,
) -> Optional[Dict[str, Any]]:
    """Find the nearest Mapillary image within max_offset_m from one coordinate."""
    data = query_images_in_bbox(
        latitude=latitude,
        longitude=longitude,
        search_radius_m=max_offset_m,
        image_type=image_type,
    )

    features = data.get("features", [])
    if not features:
        return None

    best_result = None
    best_distance = float("inf")

    for feature in features:
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue

        image_lon, image_lat = coords[0], coords[1]
        props = feature.get("properties", {})
        is_pano = bool(props.get("is_pano", False))

        if image_type == "pano" and not is_pano:
            continue

        if image_type == "flat" and is_pano:
            continue

        distance_m = haversine_meters(
            latitude,
            longitude,
            image_lat,
            image_lon,
        )

        if distance_m <= max_offset_m and distance_m < best_distance:
            best_distance = distance_m
            best_result = {
                "image_id": str(props.get("id")),
                "sequence_id": props.get("sequence_id"),
                "captured_at": props.get("captured_at"),
                "compass_angle": props.get("compass_angle"),
                "is_pano": is_pano,
                "preview_latitude": image_lat,
                "preview_longitude": image_lon,
                "building_to_preview_distance_m": distance_m,
                "raw_feature": feature,
            }

    return best_result


def get_image_metadata(image_id: str) -> Dict[str, Any]:
    """Request Mapillary image metadata."""
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


def get_thumbnail_url(
    image_id: str,
    metadata: Dict[str, Any],
    thumbnail_resolution: int,
) -> Optional[str]:
    """Get the best available Mapillary image URL."""
    try:
        thumb_url = mly.image_thumbnail(
            image_id=image_id,
            resolution=thumbnail_resolution,
        )

        if isinstance(thumb_url, str) and thumb_url.strip():
            return thumb_url

    except Exception as e:
        log_error(f"❌ image_thumbnail() failed for {image_id}: {e}")

    props = metadata.get("properties", metadata)

    for field in [
        "thumb_original_url",
        "thumb_2048_url",
        "thumb_1024_url",
        "thumb_256_url",
    ]:
        value = props.get(field)
        if value:
            return value

    return None


def download_image_to_pil(url: str, request_timeout: int) -> Image.Image:
    """Download an image URL and return it as a PIL RGB image."""
    response = requests.get(url, timeout=request_timeout)
    response.raise_for_status()

    return Image.open(io.BytesIO(response.content)).convert("RGB")


def ensure_uint8_rgb(image_array: np.ndarray) -> np.ndarray:
    """
    Ensure that an image matrix is returned as uint8 RGB with shape (H, W, 3).

    Notes
    -----
    The function does not swap channels. It only validates that the array is a
    standard RGB image matrix generated from PIL, where channel order is:
        image_array[:, :, 0] = Red
        image_array[:, :, 1] = Green
        image_array[:, :, 2] = Blue
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"Expected a NumPy array, got {type(image_array)}.")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3), got {image_array.shape}.")

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    return image_array


def convert_rgb_to_bgr(image_array: np.ndarray) -> np.ndarray:
    """Convert an RGB image matrix to BGR, only if OpenCV-style output is needed."""
    image_array = ensure_uint8_rgb(image_array)
    return image_array[:, :, ::-1].copy()


# ============================================================
# ORTHOPHOTO GENERATION
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
    """
    Convert a 360 equirectangular image into a perspective/orthophoto image.

    The orthophoto is saved to output_path and returned as an RGB matrix.
    """
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

    # Pitch rotation
    x1 = cp * x_cam + sp * z_cam
    y1 = y_cam
    z1 = -sp * x_cam + cp * z_cam

    # Yaw rotation
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

    orthophoto = np.clip(out, 0, 255).astype(np.uint8)

    if vertical_flip:
        orthophoto = np.flipud(orthophoto)

    # Save with PIL because PIL expects RGB channel order.
    # This avoids the common OpenCV issue where RGB arrays are interpreted as BGR.
    orthophoto = ensure_uint8_rgb(orthophoto)
    Image.fromarray(orthophoto, mode="RGB").save(output_path)

    return orthophoto


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
    earth_radius_m = 6371000.0

    lat_ref_rad = math.radians(lat_ref)
    lon_ref_rad = math.radians(lon_ref)
    lat_pt_rad = math.radians(lat_pt)
    lon_pt_rad = math.radians(lon_pt)

    dlat = lat_pt_rad - lat_ref_rad
    dlon = lon_pt_rad - lon_ref_rad

    x = earth_radius_m * dlon * math.cos((lat_ref_rad + lat_pt_rad) / 2.0)
    y = earth_radius_m * dlat

    return x, y


def calculate_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate azimuth from point A to point B."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    )

    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def image_side(
    building_lat: float,
    building_lon: float,
    preview_lat: Optional[float],
    preview_lon: Optional[float],
    direction_azimuth: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Determine whether the coordinate is on the left or right side of the camera direction."""
    if preview_lat is None or preview_lon is None or direction_azimuth is None:
        return None

    dx, dy = azimuth_to_vector(float(direction_azimuth))
    rx, ry = latlon_to_local_xy(
        preview_lat,
        preview_lon,
        building_lat,
        building_lon,
    )

    cross = dx * ry - dy * rx

    if abs(cross) <= 1e-9:
        side = "aligned"
    elif cross > 0:
        side = "left"
    else:
        side = "right"

    return {
        "side": side,
        "cross": cross,
        "azimuth_preview_to_building": calculate_azimuth(
            preview_lat,
            preview_lon,
            building_lat,
            building_lon,
        ),
        "azimuth_building_to_preview": calculate_azimuth(
            building_lat,
            building_lon,
            preview_lat,
            preview_lon,
        ),
    }


def select_yaw_offset(side_result: Optional[Dict[str, Any]]) -> int:
    """Select the yaw offset using the same left/right logic as the previous script."""
    if side_result is not None and side_result.get("side") == "right":
        return 180

    return 0


# ============================================================
# MAIN FUNCTION TO CALL FROM YOUR SCRIPT
# ============================================================

def mapillary_image_source(
    access_token: str,
    latitude: float,
    longitude: float,
    point_id: str = "point_1",
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    max_offset_m: float = DEFAULT_MAX_OFFSET_METERS,
    image_type: str = DEFAULT_IMAGE_TYPE,
    image_number: Optional[Union[int, str]] = None,
    save_original_image: bool = True,
    save_json_files: bool = False,
    return_info: bool = False,
    return_color_order: Literal["RGB", "BGR"] = "RGB",
    ortho_fov_deg: float = DEFAULT_ORTHO_FOV_DEG,
    ortho_pitch_deg: float = DEFAULT_ORTHO_PITCH_DEG,
    ortho_base_yaw_deg: float = DEFAULT_ORTHO_BASE_YAW_DEG,
    ortho_out_h: int = DEFAULT_ORTHO_OUT_H,
    ortho_out_w: int = DEFAULT_ORTHO_OUT_W,
    ortho_vertical_flip: bool = DEFAULT_ORTHO_VERTICAL_FLIP,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Retrieve one Mapillary 360 image for one coordinate, generate an orthophoto,
    save the orthophoto locally, and return the orthophoto as a 3D RGB matrix.
    """

    if not access_token or "YOUR_ACCESS_TOKEN_HERE" in access_token:
        raise ValueError("Please provide a valid Mapillary access token.")

    if return_color_order not in {"RGB", "BGR"}:
        raise ValueError('return_color_order must be either "RGB" or "BGR".')

    output_dir = Path(output_dir)
    folders = ensure_dirs(output_dir)

    mly.set_access_token(access_token)

    info: Dict[str, Any] = {
        "point_id": point_id,
        "building_latitude": latitude,
        "building_longitude": longitude,
        "image_id": None,
        "image_number": image_number,
        "image_path": None,
        "orthophoto_path": None,
        "returned_image_type": "orthophoto",
        "color_order": return_color_order,
        "matrix_shape": None,
        "error": None,
        "ortho_fov_deg": ortho_fov_deg,
        "ortho_pitch_deg": ortho_pitch_deg,
        "ortho_base_yaw_deg": ortho_base_yaw_deg,
    }

    try:
        nearest = find_nearest_image_from_coordinate(
            latitude=latitude,
            longitude=longitude,
            max_offset_m=max_offset_m,
            image_type=image_type,
        )

        if nearest is None:
            raise RuntimeError(f"No Mapillary image found within {max_offset_m} m.")

        if not nearest.get("is_pano"):
            raise RuntimeError("The selected Mapillary image is not a 360/pano image.")

        image_id = nearest["image_id"]
        info["image_id"] = image_id
        info["building_to_preview_distance_m"] = nearest.get(
            "building_to_preview_distance_m"
        )

        if save_json_files:
            save_json(
                nearest["raw_feature"],
                folders["raw"] / f"{point_id}_nearest_feature.json",
            )

        metadata = get_image_metadata(image_id)
        props = metadata.get("properties", metadata)

        if save_json_files:
            save_json(
                metadata,
                folders["raw"] / f"{point_id}_{image_id}_metadata.json",
            )

        image_url = get_thumbnail_url(
            image_id=image_id,
            metadata=metadata,
            thumbnail_resolution=DEFAULT_THUMBNAIL_RESOLUTION,
        )

        if not image_url:
            raise RuntimeError("No thumbnail URL found for the selected image.")

        pil_image = download_image_to_pil(
            url=image_url,
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
        )

        image_path = folders["images"] / f"{point_id}_{image_id}.jpg"
        pil_image.save(image_path)

        if save_original_image:
            info["image_path"] = str(image_path)

        direction_azimuth = props.get("computed_compass_angle")
        if direction_azimuth is None:
            direction_azimuth = nearest.get("compass_angle")

        side_result = image_side(
            building_lat=latitude,
            building_lon=longitude,
            preview_lat=nearest.get("preview_latitude"),
            preview_lon=nearest.get("preview_longitude"),
            direction_azimuth=direction_azimuth,
        )

        if side_result is not None:
            info.update(side_result)

        yaw_offset = select_yaw_offset(side_result)

        # Updated line:
        # The base yaw can now be controlled from the function input.
        yaw_deg = (ortho_base_yaw_deg + yaw_offset) % 360

        info["yaw_offset"] = yaw_offset
        info["yaw_deg"] = yaw_deg

        if image_number is not None:
            orthophoto_filename = f"{image_number}.jpg"
        else:
            orthophoto_filename = (
                f"{point_id}_{image_id}_orthophoto"
                f"_yaw{int(round(yaw_deg))}"
                f"_pitch{int(round(ortho_pitch_deg))}.jpg"
            )

        orthophoto_path = folders["orthophotos"] / orthophoto_filename

        orthophoto_matrix = generate_orthophoto_from_360(
            image_path=image_path,
            output_path=orthophoto_path,
            yaw_deg=yaw_deg,
            fov_deg=ortho_fov_deg,
            pitch_deg=ortho_pitch_deg,
            out_h=ortho_out_h,
            out_w=ortho_out_w,
            vertical_flip=ortho_vertical_flip,
        )

        # The matrix generated by this script is RGB because it comes from PIL.
        # If another part of your workflow needs OpenCV format, request BGR explicitly.
        orthophoto_matrix = ensure_uint8_rgb(orthophoto_matrix)

        if return_color_order == "BGR":
            returned_matrix = convert_rgb_to_bgr(orthophoto_matrix)
        else:
            returned_matrix = orthophoto_matrix

        info["orthophoto_path"] = str(orthophoto_path)
        info["matrix_shape"] = returned_matrix.shape
        info["color_order"] = return_color_order

        log_info(f"Orthophoto saved to: {orthophoto_path}")

        if return_info:
            return returned_matrix, info

        return returned_matrix

    except Exception as e:
        info["error"] = str(e)
        log_error(f"❌ {info['error']}")
        raise