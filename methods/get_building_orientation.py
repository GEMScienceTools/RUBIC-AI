"""Retrieve road orientation and Google Street View imagery.

This module provides utilities for calculating geographic azimuths, obtaining
nearby road orientation from the Google Roads API, and downloading outdoor
Google Street View images for a specified location.
"""

import math
from typing import Any

import cv2
import numpy as np
import requests


STREET_VIEW_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
STREET_VIEW_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
MAPS_PANORAMA_URL = "https://www.google.com/maps/@?api=1&map_action=pano"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RADIUS = 20
DEFAULT_RADIUS_STEP = 5

def _read_api_key(path):
    """Read an API key from a text file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the text file containing the API key.

    Returns
    -------
    str
        API key with surrounding whitespace removed.
    """
    with path.open(encoding="utf-8") as file:
        return file.read().strip()

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


def decode_street_view_image(
    params: dict[str, Any],
) -> np.ndarray | None:
    """Decode one Street View image."""
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

    # A heading is meaningful only between two distinct geographic points.
    # The Street View metadata position is the camera location, whereas
    # `location` is the target building location.
    if panorama_latitude is None or panorama_longitude is None:
        print("Street View metadata has no panorama coordinates; cannot aim at building.")
        return None, None, year

    panorama_coordinates = (
        float(panorama_latitude),
        float(panorama_longitude),
    )
    if not all(math.isfinite(value) for value in (*panorama_coordinates, *location)):
        print("Invalid panorama or building coordinates; cannot compute heading.")
        return None, None, year

    if (math.isclose(panorama_coordinates[0], location[0], abs_tol=1e-9)
            and math.isclose(panorama_coordinates[1], location[1], abs_tol=1e-9)):
        print("Panorama coincides with the target coordinates; heading is undefined.")
        return None, None, year

    base_heading = compute_azimuth(panorama_coordinates, location)
    heading = (base_heading + angle) % 360

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
        # Pin the image request to the camera position returned by metadata.
        # Requesting the building location could select a different panorama.
        image_params["location"] = (
            f"{panorama_coordinates[0]},{panorama_coordinates[1]}"
        )

    try:
        image = decode_street_view_image(image_params)
    except requests.RequestException as error:
        print(f"Street View image request failed: {error}")
        return None, None, year

    viewpoint_latitude, viewpoint_longitude = panorama_coordinates
    maps_url = (
        "https://www.google.com/maps/@?api=1&map_action=pano"
        f"&viewpoint={viewpoint_latitude},{viewpoint_longitude}"
        f"&heading={heading}&pitch={pitch}&fov={field_of_view}"
    )
    return maps_url, image, year, base_heading
