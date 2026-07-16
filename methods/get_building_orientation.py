"""Retrieve road orientation and Google Street View imagery.

This module provides utilities for calculating geographic azimuths, obtaining
nearby road orientation from the Google Roads API, and downloading outdoor
Google Street View images for a specified location.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import requests

ROADS_API_KEY_PATH = Path("methods/roads_api_key.txt")
ROADS_API_URL = "https://roads.googleapis.com/v1/nearestRoads"
STREET_VIEW_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
STREET_VIEW_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
MAPS_PANORAMA_URL = "https://www.google.com/maps/@?api=1&map_action=pano"
REQUEST_TIMEOUT = 30
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

def compute_azimuth(point1, point2):
    """Compute the azimuth between two geographic points.

    Parameters
    ----------
    point1 : tuple[float, float]
        Latitude and longitude of the starting point in degrees.
    point2 : tuple[float, float]
        Latitude and longitude of the destination point in degrees.

    Returns
    -------
    float
        Bearing from the first point to the second in degrees, measured
        clockwise from north within the interval ``[0, 360)``.
    """
    lat1, lon1 = map(math.radians, point1)
    lat2, lon2 = map(math.radians, point2)
    longitude_difference = lon2 - lon1

    x_component = math.sin(longitude_difference) * math.cos(lat2)
    y_component = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(longitude_difference)
    azimuth = math.degrees(math.atan2(x_component, y_component))
    return (azimuth + 360) % 360


def get_road_orientation(location):
    """Determine the road orientation near a geographic location.

    Parameters
    ----------
    location : tuple[float, float]
        Latitude and longitude of the target location.

    Returns
    -------
    float or None
        Estimated road orientation in degrees, or ``None`` when no nearby
        road can be found or the API request fails.
    """
    roads_api_key = _read_api_key(ROADS_API_KEY_PATH)
    params = {
        "points": f"{location[0]},{location[1]}",
        "key": roads_api_key,
    }

    try:
        response = requests.get(
            ROADS_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Unable to retrieve the nearest road: {error}")
        return None

    snapped_points = response.json().get("snappedPoints", [])
    if not snapped_points:
        print("No road found near the location.")
        return None

    snapped_location = snapped_points[0]["location"]
    road_point = (
        snapped_location["latitude"],
        snapped_location["longitude"],
    )
    return compute_azimuth(location, road_point)


def _request_metadata(location, api_key, radius=None):
    """Request Street View metadata for an outdoor panorama.

    Parameters
    ----------
    location : tuple[float, float]
        Latitude and longitude of the target location.
    api_key : str
        Google Street View API key.
    radius : int or None, optional
        Search radius in metres. When omitted, the API default is used.

    Returns
    -------
    dict
        Decoded Street View metadata response.
    """
    params = {
        "location": f"{location[0]},{location[1]}",
        "source": "outdoor",
        "key": api_key,
    }
    if radius is not None:
        params["radius"] = radius

    try:
        response = requests.get(
            STREET_VIEW_METADATA_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Unable to retrieve Street View metadata: {error}")
        return {}

    return response.json()


def _find_nearby_panorama(location, api_key, show_debug=True):
    """Find the nearest outdoor panorama within the configured radius.

    Parameters
    ----------
    location : tuple[float, float]
        Latitude and longitude of the target location.
    api_key : str
        Google Street View API key.
    show_debug : bool, optional
        Whether to print information about each search radius.

    Returns
    -------
    tuple[str, float, float, str | None] or None
        Panorama ID, latitude, longitude, and capture year, or ``None`` when
        no panorama is available.
    """
    radii = range(
        DEFAULT_RADIUS_STEP,
        DEFAULT_MAX_RADIUS + DEFAULT_RADIUS_STEP,
        DEFAULT_RADIUS_STEP,
    )

    for radius in radii:
        metadata = _request_metadata(location, api_key, radius)
        status = metadata.get("status")
        if show_debug:
            print(f"Checking radius {radius} m: status={status}")

        if status != "OK":
            continue

        panorama_id = metadata.get("pano_id") or metadata.get("panoId")
        panorama_location = metadata.get("location", {})
        panorama_latitude = panorama_location.get("lat")
        panorama_longitude = panorama_location.get("lng")
        year = metadata.get("date", "").split("-")[0] or None

        if show_debug:
            print(
                "Outdoor panorama found at "
                f"{radius} m: ({panorama_latitude}, {panorama_longitude})"
            )

        return (
            panorama_id,
            panorama_latitude,
            panorama_longitude,
            year,
        )

    print(f"No outdoor panorama found within {DEFAULT_MAX_RADIUS} m of {location}.")
    return None


def _calculate_heading(location, angle):
    """Calculate the Street View heading for a location and angle.

    Parameters
    ----------
    location : tuple[float, float]
        Latitude and longitude of the target location.
    angle : float
        Additional viewing angle in degrees.

    Returns
    -------
    float
        Final Street View heading in degrees.
    """
    try:
        road_orientation = get_road_orientation(location)
        h_angle = (road_orientation + angle + 180) % 360
        api_error = False
    except TypeError:
        road_orientation = 0
        h_angle = (road_orientation + angle + 180) % 360
        api_error = True

    return h_angle , api_error


def _decode_street_view_image(api_key, heading, pitch, fov, **location):
    """Decode a Street View image.

    Parameters
    ----------
    api_key : str
        Google Street View API key.
    heading : float
        Camera heading in degrees.
    pitch : float
        Camera pitch in degrees.
    fov : float
        Horizontal field of view in degrees.
    **location : str
        Either a ``location`` coordinate string or a ``pano`` identifier.

    Returns
    -------
    numpy.ndarray or None
        Decoded OpenCV image, or ``None`` when the request fails.
    """
    params = {
        "size": "640x480",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "source": "outdoor",
        "key": api_key,
        **location,
    }
    if "location" in location:
        params["scale"] = 2

    try:
        response = requests.get(
            STREET_VIEW_IMAGE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Unable to fetch the Street View image: {error}")
        return None

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        print("The Street View response did not contain an image.")
        return None

    image_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def _build_maps_url(viewpoint, heading, pitch, fov):
    """Build a Google Maps panorama URL.

    Parameters
    ----------
    viewpoint : tuple[float, float]
        Panorama latitude and longitude.
    heading : float
        Camera heading in degrees.
    pitch : float
        Camera pitch in degrees.
    fov : float
        Horizontal field of view in degrees.

    Returns
    -------
    str
        Google Maps URL for visualizing the panorama.
    """
    return (
        f"{MAPS_PANORAMA_URL}"
        f"&viewpoint={viewpoint[0]},{viewpoint[1]}"
        f"&heading={heading}&pitch={pitch}&fov={fov}"
    )


def _get_image_from_nearby_panorama(location, api_key, pitch, fov):
    """Retrieve an image from a nearby outdoor panorama.

    Parameters
    ----------
    location : tuple[float, float]
        Latitude and longitude of the target location.
    api_key : str
        Google Street View API key.
    pitch : float
        Camera pitch in degrees.
    fov : float
        Horizontal field of view in degrees.

    Returns
    -------
    tuple[str | None, numpy.ndarray | None, str | None]
        Maps URL, decoded image, and capture year.
    """
    panorama = _find_nearby_panorama(location, api_key)
    if panorama is None:
        return None, None, None

    panorama_id, panorama_latitude, panorama_longitude, year = panorama
    angle = 180 if panorama_longitude > location[1] else 0
    heading = _calculate_heading(location, angle)[0]
    image = _decode_street_view_image(
        api_key,
        heading,
        pitch,
        fov,
        pano=panorama_id,
    )
    maps_url = _build_maps_url(
        (panorama_latitude, panorama_longitude),
        heading,
        pitch,
        fov,
    )
    return maps_url, image, year


def get_street_view_image(location, api_key, angle, pitch, fov):
    """Fetch an outdoor Google Street View image.

    The function first requests imagery at the exact location. If no outdoor
    panorama is available, it searches incrementally within 20 metres and
    retrieves an image from the nearest panorama found.

    Parameters
    ----------
    location : tuple[float, float]
        Latitude and longitude of the target location.
    api_key : str
        Google Street View API key.
    angle : float
        Additional camera angle in degrees.
    pitch : float
        Camera pitch in degrees.
    fov : float
        Horizontal field of view in degrees.

    Returns
    -------
    tuple[str | None, numpy.ndarray | None, str | None]
        Google Maps URL, decoded OpenCV image, and panorama capture year.
    """
    metadata = _request_metadata(location, api_key)
    if metadata.get("status") != "OK":
        status = metadata.get("status")
        print(f"No outdoor panorama available at {location}. Status: {status}")
        return _get_image_from_nearby_panorama(
            location,
            api_key,
            pitch,
            fov,
        )

    year = metadata.get("date", "").split("-")[0] or None
    heading, roads_api_error = _calculate_heading(location, angle)
    image = _decode_street_view_image(
        api_key,
        heading,
        pitch,
        fov,
        location=f"{location[0]},{location[1]}",
    )
    maps_url = _build_maps_url(location, heading, pitch, fov)
    return maps_url, image, year, roads_api_error
