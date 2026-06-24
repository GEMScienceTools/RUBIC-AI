"""
get_building_orientation.py
===========================
This module provides utility functions to determine road orientation and retrieve
Google Street View imagery based on a geographic location.
"""

import requests
import math
import numpy as np
import cv2


def get_road_orientation(location):
    """
    Determine the road orientation (azimuth) near a specified location using the Google Roads API.
    """
    with open("methods/roads_api_key.txt", "r") as f:
        roads_api_key = f.read().strip()

    base_url = "https://roads.googleapis.com/v1/nearestRoads"
    params = {"points": f"{location[0]},{location[1]}", "key": roads_api_key}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "snappedPoints" in data and data["snappedPoints"]:
            snapped_point = data["snappedPoints"][0]
            road_lat = snapped_point["location"]["latitude"]
            road_lng = snapped_point["location"]["longitude"]
            orientation = compute_azimuth(location, (road_lat, road_lng))
            return orientation
        else:
            print("No road found near the location.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def compute_azimuth(point1, point2):
    """Compute the azimuth (bearing) between two geographic points."""
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    azimuth = math.degrees(math.atan2(x, y))
    return (azimuth + 360) % 360


def get_street_view_image(location, api_key, angle, pitch, fov):
    """
    Fetch a Google Street View image (outdoor-only) and generate its corresponding Maps URL.
    """
    # --- Metadata request (for year and indoor/outdoor detection) ---
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    meta_params = {
        "location": f"{location[0]},{location[1]}",
        "source": "outdoor",  # ✅ only request outdoor panoramas
        "key": api_key,
    }
    meta_response = requests.get(meta_url, params=meta_params)
    meta_data = meta_response.json()

    # Check if outdoor panorama is available
    if meta_data.get("status") != "OK":
        print(f"No outdoor panorama available at {location}. Status: {meta_data.get('status')}")
    
       ######################################################
       ############# NEW FUNCTION ###########################
       ######################################################
        found_close = False
        max_radius = 20
        step = 5
        show_debug = True

        with open("methods/gsv_api_key.txt", "r") as f:
            api_key = f.read().strip()

        meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        img_url = "https://maps.googleapis.com/maps/api/streetview"

        pano_id = None
        pano_lat, pano_lon, year = None, None, None

        for radius in range(step, max_radius + step, step):
            meta_params = {
                "location": f"{location[0]},{location[1]}",
                "radius": radius,
                "source": "outdoor",
                "key": api_key,
            }
            r = requests.get(meta_url, params=meta_params)
            meta = r.json()
            status = meta.get("status")

            if show_debug:
                print(f"Checking radius {radius} m → status={status}")

            if status == "OK":
                pano_id = meta.get("pano_id") or meta.get("panoId")
                pano_loc = meta.get("location", {})
                pano_lat, pano_lon = pano_loc.get("lat"), pano_loc.get("lng")
                found_close = True
                if "date" in meta:
                    year = meta["date"].split("-")[0]
                if show_debug:
                    print(f"✅ Outdoor pano found at {radius} m → ({pano_lat}, {pano_lon})")
                break

        if not found_close:
            print(f"⚠️ No outdoor pano found within {max_radius} m of {location}")
            return None, None, None

        # Determine if pano is to the right or left
        try:
            if pano_lon > location[1]:
                angle = 180
                side = "right"
            else:
                angle = 0
                side = "left"

            if show_debug:
                print(f"Pano is located to the {side} of the building → angle={angle}°")

            # Compute road orientation
            road_orientation = get_road_orientation(location)
            if road_orientation is None:
                road_orientation = 0
            heading = (road_orientation + angle + 180) % 360

            if show_debug:
                print(f"Road orientation: {road_orientation}")
                print(f"Final heading: {heading}")

            # Fetch image from pano ID
            params = {
                "size": "640x480",
                "pano": pano_id,
                "heading": heading,
                "pitch": pitch,
                "fov": fov,
                "source": "outdoor",
                "key": api_key,
            }

            resp = requests.get(img_url, params=params)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                np_arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                print(f"❌ Error fetching image: {resp.status_code}")
                img = None

            maps_url = (
                f"https://www.google.com/maps/@?api=1&map_action=pano"
                f"&viewpoint={pano_lat},{pano_lon}&heading={heading}&pitch=5&fov=120"
            )

            return maps_url, img, year

        except Exception as e:
            print(f"Error determining pano direction: {e}")
            return None, None, None

    else:
        # --- Extract available info ---
        year = None
        if "date" in meta_data:
            year = meta_data["date"].split("-")[0]
    
        # --- Compute road orientation ---
        road_orientation = get_road_orientation(location)
        if road_orientation is None:
            road_orientation = 0
        heading = (road_orientation + angle + 180) % 360
    
        # --- Secure API key load ---
        with open("methods/gsv_api_key.txt", "r") as f:
            api_key = f.read().strip()
    
        # --- Image capture parameters ---
        scale = 2
    
        # --- Base URLs ---
        base_url = "https://maps.googleapis.com/maps/api/streetview"
    
        # --- Define parameters for outdoor imagery ---
        params = {
            "size": "640x480",
            "location": f"{location[0]},{location[1]}",
            "heading": heading,
            "fov": fov,
            "pitch": pitch,
            "scale": scale,
            "source": "outdoor",
            "key": api_key,
        }
    
        # --- Build visualization URL ---
        maps_url = (
            f"https://www.google.com/maps/@?api=1&map_action=pano"
            f"&viewpoint={location[0]},{location[1]}&heading={heading}&pitch={pitch}&fov={fov}"
        )
    
        # --- Request the image ---
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            np_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        else:
            print("Error fetching image:", response.status_code)
            img = None
    
        return maps_url, img, year

