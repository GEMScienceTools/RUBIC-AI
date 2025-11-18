import requests
import math
import numpy as np
import cv2

# ============================================================
# === Helper functions: Road orientation + azimuth ===========
# ============================================================

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
            print("⚠️ No road found near the location.")
            return None
    else:
        print(f"❌ Roads API error: {response.status_code}, {response.text}")
        return None


def compute_azimuth(point1, point2):
    """
    Compute azimuth (bearing) in degrees between two geographic points.
    """
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    azimuth = math.degrees(math.atan2(x, y))
    return (azimuth + 360) % 360


# ============================================================
# === Main function: Find outdoor image + orientation ========
# ============================================================

def get_street_view_image(
    location,
    found_close=False,
    max_radius=50,
    step=20,
    show_debug=True
):
    """
    Fetch the nearest OUTDOOR Street View image aligned with the road orientation.
    Automatically adjusts the angle (0° or 180°) depending on whether
    the pano is to the right or left of the requested location.
    """

    # Load GSV API key
    with open("methods/gsv_api_key.txt", "r") as f:
        api_key = f.read().strip()

    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    img_url = "https://maps.googleapis.com/maps/api/streetview"

    # --- Step 1: Search nearest outdoor pano ---
    pano_id = None
    pano_lat, pano_lon, used_radius, year = None, None, None, None

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
            found_close, used_radius = True, radius
            if "date" in meta:
                year = meta["date"].split("-")[0]
            if show_debug:
                print(f"✅ Outdoor pano found at {radius} m → ({pano_lat}, {pano_lon})")
            break

    if not found_close:
        print(f"⚠️ No outdoor pano found within {max_radius} m of {location}")
        return None, None, None, None, False, None, None, None

    # --- Step 2: Determine if pano is to the right or left ---
    try:
        if pano_lon > location[1]:
            angle = 180   # pano is to the right → look towards it directly
            side = "right"
        else:
            angle = 0 # pano is to the left → turn 180°
            side = "left"
    
        if show_debug:
            print(f"Pano is located to the {side} of the building → angle={angle}°")
    
        # --- Step 3: Compute road orientation ---
        road_orientation = get_road_orientation(location)
        heading = ((road_orientation or 0) + angle + 180) % 360
    
        if show_debug:
            print(f"Road orientation: {road_orientation}")
            print(f"Final heading: {heading}")
    
        # --- Step 4: Fetch image from pano ID ---
        params = {
            "size": "640x480",
            "pano": pano_id,
            "heading": heading,
            "pitch": 5,
            "fov": 120,
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
    
        # --- Step 5: Construct Google Maps URL for reference ---
        maps_url = (
            f"https://www.google.com/maps/@?api=1&map_action=pano"
            f"&viewpoint={pano_lat},{pano_lon}&heading={heading}&pitch=5&fov=120"
        )
    
        return maps_url, img, year, (pano_lat, pano_lon), found_close, used_radius, side, angle
    except:
        pass

# ============================================================
# === Example usage ==========================================
# ============================================================

if __name__ == "__main__":
    coord = (6.24902626 , -75.59203566)  # Lisbon example

    maps_url, img, year, pano_coords, found_close, radius, side, angle = get_street_view_image(
        location=coord,
        found_close=False,
        max_radius=50,
        step=20,
        show_debug=True
    )

    print("\n=== SUMMARY ===")
    print("Found outdoor pano:", found_close)
    print("Used radius (m):", radius)
    print("Outdoor pano coords:", pano_coords)
    print("Year:", year)
    print("Side of building:", side)
    print("Auto-selected angle:", angle)
    print("Google Maps URL:", maps_url)

    if img is not None:
        cv2.imshow("Street View", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

