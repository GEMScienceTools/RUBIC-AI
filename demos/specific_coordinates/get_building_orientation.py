import requests
import math
import numpy as np
import cv2
from pathlib import Path

root_dir = Path(__file__).parent.resolve()
gsv_dir = (root_dir / '..' / '..' / 'methods').resolve()
    
def get_road_orientation(location):
    """
    Determine the road orientation (azimuth) near a specified location using the Google Roads API.

    This function queries the Google Roads API to find the nearest road to the given geographic 
    location (latitude, longitude). If a snapped road point is found, it calculates the azimuth 
    (orientation) of the road segment relative to the specified location.

    Args:
        location (tuple): A tuple containing the latitude and longitude of the target location 
        (e.g., `(latitude, longitude)`).

    Returns:
        float: The road azimuth in degrees, or `None` if no road is found or an error occurs.

    Effects:
        - Sends a request to the Google Roads API to find the nearest road.
        - Prints messages to the console for debugging or error handling.

    Notes:
        - Requires a valid Google Roads API key (`api_key`) to make requests.
        - Uses a helper function `compute_azimuth` to calculate the azimuth between two points.
        - Handles API errors and missing road data gracefully.
    """
    # Model parameters   
    with open(gsv_dir / "roads_api_key.txt", "r") as f:
        roads_api_key = f.read().strip()
    base_url = "https://roads.googleapis.com/v1/nearestRoads"
    params = {
        "points": f"{location[0]},{location[1]}",
        "key": roads_api_key,
    }
    # Request for Google roads metadata
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "snappedPoints" in data and data["snappedPoints"]:
            # Calculate azimuth from geometry data if available
            snapped_point = data["snappedPoints"][0]
            road_lat = snapped_point["location"]["latitude"]
            road_lng = snapped_point["location"]["longitude"]
            # Compute azimuth based on snapped points
            orientation = compute_azimuth(location, (road_lat, road_lng))
            return orientation
        else:
            print("No road found near the location.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def compute_azimuth(point1, point2):
    """
    Compute the azimuth (bearing) between two geographic points.

    This function calculates the azimuth in degrees between two points on the Earth's surface, 
    given their latitude and longitude. The azimuth represents the angle between the north direction 
    and the line connecting the two points, measured clockwise.

    Args:
        point1 (tuple): A tuple containing the latitude and longitude of the first point 
        (e.g., `(latitude1, longitude1)`).
        point2 (tuple): A tuple containing the latitude and longitude of the second point 
        (e.g., `(latitude2, longitude2)`).

    Returns:
        float: The azimuth in degrees, ranging from 0° to 360°.

    Notes:
        - The calculation uses the haversine formula and trigonometric functions.
        - The azimuth is normalized to ensure it falls within the range of 0° to 360°.
    """
    # Coordinates of the diffents points of analysis
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    # Determine difference in longitude and components for azimuth calculation
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    # Determine azimuth
    azimuth = math.degrees(math.atan2(x, y))
    return (azimuth + 360) % 360

def get_street_view_image(location, api_key, angle):
    """
    Fetch a Google Street View image and generate its corresponding Maps URL.

    This function requests a Street View image from the Google Street View Static API 
    for a given location and angle. It also computes the road orientation to adjust 
    the camera heading direction accordingly.

    Args:
        location (tuple): A tuple containing latitude and longitude as (lat, lon).
        api_key (str): Google Maps API key for authentication.
        angle (int or float): Angle adjustment for the Street View heading.

    Returns:
        tuple:
            - maps_url (str): A URL to visualize the location in Google Maps Street View.
            - img (np.ndarray or None): The fetched Street View image in OpenCV (BGR) format, 
              or `None` if the request fails.

    Effects:
        - Calls `get_road_orientation(location)` to determine the road direction.
        - Sends an HTTP request to the Google Street View API.
        - Converts the image response to a NumPy array and decodes it into an OpenCV image.
        - Prints an error message if the API request fails.

    Notes:
        - The API key is hardcoded in this function for now but should be stored securely 
          in a production environment.
        - The function ensures that the heading direction remains within [0, 360] degrees.
        - The field of view (FOV) is set to 120° for a wide-angle capture.
        - Uses a scale factor of 2 for higher image resolution.
        - Returns both the image and a Google Maps URL for additional visualization.
    """

    # Retrieve the road orientation at the given location
    road_orientation = get_road_orientation(location)
    # Compute the heading direction for the Street View API (ensuring it's within [0, 360] degrees)
    heading = (road_orientation + angle + 180) % 360
    # Google Maps API key (should be kept secure and not hardcoded in production)
    with open(gsv_dir / "gsv_api_key.txt", "r") as f:
        api_key = f.read().strip()

    # Define image capture parameters
    fov = 120          # Field of view (wider angle for more coverage)
    pitch = 5          # Camera pitch angle (tilt up/down; 0 is level with horizon)
    scale = 2          # Scale factor (2 = high resolution)

    # Base URL for Google Street View API
    base_url = "https://maps.googleapis.com/maps/api/streetview"

    # Define query parameters for the Street View API request
    params = {
        "size": "640x480",  # Image resolution
        "location": f"{location[0]},{location[1]}",  # Latitude and longitude of the location
        "heading": heading,  # Adjusted heading direction
        "fov": fov,  # Field of view
        "pitch": pitch,  # Camera tilt
        "scale": scale,  # High-resolution mode
        "key": api_key,  # API key for authentication
    }

    # Generate a Google Maps Street View URL for visualization in a browser
    maps_url = (f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={location[0]},{location[1]}"
                f"&heading={heading}&pitch={pitch}&fov={fov}")

    # Construct the full request URL for the Street View image
    url = (f"{base_url}?size=640x480&location={location[0]},{location[1]}"
            f"&heading={heading}&pitch={pitch}&fov={fov}&key={api_key}")

    # Send a GET request to the Google Street View API
    response = requests.get(url)

    # Check if the response was successful
    if response.status_code == 200:
        # Convert the response content to a NumPy array and decode it into an OpenCV image
        np_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # OpenCV reads images in BGR format
    else:
        # Print an error message if the API request fails
        print("Error fetching image:", response.status_code)
        img = None

    return maps_url, img



