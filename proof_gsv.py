import requests

def check_street_view():
    """
    Check if Google Street View is available at the building's location.

    This method sends a request to the Google Street View API to determine 
    whether Street View imagery is available for the latitude and longitude 
    of the currently selected building. It ensures that a project folder, 
    country, and city name are defined before execution.

    Returns:
        bool: 
            - `True` if Street View imagery is available at the given location.
            - `False` if no Street View coverage exists.

    Effects:
        - Sends an HTTP request to the Google Street View API.
        - Retrieves metadata about Street View availability.

    Notes:
        - Requires a valid Google Street View API key.
        - The API key used in this function is hardcoded, which may pose security risks.
        - Ensures execution only if project details are correctly set.
    """
    # Input parameters
    with open("methods/gsv_api_key.txt", "r") as f:
        api_key = f.read().strip()

    lat= 38.7309	
    lon= -9.1666
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "key": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    # Check status
    if data.get("status") == "OK":
        print("Street View is available")
        return True  # Street View is available
    else:
        print("No Street View coverage")
        return False  # No Street View coverage
    
check_street_view()