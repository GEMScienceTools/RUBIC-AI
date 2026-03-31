import re 
import sys
import os
from pathlib import Path
import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import math
import cv2

#########################################################
#######===========  General functions ==========#########
#########################################################

rubicai = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(rubicai))

from methods.taxonomy import check_taxonomy

gsv_api_file = rubicai / 'methods/gsv_api_key.txt'
assert gsv_api_file.exists(), "`gsv_api_key.txt` not found in `methods` directory."

roads_api_file = rubicai / 'methods/roads_api_key.txt'
assert roads_api_file.exists(), "`roads_api_key.txt` not found in `methods` directory."

def create_database(local_building_info):
    global footprint_data
    # Load data
    footprint_data = pd.read_csv(local_building_info)
    
    # Define the column namesfor the inspection database
    column_names = ["id", 
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
                    "image filename or link"]
    
    # Create an empty DataFrame for number of footprint available
    data_ai = pd.DataFrame(np.full((footprint_data.shape[0], len(column_names)), None), columns=column_names)
        
    return data_ai

def get_road_orientation(location):
    """
    Determine the road orientation (azimuth) near a specified location using the Google Roads API.
    """
    with open(roads_api_file, "r") as f:
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

        with open(gsv_api_file, "r") as f:
            api_key = f.read().strip()

        meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        img_url = "https://maps.googleapis.com/maps/api/streetview"

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
            heading = ((road_orientation or 0) + angle + 180) % 360

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
        try:
            heading = (road_orientation + angle + 180) % 360
        except:
            heading = (0 + angle + 180) % 360
    
        # --- Secure API key load ---
        with open(gsv_api_file, "r") as f:
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
    
############ Checks if there is GSV availability ################  
def check_street_view(lat, lon):
    # Input parameters
    with open(gsv_api_file, "r") as f:
        api_key = f.read().strip()
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "key": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    # Check status
    if data.get("status") == "OK":
        return True  # Street View is available
    else:
        return False  # No Street View coverage
    
############# Downnload GSV building images ################   
def fetch_three_step_views(lat, lon):                                                                             
    # Building coordinates
    location = (float(lat), float(lon))
    # API key is required; without it, access to GSV is not possible
    with open(gsv_api_file, "r") as f:
        api_key = f.read().strip()  
    
    if check_street_view(lat, lon) == True:
        # Get image from GSV
        angle = 0
        url_gsv, img_gsv, year = get_street_view_image(location, api_key, angle, 5, 120)
    else:
        print("Street View not available")
        url_gsv = "Street View not available"
        img_gsv = []
        
    return img_gsv, url_gsv  
    
############ Building detector model ################
def object_detector_building(lat, lon):
    global url_gsv
    # Class mapping (update this with your actual mappings)
    weight_path = dl_dir / "building_detector.pt" # Replace with your YOLO .pt file
    model = YOLO(weight_path)
    TARGET_CLASS = 'building-xzyh'
    CONF_THRESHOLD = 0.5
    # Set device GPU or CPU
    device= "cuda" if torch.cuda.is_available() else "cpu"

    img_gsv, url_gsv  = fetch_three_step_views(lat, lon)
    try:
        # Run inference
        results = model.predict(img_gsv, device=device)[0]
    
        h, w, _ = img_gsv.shape
    
        # Get class names
        class_names = model.names
    
        best_box = None
        best_conf = 0
    
        # Loop through detected boxes
        if results.boxes is not None:
            for box in results.boxes:
    
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                conf = float(box.conf[0])
                if label == TARGET_CLASS and conf > CONF_THRESHOLD:
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box.xyxy[0].cpu().numpy().astype(int)
    
        if best_box is None:
            print("❌ No Building detected in image.")
            print()
            return
    
        x1, y1, x2, y2 = best_box
    
        # ✅ Ensure values inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
    
        # ✅ Crop image
        cropped_image = img_gsv[y1:y2, x1:x2]
        return cropped_image
    except:
        cropped_image = []
    
############ Get city name using coordinates ################
def get_city_name(lat, lon):           
    try:    
        geolocator = Nominatim(user_agent="city_name_locator")
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en", timeout=3)
        if location and 'address' in location.raw:
            address = location.raw['address']
            city = (address.get("city") or address.get("town") or address.get("village")
                or address.get("municipality") or address.get("county") or address.get("state_district")
                or "Unknown")
            country = address.get('country', 'Unknown')
            return city , country
    except:
        city = "Unknown"
        country = "Unknown"
        return city , country
    
    
#########################################################
#######==========  DL models definition ========#########
#########################################################

# Define the device (CPU-only if no GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformation (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
root_dir = Path(__file__).parent.resolve()
dl_dir = (root_dir / '..' / '..' / 'dl_weights').resolve()

# ---------------- Metadata encoder (same as training) ----------------
class ManualOneHotMeta:
    def __init__(self, cols):
        self.cols = cols
        self.vocab = {}
        self.index = {}
        self.offsets = {}
        self.meta_dim = 0

    def fit(self, df_train):
        offset = 0
        for col in self.cols:
            cats = sorted(df_train[col].unique().tolist())
            if "UNKNOWN" not in cats:
                cats.append("UNKNOWN")
            self.vocab[col] = cats
            self.index[col] = {v: i for i, v in enumerate(cats)}
            self.offsets[col] = offset
            offset += len(cats)
        self.meta_dim = offset

    def encode(self, row):
        vec = np.zeros((self.meta_dim,), dtype=np.float32)
        for col in self.cols:
            v = row.get(col, "UNKNOWN")
            if v is None:
                v = "UNKNOWN"
            v = str(v).strip()
            i = self.index[col].get(v, self.index[col]["UNKNOWN"])
            vec[self.offsets[col] + i] = 1.0
        return vec

# ---------------- Model (same as training) ----------------
class DenseNet201WithMetadata(nn.Module):
    def __init__(self, num_classes, meta_dim, fusion_mode="concat", meta_hidden=128, dropout=0.5):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        img_feat_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if fusion_mode == "concat":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(img_feat_dim + meta_hidden, num_classes),
            )
        else:
            raise ValueError("This inference example expects fusion_mode='concat'")

    def forward(self, x, meta):
        img_feat = self.backbone(x)
        meta_feat = self.meta_mlp(meta)
        feat = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(feat)


print("Uploading DL models")
# Load the model_material architecture

# ---------------- CONFIG ----------------
META_COLS = ("country", "city")
encoder = ManualOneHotMeta(META_COLS)

# ---------------- Updated prediction function ----------------
def predict_llrs_img(image_path, country="UNKNOWN", city="UNKNOWN"):
    # Build encoder from training vocab
    WEIGHTS_PATH = rubicai / 'Meta_data/New_weights_metadata/densenet201_llrs_with_metadata.pt'
    TRAIN_META_CSV = rubicai / 'Meta_data/train_data.csv'
    CLASS_NAMES = ['LDUAL', 'LFINF', 'LFM', 'LN', 'LWAL', 'TW']

    df_train = pd.read_csv(TRAIN_META_CSV)
    df_train["country"] = df_train["country"].fillna("UNKNOWN").astype(str).str.strip()
    df_train["city"]    = df_train["city"].fillna("UNKNOWN").astype(str).str.strip()
    encoder.fit(df_train)

    model_llrs = DenseNet201WithMetadata(
        num_classes=len(CLASS_NAMES),
        meta_dim=encoder.meta_dim
    ).to(device)
    model_llrs.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model_llrs.eval()
    
    # --- Image ---
    image = Image.fromarray(image_path.astype('uint8'))
    x = transform(image).unsqueeze(0).to(device)           # (1, 3, 256, 256)

    # --- Metadata ---
    row = {"country": str(country).strip(), "city": str(city).strip()}
    meta_vec = encoder.encode(row)
    m = torch.tensor(meta_vec).unsqueeze(0).to(device)     # (1, meta_dim)

    # --- Inference ---
    with torch.no_grad():
        logits = model_llrs(x, m)
        prediction = torch.argmax(logits, dim=1).item()

    return CLASS_NAMES[prediction]
  
def tax_check(tax_value):
    # 1) Create a small DataFrame with taxonomy strings
    df = pd.DataFrame({"TAXONOMY": [tax_value]})
    
    try:
        tax = check_taxonomy(df, taxo_col="TAXONOMY")
    except ValueError as e:
        print("There are invalid taxonomies ❌")
        # Convert the error to string
        err_str = str(e)
        
        # Extract the canonical value from the string using regex
        match = re.search(r"'canonical': '([^']+)'", err_str)
        if match:
            tax_canonical = match.group(1)
            print("Canonical taxonomy:", tax_canonical)
        else:
            tax_canonical = None
            print("No canonical value found.")
            
############ Obtain value of the form of each building image ################       
def inspection_database (data_ai):
    for i in range (data_ai.shape[0]):
        data_ai.iloc[i, 0] = footprint_data.loc[i, "id"]                                     # ID
        data_ai.iloc[i, 1] = footprint_data.loc[i , "latitude"]                               # Latitude
        data_ai.iloc[i, 2] = footprint_data.loc[i , "longitude"]                              # Latitude
        
        try:
            image_file = object_detector_building(float(footprint_data.loc[i,"latitude"]) , 
                                                    float(footprint_data.loc[i,"longitude"]))

            if image_file is None:
                pass
            else:
                city, country = get_city_name(float(footprint_data.loc[i,"latitude"]) , float(footprint_data.loc[i,"longitude"]))
                data_ai.iloc[i, 3], data_ai.iloc[i, 4] = country , city
                data_ai.iloc[i, 6] = predict_llrs_img (image_file)                                # LLRS 
        except:
            print(" Error in building ID: " + str(footprint_data.loc[i, "id"]))
            pass
        
        print("Inspection: " + str(i+1)+"/"+str(data_ai.shape[0]) +" -------------------------------------")
       
        
#########################################################
#######===========  Input parameters =========###########
#########################################################

# Input data
local_building_info = rubicai / "ECEE/ECEE_building_info.csv"
# Output path
saved_path = rubicai / "ECEE/console_mode_metadata/ECEE_AI_material.csv"

#########################################################
#######===========  Function results =========###########
#########################################################

data_ai = create_database(local_building_info)
inspection_database(data_ai)
os.makedirs(os.path.dirname(saved_path), exist_ok=True)
data_ai.to_csv(saved_path, index= False)

