import re
import sys
import os
from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from geopy.geocoders import Nominatim
from PIL import Image
from torchvision import models
from ultralytics import YOLO

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

with open(gsv_api_file, "r") as f:
    GSV_API_KEY = f.read().strip()

with open(roads_api_file, "r") as f:
    ROADS_API_KEY = f.read().strip()

GEOCODER = Nominatim(user_agent="city_name_locator")


def create_database(local_building_info):
    global footprint_data
    footprint_data = pd.read_csv(local_building_info)

    column_names = [
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

    data_ai = pd.DataFrame(
        np.full((footprint_data.shape[0], len(column_names)), None),
        columns=column_names,
    )
    return data_ai


def get_road_orientation(location):
    """Determine the road orientation (azimuth) near a specified location using the Google Roads API."""
    base_url = "https://roads.googleapis.com/v1/nearestRoads"
    params = {"points": f"{location[0]},{location[1]}", "key": ROADS_API_KEY}

    response = requests.get(base_url, params=params, timeout=20)
    if response.status_code == 200:
        data = response.json()
        if "snappedPoints" in data and data["snappedPoints"]:
            snapped_point = data["snappedPoints"][0]
            road_lat = snapped_point["location"]["latitude"]
            road_lng = snapped_point["location"]["longitude"]
            return compute_azimuth(location, (road_lat, road_lng))

        print("No road found near the location.")
        return None

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
    """Fetch a Google Street View image (outdoor-only) and generate its corresponding Maps URL."""
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    meta_params = {
        "location": f"{location[0]},{location[1]}",
        "source": "outdoor",
        "key": api_key,
    }
    meta_response = requests.get(meta_url, params=meta_params, timeout=20)
    meta_data = meta_response.json()

    if meta_data.get("status") != "OK":
        print(f"No outdoor panorama available at {location}. Status: {meta_data.get('status')}")

        found_close = False
        max_radius = 20
        step = 5
        show_debug = True

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
            r = requests.get(meta_url, params=meta_params, timeout=20)
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

        try:
            if pano_lon > location[1]:
                angle = 180
                side = "right"
            else:
                angle = 0
                side = "left"

            if show_debug:
                print(f"Pano is located to the {side} of the building → angle={angle}°")

            road_orientation = get_road_orientation(location)
            heading = ((road_orientation or 0) + angle + 180) % 360

            if show_debug:
                print(f"Road orientation: {road_orientation}")
                print(f"Final heading: {heading}")

            params = {
                "size": "640x480",
                "pano": pano_id,
                "heading": heading,
                "pitch": pitch,
                "fov": fov,
                "source": "outdoor",
                "key": api_key,
            }

            resp = requests.get(img_url, params=params, timeout=30)
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

    year = None
    if "date" in meta_data:
        year = meta_data["date"].split("-")[0]

    road_orientation = get_road_orientation(location)
    try:
        heading = (road_orientation + angle + 180) % 360
    except Exception:
        heading = (0 + angle + 180) % 360

    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x480",
        "location": f"{location[0]},{location[1]}",
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "scale": 2,
        "source": "outdoor",
        "key": api_key,
    }

    maps_url = (
        f"https://www.google.com/maps/@?api=1&map_action=pano"
        f"&viewpoint={location[0]},{location[1]}&heading={heading}&pitch={pitch}&fov={fov}"
    )

    response = requests.get(base_url, params=params, timeout=30)
    if response.status_code == 200:
        np_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    else:
        print("Error fetching image:", response.status_code)
        img = None

    return maps_url, img, year


def fetch_three_step_views(lat, lon):
    location = (float(lat), float(lon))
    angle = 0
    url_gsv, img_gsv, year = get_street_view_image(location, GSV_API_KEY, angle, 5, 120)

    if img_gsv is None:
        print("Street View not available")
        url_gsv = "Street View not available"

    return img_gsv, url_gsv


#########################################################
#######==========  DL models definition ========#########
#########################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

root_dir = Path(__file__).parent.resolve()
dl_dir = (root_dir / '..' / '..' / 'dl_weights').resolve()


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


META_COLS = ("country", "city")

material_classes = ['CR', 'HYB(MCF;MUR)', 'INF', 'MCF', 'MR', 'MUR', 'S', 'W']
llrs_classes = ['LDUAL', 'LFINF', 'LFM', 'LN', 'LWAL', 'LWAL']
code_level_classes = ['CDH', 'CDL', 'CDM', 'CDN']
ns_classes = ['10-12', '13+', '1', '2', '3', '4', '5', '6-7', '8-9']
occupancy_classes = ['COM', 'IND', 'MIX(RES;COM)', 'RES']
block_position_classes = ['BP1', 'BP2', 'BP3', 'BPD']
roof_shape_classes = ['RSH1', 'RSH2', 'RSH3', 'RSH5', 'RSH7']
roof_material_classes = ['RMN', 'RMT1', 'RMT6']


def load_yolo_detector():
    weight_path = dl_dir / "building_detector.pt"
    return YOLO(weight_path)


def build_meta_encoder_16f():
    train_meta_csv = rubicai / 'Meta_data/train_data_16f.csv'
    df_train = pd.read_csv(train_meta_csv)
    df_train["country"] = df_train["country"].fillna("UNKNOWN").astype(str).str.strip()
    df_train["city"] = df_train["city"].fillna("UNKNOWN").astype(str).str.strip()

    encoder_16f = ManualOneHotMeta(META_COLS)
    encoder_16f.fit(df_train)
    return encoder_16f

def build_meta_encoder_14f():
    train_meta_csv = rubicai / 'Meta_data/train_data_14f.csv'
    df_train = pd.read_csv(train_meta_csv)
    df_train["country"] = df_train["country"].fillna("UNKNOWN").astype(str).str.strip()
    df_train["city"] = df_train["city"].fillna("UNKNOWN").astype(str).str.strip()

    encoder_14f = ManualOneHotMeta(META_COLS)
    encoder_14f.fit(df_train)
    return encoder_14f

def build_meta_encoder_13f():
    train_meta_csv = rubicai / 'Meta_data/train_data_13f.csv'
    df_train = pd.read_csv(train_meta_csv)
    df_train["country"] = df_train["country"].fillna("UNKNOWN").astype(str).str.strip()
    df_train["city"] = df_train["city"].fillna("UNKNOWN").astype(str).str.strip()

    encoder_13f = ManualOneHotMeta(META_COLS)
    encoder_13f.fit(df_train)
    return encoder_13f

def load_metadata_model(weights_path, class_names, encoder):
    model = DenseNet201WithMetadata(
        num_classes=len(class_names),
        meta_dim=encoder.meta_dim,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


print("Loading DL models only once...")
YOLO_MODEL = load_yolo_detector()
META_ENCODER_16f = build_meta_encoder_16f()
META_ENCODER_14f = build_meta_encoder_14f()
META_ENCODER_13f = build_meta_encoder_13f()

MODEL_SPECS_16f = {
    "material": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_material_with_metadata.pt',
        "classes": material_classes,
        "column": "material",
    },
    "llrs": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_llrs_with_metadata.pt',
        "classes": llrs_classes,
        "column": "llrs",
    },
    "code_level": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_code_with_metadata.pt',
        "classes": code_level_classes,
        "column": "code_level",
    },
    "n_stories": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_ns_with_metadata.pt',
        "classes": ns_classes,
        "column": "n_stories",
    },
}

MODEL_SPECS_14f = {
    "occupancy": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_occ_with_metadata.pt',
        "classes": occupancy_classes,
        "column": "occupancy",
    },
    "block_position": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_bp_with_metadata.pt',
        "classes": block_position_classes,
        "column": "block_position",
    },
}

MODEL_SPECS_13f = {
    "roof_shape": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_r_shape_with_metadata.pt',
        "classes": roof_shape_classes,
        "column": "roof_shape",
    },
    "roof_material": {
        "weights": rubicai / 'Meta_data/New_weights_metadata/densenet201_r_mat_with_metadata.pt',
        "classes": roof_material_classes,
        "column": "roof_material",
    },
}


LOADED_MODELS_16f = {
    model_name: load_metadata_model(spec["weights"], spec["classes"], META_ENCODER_16f)
    for model_name, spec in MODEL_SPECS_16f.items() }

LOADED_MODELS_14f = {
    model_name: load_metadata_model(spec["weights"], spec["classes"], META_ENCODER_14f)
    for model_name, spec in MODEL_SPECS_14f.items()}

LOADED_MODELS_13f = {
    model_name: load_metadata_model(spec["weights"], spec["classes"], META_ENCODER_13f)
    for model_name, spec in MODEL_SPECS_13f.items()}


def object_detector_building(lat, lon):
    global url_gsv
    target_class = 'building-xzyh'
    conf_threshold = 0.5
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    img_gsv, url_gsv = fetch_three_step_views(lat, lon)
    if img_gsv is None:
        return None

    try:
        results = YOLO_MODEL.predict(img_gsv, device=device_name, verbose=False)[0]
        h, w, _ = img_gsv.shape
        class_names = YOLO_MODEL.names

        best_box = None
        best_conf = 0.0
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                conf = float(box.conf[0])
                if label == target_class and conf > conf_threshold and conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy().astype(int)

        if best_box is None:
            print("❌ No Building detected in image.")
            return None

        x1, y1, x2, y2 = best_box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return img_gsv[y1:y2, x1:x2]
    except Exception as e:
        print(f"Object detector error: {e}")
        return None


# def get_city_name(lat, lon):
#     try:
#         location = GEOCODER.reverse((lat, lon), exactly_one=True, language="en", timeout=3)
#         if location and 'address' in location.raw:
#             address = location.raw['address']
#             city = (
#                 address.get("city")
#                 or address.get("town")
#                 or address.get("village")
#                 or address.get("municipality")
#                 or address.get("county")
#                 or address.get("state_district")
#                 or "Unknown"
#             )
#             country = address.get('country', 'Unknown')
#             return city, country
#     except Exception:
#         pass

#     return "Unknown", "Unknown"


def predict_with_metadata_model(image_array, model, class_names, encoder, country="UNKNOWN", city="UNKNOWN"):
    image = Image.fromarray(image_array.astype('uint8')).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    row = {"country": str(country).strip(), "city": str(city).strip()}
    meta_vec = encoder.encode(row)
    m = torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x, m)
        prediction = torch.argmax(logits, dim=1).item()

    return class_names[prediction]


def tax_check(tax_value):
    df = pd.DataFrame({"TAXONOMY": [tax_value]})
    try:
        check_taxonomy(df, taxo_col="TAXONOMY")
    except ValueError as e:
        print("There are invalid taxonomies ❌")
        err_str = str(e)
        match = re.search(r"'canonical': '([^']+)'", err_str)
        if match:
            tax_canonical = match.group(1)
            print("Canonical taxonomy:", tax_canonical)
        else:
            print("No canonical value found.")


def inspection_database(data_ai):
    for i in range(data_ai.shape[0]):
        latitude = float(footprint_data.loc[i, "latitude"])
        longitude = float(footprint_data.loc[i, "longitude"])

        data_ai.iloc[i, 0] = footprint_data.loc[i, "id"]
        data_ai.iloc[i, 1] = latitude
        data_ai.iloc[i, 2] = longitude

        try:
            image_file = object_detector_building(latitude, longitude)

            if image_file is not None:
                # city, country = get_city_name(latitude, longitude)
                city = "Lisbon"
                country = "Portugal"
                data_ai.iloc[i, 3] = country
                data_ai.iloc[i, 4] = city

                for model_name, spec in MODEL_SPECS_16f.items():
                    prediction = predict_with_metadata_model(
                        image_array=image_file,
                        model=LOADED_MODELS_16f[model_name],
                        class_names=spec["classes"],
                        encoder=META_ENCODER_16f,
                        # country=country,
                        # city=city,
                        country="Portugal",
                        city="Lisbon",
                    )
                    data_ai.at[i, spec["column"]] = prediction
                    
                for model_name, spec in MODEL_SPECS_14f.items():
                    prediction = predict_with_metadata_model(
                        image_array=image_file,
                        model=LOADED_MODELS_14f[model_name],
                        class_names=spec["classes"],
                        encoder=META_ENCODER_14f,
                        # country=country,
                        # city=city,
                        country="Portugal",
                        city="Lisbon",
                    )
                    data_ai.at[i, spec["column"]] = prediction
                    
                for model_name, spec in MODEL_SPECS_13f.items():
                    prediction = predict_with_metadata_model(
                        image_array=image_file,
                        model=LOADED_MODELS_13f[model_name],
                        class_names=spec["classes"],
                        encoder=META_ENCODER_13f,
                        # country=country,
                        # city=city,
                        country="Portugal",
                        city="Lisbon",
                    )
                    data_ai.at[i, spec["column"]] = prediction

        except Exception as e:
            print("Error in building ID:", footprint_data.loc[i, "id"], "->", e)

        print(f"Inspection: {i + 1}/{data_ai.shape[0]} -------------------------------------")


#########################################################
#######===========  Input parameters =========###########
#########################################################

local_building_info = rubicai / "ECEE/ECEE_building_info.csv"
saved_path = rubicai / "ECEE/console_mode_metadata/ECEE_AI_multi_metadata.csv"

#########################################################
#######===========  Function results =========###########
#########################################################

if __name__ == "__main__":
    data_ai = create_database(local_building_info)
    inspection_database(data_ai)
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    data_ai.to_csv(saved_path, index=False)
