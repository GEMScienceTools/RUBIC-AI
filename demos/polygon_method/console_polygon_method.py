import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from get_building_orientation import get_street_view_image
# Libries for shape and geopackage creation
from shapely.geometry import Polygon
from shapely.geometry import box
import osmnx as ox
import geopandas as gpd
import os
#########################################################
#######===========  General functions ==========#########
#########################################################
def polygon_coordinates(file_path, polygon_name):
    output_gpkg = polygon_name + "_boundary.gpkg"
    df = pd.read_csv(file_path)    
    coordinates = list(zip(df["longitude"], df["latitude"]))
    polygon = Polygon(coordinates)
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:4326")
    gdf.to_file(output_gpkg, driver="GPKG", layer="polygon_layer")

def download_building_footprints_polygon(polygon_name):
    output_file = polygon_name + "_buildings_footprint.gpkg"
    boundary_path = polygon_name + "_boundary.gpkg"
    # Conditionional checks for an existing boundary file, and if it exists, avoids creating a duplicate
    if os.path.exists(output_file):
        buildings = gpd.read_file(output_file)
    else:
        # Check if a boundary file exists to download the building footprints within it
        if os.path.exists(boundary_path):               
            # Define input and output file paths
            geopackage_path = boundary_path
            # Load the single layer from the GeoPackage
            gdf = gpd.read_file(geopackage_path)
            # Ensure the CRS is EPSG:4326
            if gdf.crs.to_string() != "EPSG:4326":
                print("Reprojecting to EPSG:4326...")
                gdf = gdf.to_crs("EPSG:4326")
            # Download building footprints from OSM
            polygon = gdf.union_all()
            # For latest osmnx versions
            try:
                buildings = ox.geometries_from_polygon(polygon, tags={"building": True})
            except AttributeError:
                buildings = ox.features_from_polygon(polygon, tags={"building": True})
            # Save the downloaded footprints to a new GeoPackage
            if buildings.empty:
                return None
            # Filter only Polygon and MultiPolygon geometries
            buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])]
            if buildings.empty:
                return None   
            # Drop the AREA column if it exists
            if "AREA" in buildings.columns:
                buildings = buildings.drop(columns=["AREA"])
            # Save to GeoPackage
            buildings.to_file(output_file, driver="GPKG")
    
    return len(buildings)
############ Random subset buildings ################  
def extract_random_subset(polygon_name , sample_size):
    # Load buildng footprints
    footprint = polygon_name + "_buildings_footprint.gpkg"
    # Create output file for building footprints
    output_file= polygon_name + "_subset_footprints.gpkg"
    # Ensure sample size is not greater than the number of points in the dataset
    seed=10
    # Check if a subset file exists
    if os.path.exists(output_file):
        pass
    else:
        sample_size = sample_size
        if 0 < sample_size < 1:
            # Load the input point layer
            gdf = gpd.read_file(footprint)
            sample_size = int(gdf.shape[0]*sample_size)
            # Extract a random sample
            subset = gdf.sample(n=sample_size, random_state=seed)
            subset.to_file(output_file, driver="GPKG", layer="random_subset")

############ Create a point layer and extract the coordinates of a subset of buildings ################ 
def create_centroid_layer(polygon_name):
    # Load selected subset building
    subset_file = polygon_name + "_subset_footprints.gpkg"
    # Create output file for building footprints
    output_file = polygon_name + "_subset_centroids.gpkg"

    # Check if a centroid file exists
    if os.path.exists(output_file):
        return
    
    # Load the building footprints
    buildings = gpd.read_file(subset_file)

    # Reproject to a suitable projected CRS for centroid calculation
    # If data is in EPSG:4326, pick an appropriate UTM zone
    if buildings.crs.to_string() == "EPSG:4326":
        # Example: for Colombia ~ EPSG:32618 (WGS84 / UTM zone 18N)
        # You may need to adjust depending on your polygon location
        projected = buildings.to_crs(epsg=32618)
    else:
        projected = buildings

    # Calculate centroids in projected CRS
    centroids_geom = projected.centroid

    # Reproject centroids back to geographic CRS (lat/lon)
    centroids_geom = centroids_geom.to_crs(epsg=4326)

    # Build GeoDataFrame for centroids
    centroids = gpd.GeoDataFrame(
        buildings.drop(columns="geometry"), 
        geometry=centroids_geom, 
        crs="EPSG:4326"
    )

    # Add latitude and longitude columns
    centroids["latitude"] = centroids.geometry.y
    centroids["longitude"] = centroids.geometry.x

    # Save to file
    centroids.to_file(output_file, driver="GPKG", layer="centroids")
    
    # Filter columns
    filtered_gdf = centroids[['id', 'latitude', 'longitude']]
    # Export to CSV
    database_file = polygon_name + "_building_info.csv"
    filtered_gdf.to_csv(database_file, index=False)             

def create_database(polygon_name):
    global footprint_data
    # Load data
    footprint_data = pd.read_csv(polygon_name + "_building_info.csv")
    
    # Define the column namesfor the inspection database
    column_names = ["ID", 
                    "Latitude", 
                    "Longitude",
                    "Country",
                    "City",
                    "LLRS Material",
                    "LLRS",
                    "Code Level",
                    "Number of Stories",
                    "Occupancy",
                    "Block Position",
                    "Roof shape",
                    "Roof material",
                    "Taxonomy",
                    "Image filename or link"]
    
    # Create an empty DataFrame for number of footprint available
    data_ai = pd.DataFrame(np.full((footprint_data.shape[0], len(column_names)), None), columns=column_names)
        
    return data_ai

############ Checks if there is GSV availability ################  
def check_street_view(lat, lon):
    # Input parameters
    with open("gsv_api_key.txt", "r") as f:
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
    with open("gsv_api_key.txt", "r") as f:
        api_key = f.read().strip()  
    
    if check_street_view(lat, lon) == True:
        # Get image from GSV
        angle = 0
        url_gsv = get_street_view_image(location, api_key, angle)[0]
        img_gsv = get_street_view_image(location, api_key, angle)[1]
    else:
        print("Street View not available")
        url_gsv = "Street View not available"
        img_gsv = []
        
    print
    return img_gsv, url_gsv
        
    
############ Building detector model ################
def object_detector_building(lat, lon):
    global url_gsv
    # Class mapping (update this with your actual mappings)
    class_map = {0: "building-xzyh"}  # Replace with the correct mapping
    weight_path = "dl_weights/building_detector.pt" # Replace with your YOLO .pt file
    # Load the YOLO model
    model = YOLO(weight_path)
    # Set device GPU or CPU
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
       
    img_gsv, url_gsv  = fetch_three_step_views(lat, lon)
    
    # Run inference
    try:
        results = model.predict(img_gsv)
    
        highest_conf = 0
        highest_conf_box = None
    
        # Process the results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs
    
            for box, conf, cls in zip(boxes, confs, classes):
                cls = int(cls)
                # Getting the building image with higher confidence as selected bounding box
                if class_map.get(cls) == "building-xzyh" and conf > highest_conf:
                    highest_conf = conf
                    highest_conf_box = box
                
        # Extracting selecting bounding box coordinates witin the image
        if highest_conf_box is not None:
            x1, y1, x2, y2 = map(int, highest_conf_box)
            # Crop the area within the selected bounding box
            cropped_image = img_gsv[y1:y2, x1:x2] 
        else: 
            pass
        
        return cropped_image
    except:
        pass
    
############ Get city name using coordinates ################
def get_city_name(lat, lon):           
        geolocator = Nominatim(user_agent="city_name_locator")
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        
        if location and 'address' in location.raw:
            address = location.raw['address']
            city = address.get('city', address.get('town', address.get('village', 'Unknown')))
            country = address.get('country', 'Unknown')
            return city , country



#########################################################
#######==========  DL models definition ========#########
#########################################################


################### Material model #########################
# Define the device (CPU-only if no GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformation (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
# Load the model_material architecture
model_material = models.densenet201(weights=None)  # Initialize model_material without pre-trained weights
num_features = model_material.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_material.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 8),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_material.load_state_dict(torch.load("dl_weights/densenet201_material.pt", map_location=device))
model_material.to(device)
model_material.eval()


################### LLRS model #########################
# Define the device (CPU-only if no GPU is available)

# Load the model architecture
model_llrs = models.densenet201(weights=None)  # Initialize model without pre-trained weights
num_features = model_llrs.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_llrs.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 6),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_llrs.load_state_dict(torch.load("dl_weights/densenet201_llrs.pt", map_location=device))
model_llrs.to(device)
model_llrs.eval()


################### CODE model #########################
# Define the device (CPU-only if no GPU is available)

# Load the model architecture
model_code = models.densenet201(weights=None)  # Initialize model without pre-trained weights
num_features = model_code.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_code.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 4),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_code.load_state_dict(torch.load("dl_weights/densenet201_code.pt", map_location=device))
model_code.to(device)
model_code.eval()


################### N STORIES model #########################
# Define the device (CPU-only if no GPU is available)

# Load the model architecture
model_n_stories = models.densenet201(weights=None)  # Initialize model without pre-trained weights
num_features = model_n_stories.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_n_stories.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 9),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_n_stories.load_state_dict(torch.load("dl_weights/densenet201_n_stories.pt", map_location=device))
model_n_stories.to(device)
model_n_stories.eval()


################### OCCUPANCY model #########################
# Define the device (CPU-only if no GPU is available)

# Load the model architecture
model_occupancy = models.densenet201(weights=None)  # Initialize model without pre-trained weights
num_features = model_occupancy.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_occupancy.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 7),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_occupancy.load_state_dict(torch.load("dl_weights/densenet201_occupancy.pt", map_location=device))
model_occupancy.to(device)
model_occupancy.eval()


################### BLOCK POSTION model #########################
# Define the device (CPU-only if no GPU is available)

# Load the model architecture
model_bp = models.densenet201(weights=None)  # Initialize model without pre-trained weights
num_features = model_bp.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_bp.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 3),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_bp.load_state_dict(torch.load("dl_weights/densenet201_block.pt", map_location=device))
model_bp.to(device)
model_bp.eval()


################### Roof Shape model #########################
# Define the device (CPU-only if no GPU is available)

# Load the model architecture
model_rshp = models.densenet201(weights=None)  # Initialize model without pre-trained weights
num_features = model_rshp.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_rshp.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 3),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_rshp.load_state_dict(torch.load("dl_weights/densenet201_roof_shape.pt", map_location=device))
model_rshp.to(device)
model_rshp.eval()
    

################### Roof Material model #########################
# Define the device (CPU-only if no GPU is available)

# Load the model architecture
model_rmt = models.densenet201(weights=None)  # Initialize model without pre-trained weights
num_features = model_rmt.classifier.in_features

# Use the correct number of output classes (9 as indicated in the error)
model_rmt.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(num_features, 3),  # Match the number of classes
    torch.nn.LogSoftmax(dim=1)
)

# Load the trained weights
model_rmt.load_state_dict(torch.load("dl_weights/densenet201_roof_material.pt", map_location=device))
model_rmt.to(device)
model_rmt.eval()


#########################################################
#######===========  Models predicition =========#########
#########################################################


############ Material prediction ################
def predict_material_img (image_path):
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model_material(image)
        prediction = torch.argmax(output, dim=1).item()

    # LLRS building image sets prediction
    material_classes = ['ADO', 'CR', 'MCF', 'MR', 'MUR', 'MX', 'S', 'W']
    material_id = material_classes[prediction]

    return material_id

############ LLRS prediction ################
def predict_llrs_img (image_path):
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model_llrs(image)
        prediction = torch.argmax(output, dim=1).item()
        
    # LLRS building image sets prediction
    llrs_classes = ['LDUAL', 'LFINF', 'LFM', 'LWAL', 'TW', 'W']
    llrs_id = llrs_classes[prediction]
    
    return llrs_id

############ Code level prediction ################
def predict_code_img (image_path):
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model_code(image)
        prediction = torch.argmax(output, dim=1).item()
        
    # code_level building image sets prediction
    code_level_classes = ['CDH', 'CDM', 'CDM', 'CDN']
    code_level_id = code_level_classes[prediction]
    return code_level_id

############ Number of Stories prediction ################
def predict_n_stories_img (image_path):
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model_n_stories(image)
        prediction = torch.argmax(output, dim=1).item()
        
    class_names = ['10-12', '13+', '1', '2', '3', '4', '5', '6-7', '8-9']
    n_stories_id = class_names[prediction]
    return n_stories_id

############ Occupancy prediction ################
def predict_occupancy_img (image_path): 
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model_occupancy(image)
        prediction = torch.argmax(output, dim=1).item()
        
    occupancy_class = ['COM', 'EDU', 'GOV', 'IND', 'MIX', 'OCO', 'RES']
    occupancy_id = occupancy_class[prediction]
    return occupancy_id

############ Block Position prediction ################
def predict_block_position_img (image_path):
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model_bp(image)
        prediction = torch.argmax(output, dim=1).item()
        
    block_position_classes = ['BP1', 'BP2', 'BPD']
    block_position_id = block_position_classes[prediction]
    return block_position_id

############ Roof Shape prediction ################
def predict_roof_shape_img (image_path):
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model_rshp(image)
        prediction = torch.argmax(output, dim=1).item()
        
    roof_shape_classes = ['RSH1', 'RSH2', 'RSH3']
    roof_shape_id = roof_shape_classes[prediction]
    return roof_shape_id


############ Roof Material prediction ################
def predict_roof_material_img (image_path):
    # Function to predict the class of an image
    image = Image.fromarray(image_path.astype('uint8'))
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model_rmt(image)
        prediction = torch.argmax(output, dim=1).item()
        
    roof_material_classes = ['RMN', 'RMT1', 'RMT6']
    roof_material_id = roof_material_classes[prediction]
    return roof_material_id

def get_polygon_and_sample(polygon_name, sample_size, file_path):
    polygon_coordinates(file_path, polygon_name)
    download_building_footprints_polygon(polygon_name)
    extract_random_subset(polygon_name , sample_size)
    create_centroid_layer(polygon_name)
    
############ Obtain value of the form of each building image ################       
def inspection_database (data_ai):
    for i in range (data_ai.shape[0]):
        data_ai.iloc[i, 0] = footprint_data.iloc[i,0]                                                 # ID
        data_ai.iloc[i, 1] = footprint_data.loc[i , "latitude"]                                                 # Latitude
        data_ai.iloc[i, 2] = footprint_data.loc[i , "longitude"] 
        
        image_file = object_detector_building(float(footprint_data.loc[i,"latitude"]) , float(footprint_data.loc[i,"longitude"]))

        if image_file is None:
            pass
        else:
            city, country = get_city_name(float(footprint_data.loc[i,"latitude"]) , float(footprint_data.loc[i,"longitude"]))
                                             
            data_ai.iloc[i, 3], data_ai.iloc[i, 4] = city, country
            data_ai.iloc[i, 5] = predict_material_img (image_file)                            # LLRS Material
            data_ai.iloc[i, 6] = predict_llrs_img (image_file)                                # LLRS 
            data_ai.iloc[i, 7] = predict_code_img (image_file)                                # Code Level 
            data_ai.iloc[i, 8] = predict_n_stories_img (image_file)                           # Number of Stories 
            data_ai.iloc[i, 9] = predict_occupancy_img (image_file)                           # Occupancy
            data_ai.iloc[i, 10] = predict_block_position_img (image_file)                     # Block Position
            data_ai.iloc[i, 11] = predict_roof_shape_img (image_file)                         # Roof shape
            data_ai.iloc[i, 12] = predict_roof_material_img (image_file)                      # Roof material
            data_ai.iloc[i, 13] = (data_ai.iloc[i, 5]+"/"+
                                    data_ai.iloc[i, 6]+"+"+
                                    data_ai.iloc[i, 7]+"/H:"+
                                    data_ai.iloc[i, 8]+"/"+
                                    data_ai.iloc[i, 9]+"/"+
                                    data_ai.iloc[i, 10]+"/"+
                                    data_ai.iloc[i, 11]+"+"+
                                    data_ai.iloc[i, 12])                       # Taxonomy
            
            data_ai.iloc[i, 14] = url_gsv
        
        print("Inspection: " + str(i+1)+"/"+str(data_ai.shape[0]) +" -------------------------------------")
       
        
# #########################################################
#######===========  Input parameters =========###########
#########################################################

saved_path = "example_prediction_result.csv"
file_path = r"H:\My Drive\Sura_2025_AI\Console_polygon_method\polygon_method_example.csv"
polygon_name = "proof_polygon"
sample_size = 0.05
#########################################################
#######===========  Function results =========###########
#########################################################

get_polygon_and_sample(polygon_name, sample_size, file_path)
data_ai = create_database(polygon_name)
inspection_database(data_ai)
data_ai.to_csv(saved_path, index= False)

