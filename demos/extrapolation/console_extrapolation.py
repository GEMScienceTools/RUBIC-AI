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
from geopy.distance import geodesic
from collections import defaultdict
from pathlib import Path

# Function to calculate Geodesic distance (in km)
def geodesic_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).km

# Function to find 3 nearest neighbors using geodesic distance
def find_nearest_neighbors_geodesic(input_row, info_df, n_neighbors=2):
    # Apply geodesic distance for each row in reference dataframe
    distances = info_df.apply(
        lambda row: geodesic_distance(
            input_row['latitude'],
            input_row['longitude'],
            row['latitude'],
            row['longitude']
        ), axis=1)

    nearest_indices = distances.nsmallest(n_neighbors).index
    
    # Extract neighbor data
    neighbor_data = info_df.loc[nearest_indices].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Add distance column to neighbor data
    neighbor_data['distance_km'] = distances.loc[nearest_indices].values
    return neighbor_data

# Function to compute taxonomy probabilities using inverse-distance weighted soft voting
def compute_taxonomy_distribution_full_structure(nearest_neighbors, input_row):
    """
    Computes taxonomy probabilities using weighted soft voting based on geodesic distance.

    Parameters:
    - nearest_neighbors: DataFrame containing the neighbors with 'Taxonomy' and 'distance_km' columns.
    - input_row: The row of the input point.
    - kernel: Kernel type ('inverse' or 'gaussian').
    - bandwidth: Bandwidth for the Gaussian kernel.

    Returns:
    - List of dictionaries, each representing a taxonomy and its probability, along with extra metadata.
    """
    class_weights = defaultdict(float)

    # Assign weights to each neighbor based on the chosen kernel
    for _, row in nearest_neighbors.iterrows():
        dist = row['distance_km']
        label = row['taxonomy']
        
        # Compute weight based on kernel
        weight = 1 / (dist + 1e-6)  # Avoid division by zero
        class_weights[label] += weight

    # Normalize weights to create a probability distribution
    total_weight = sum(class_weights.values())
    probs = {label: weight / total_weight for label, weight in class_weights.items()}

    # Prepare output rows
    distribution_rows = []
    for taxonomy, prob in probs.items():
        # Take a representative row (first one with the taxonomy)
        taxonomy_row = nearest_neighbors[nearest_neighbors['taxonomy'] == taxonomy].iloc[0]
        
        distribution_rows.append({
            'id': input_row['id'],
            'latitude': input_row['latitude'],
            'longitude': input_row['longitude'],
            'country': taxonomy_row['country'],
            'city': taxonomy_row['city'],
            'material': taxonomy_row['material'],
            'llrs': taxonomy_row['llrs'],
            'code_level': taxonomy_row['code_level'],
            'n_stories': taxonomy_row['n_stories'],
            'occupancy': taxonomy_row['occupancy'],
            'block_position': taxonomy_row['block_position'],
            'taxonomy': taxonomy,
            'probability': prob
        })

    return distribution_rows

def extrapolation_existing_reference(data_existing , data_extrapolation, saved_path):
  final_distribution_list_full = []   
  # Iterate over each building with no image
  for idx, input_row in data_extrapolation.iterrows():
      # Find 3 nearest neighbors using geodesic distance
      nearest_neighbors = find_nearest_neighbors_geodesic(input_row, data_existing)
      # Compute taxonomy-based distributions with full structure
      distribution_rows = compute_taxonomy_distribution_full_structure(nearest_neighbors, input_row)
      
      # Append to final result
      final_distribution_list_full.extend(distribution_rows)

  # Convert final list to DataFrame
  final_distribution_df_full = pd.DataFrame(final_distribution_list_full)
  # Export to CSV
  final_distribution_df_full.to_csv(saved_path, index=False)
  
#########################################################
#######===========  General functions ==========#########
#########################################################

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
                    "Image filename or link"]
    
    # Create an empty DataFrame for number of footprint available
    data_ai = pd.DataFrame(np.full((footprint_data.shape[0], len(column_names)), None), columns=column_names)
        
    return data_ai

root_dir = Path(__file__).parent.resolve()
gsv_dir = (root_dir / '..' / '..' / 'methods').resolve()

############ Checks if there is GSV availability ################  
def check_street_view(lat, lon):
    # Input parameters
    with open(gsv_dir / "gsv_api_key.txt", "r") as f:
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
    with open(gsv_dir / "gsv_api_key.txt", "r") as f:
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
    weight_path = dl_dir / "building_detector.pt" # Replace with your YOLO .pt file
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

# dl weight path
root_dir = Path(__file__).parent.resolve()
dl_dir = (root_dir / '..' / '..' / 'dl_weights').resolve()
    
def dl_models():
    global model_material, model_llrs, model_code, model_n_stories, model_occupancy, model_bp, model_rshp, model_rmt
       
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
    model_material.load_state_dict(torch.load(str(dl_dir / "densenet201_material.pt"), map_location=device))
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
    model_llrs.load_state_dict(torch.load(str(dl_dir / "densenet201_llrs.pt"), map_location=device))
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
    model_code.load_state_dict(torch.load(str(dl_dir / "densenet201_code.pt"), map_location=device))
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
    model_n_stories.load_state_dict(torch.load(str(dl_dir / "densenet201_n_stories.pt"), map_location=device))
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
    model_occupancy.load_state_dict(torch.load(str(dl_dir / "densenet201_occupancy.pt"), map_location=device))
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
    model_bp.load_state_dict(torch.load(str(dl_dir / "densenet201_block.pt"), map_location=device))
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
    model_rshp.load_state_dict(torch.load(str(dl_dir / "densenet201_roof_shape.pt"), map_location=device))
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
    model_rmt.load_state_dict(torch.load(str(dl_dir / "densenet201_roof_material.pt"), map_location=device))
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
       
        
#########################################################
#######===========  Input parameters =========###########
#########################################################


"""
method = 0  for existing information of reference
method = 1  for inference first a sample and create the information of reference before the extrapolation
"""
method = 0


if method == 0:
    #########################################################
    #######===========  Input parameters =========###########
    #########################################################
    data_existing = pd.read_csv("neighbor_building_info.csv")
    data_extrapolation = pd.read_csv("building_with_no_image.csv")
    saved_path = "extrapolation_data_example.csv"
    
    #########################################################
    #######===========  Function results =========###########
    #########################################################
    extrapolation_existing_reference(data_existing , data_extrapolation, saved_path)
    
elif method == 1:
    #########################################################
    #######===========  Input parameters =========###########
    #########################################################
    coord_reference = "coordinates_example.csv"
    coord_reference_building_feature_path = "coordinates_reference_results.csv"
    data_extrapolation = pd.read_csv("building_with_no_image.csv")
    saved_path = "extrapolation_data_example_using_ai.csv"
    #########################################################
    #######===========  Function results =========###########
    #########################################################
    data_existing = create_database(coord_reference)
    dl_models()
    inspection_database(data_existing)
    data_existing.to_csv(coord_reference_building_feature_path, index= False)
    extrapolation_existing_reference(data_existing , data_extrapolation, saved_path)
