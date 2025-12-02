import re 
import sys
from pathlib import Path
import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

#########################################################
#######===========  General functions ==========#########
#########################################################

rubicai = Path(__file__).parent.parent.parent.resolve()
print(f"RubicAI path: {rubicai}")
sys.path.append(str(rubicai))

from methods.taxonomy import check_taxonomy
from methods.get_building_orientation import get_street_view_image

gsv_api_file = rubicai / 'methods/gsv_api_key.txt'
assert gsv_api_file.exists(), "`gsv_api_key.txt` not found in `methods` directory."

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


################### Material model #########################
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

def dl_models():
    global model_material, model_llrs, model_code, model_n_stories, model_occupancy, model_bp, model_rshp, model_rmt
    print("Uploading DL models")
    # Load the model_material architecture
    model_material = models.densenet201(weights=None)  # Initialize model_material without pre-trained weights
    num_features = model_material.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model_material = models.densenet201(weights=None)
    num_features = model_material.classifier.in_features
    model_material.classifier = torch.nn.Linear(num_features, 8)
        
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
    model_llrs = models.densenet201(weights=None)
    num_features = model_llrs.classifier.in_features
    model_llrs.classifier = torch.nn.Linear(num_features, 6)

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
    num_features = model_code.classifier.in_features  # or model.classifier.in_features if replaced earlier
    model_code.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 4)
    )

    # Load the trained weights
    model_code.load_state_dict(torch.load(str(dl_dir / "densenet201_code_level.pt"), map_location=device))
    model_code.to(device)
    model_code.eval()


    ################### N STORIES model #########################
    # Define the device (CPU-only if no GPU is available)

    # Load the model architecture
    model_n_stories = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model_n_stories.classifier.in_features

    # Use the correct number of output classes (9 as indicated in the error)
    model_n_stories = models.densenet201(weights=None)
    num_features = model_n_stories.classifier.in_features
    model_n_stories.classifier = torch.nn.Linear(num_features, 9)

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
    num_features = model_occupancy.classifier.in_features  # or model.classifier.in_features if replaced earlier
    model_occupancy.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 4))

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
    num_features = model_bp.classifier.in_features  # or model.classifier.in_features if replaced earlier
    model_bp.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 4)
    )

    # Load the trained weights
    model_bp.load_state_dict(torch.load(str(dl_dir / "densenet201_block_position.pt"), map_location=device))
    model_bp.to(device)
    model_bp.eval()


    ################### Roof Shape model #########################
    # Define the device (CPU-only if no GPU is available)

    # Load the model architecture
    model_rshp = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model_rshp.classifier.in_features

    # Use the correct number of output classes (9 as indicated in the error)
    num_features = model_rshp.classifier.in_features  # or model.classifier.in_features if replaced earlier
    model_rshp.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 5)
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
    num_features = model_rmt.classifier.in_features  # or model.classifier.in_features if replaced earlier
    model_rmt.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 3)
    )

    # Load the trained weights
    model_rmt.load_state_dict(torch.load(str(dl_dir / "densenet201_roof_material.pt"), map_location=device))
    model_rmt.to(device)
    model_rmt.eval()

    print("The DL models have been successfully uploaded!")
#########################################################
#######===========  Models predicition =========#########
#########################################################

dl_models()

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
    material_classes = ['CR', 'HYB(MCF;MUR)', 'INF','MCF', 'MR', 'MUR','S','W']
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
    llrs_classes = ['LDUAL', 'LFBR', 'LFINF', 'LFM', 'LN', 'LWAL', 'LWAL']
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
    code_level_classes = ['CDH','CDL', 'CDM', 'CDN']
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
        
    occupancy_class = ['COM' , 'IND' ,'MIX(RES;COM)', 'RES']
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
        
    block_position_classes = ['BP1', 'BP2', 'BP3', 'BPD']
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
        
    roof_shape_classes = ['RSH1', 'RSH2', 'RSH3', 'RSH5', 'RSH7']
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
                data_ai.iloc[i, 5] = predict_material_img (image_file)                            # LLRS Material
                data_ai.iloc[i, 6] = predict_llrs_img (image_file)                                # LLRS 
                data_ai.iloc[i, 7] = predict_code_img (image_file)                                # Code Level 
                data_ai.iloc[i, 8] = predict_n_stories_img (image_file)                           # Number of Stories 
                data_ai.iloc[i, 9] = predict_occupancy_img (image_file)                           # Occupancy
                data_ai.iloc[i, 10] = predict_block_position_img (image_file)                     # Block Position
                data_ai.iloc[i, 11] = predict_roof_shape_img (image_file)                         # Roof shape
                data_ai.iloc[i, 12] = predict_roof_material_img (image_file)                      # Roof material
                
                try:
                    data_ai.iloc[i, 13] = (data_ai.iloc[i, 5]+"/"+data_ai.iloc[i, 6]+"/"+data_ai.iloc[i, 7]+"/H:"+
                                            data_ai.iloc[i, 8]+"/"+data_ai.iloc[i, 10]+"/"+data_ai.iloc[i, 11]+"+"+
                                            data_ai.iloc[i, 12]+"/"+data_ai.iloc[i, 9])
                                                                    
                    # Taxonomy
                    tax_check(data_ai.iloc[i, 13])
                except:
                    pass
                
                data_ai.iloc[i, 14] = url_gsv
        except:
            print(" Error in building ID: " + str(footprint_data.loc[i, "id"]))
            pass
        
        print("Inspection: " + str(i+1)+"/"+str(data_ai.shape[0]) +" -------------------------------------")
       
        
#########################################################
#######===========  Input parameters =========###########
#########################################################

local_building_info = rubicai / "demos/specific_coordinates/specific_coordinates_example_data.csv"
saved_path = rubicai / "demos/specific_coordinates/example_prediction_coordinates.csv"

#########################################################
#######===========  Function results =========###########
#########################################################

data_ai = create_database(local_building_info)
inspection_database(data_ai)
data_ai.to_csv(saved_path, index= False)

