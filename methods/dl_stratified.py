import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from methods.get_building_orientation import get_street_view_image

# ========== Iterative sampling using existing labels ==========
def iterative_distribution_stability_manual(
    id_feature,
    data: pd.DataFrame,
    id_column,
    initial_fraction,
    step_fraction,
    max_fraction,
    stability_threshold,
    max_iterations,
    random_state: int = 42
):
    """
    Iteratively samples data using existing class labels until label distribution stabilizes.

    Args:
        id_feature (str): Column name with class labels (e.g., 'LLRS', 'Taxonomy').
        data (pd.DataFrame): Dataset containing existing labels.
        id_column (str): Column used as unique identifier.
        initial_fraction (float): Starting fraction of dataset.
        step_fraction (float): Step increase per iteration.
        max_fraction (float): Maximum sample fraction.
        stability_threshold (float): Max change in distribution to stop iterations.
        max_iterations (int): Maximum number of iterations.
        random_state (int): Random seed.

    Returns:
        DataFrame: Final sampled data.
        dict: Final class distribution.
        int: Final sample size.
    """

    population_size = len(data)
    all_sampled = pd.DataFrame(columns=data.columns)
    previous_dist = None
    iteration = 0

    # Shuffle dataset
    np.random.seed(random_state)
    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    while iteration < max_iterations:
        current_fraction = min(initial_fraction + step_fraction * iteration, max_fraction)
        target_size = int(population_size * current_fraction)

        # Select next sample
        remaining = shuffled_data[~shuffled_data[id_column].isin(all_sampled[id_column])]
        next_sample = remaining.head(target_size - len(all_sampled))

        if next_sample.empty:
            break

        all_sampled = pd.concat([all_sampled, next_sample], ignore_index=True)

        # Compute current distribution
        current_counts = all_sampled[id_feature].value_counts(normalize=True).sort_index()
        current_dist = current_counts.to_dict()

        # Check stabilization
        if previous_dist is not None:
            all_keys = set(previous_dist) | set(current_dist)
            max_change = max(abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys)
            print(f"Iteration {iteration+1}: Sample size = {len(all_sampled)}, Max Δ = {max_change:.4f}")

            if max_change < stability_threshold:
                print("✅ Class proportions stabilized.")
                return all_sampled, current_dist, len(all_sampled)

        previous_dist = current_dist
        iteration += 1

    print("⚠️ Reached max iterations or sample limit without convergence.")
    return all_sampled, current_dist, len(all_sampled)



# ========== Main Iterative Sampling Function ==========
def iterative_label_discovery_cached_fractional(
    id_feature,
    data: pd.DataFrame,
    labeling_function,
    id_column: str = 'id',
    initial_fraction: float = 0.10,
    step_fraction: float = 0.05,
    max_fraction: float = 1.00,
    stability_threshold: float = 0.05,
    max_iterations: int = 20,
    random_state: int = 42
):
    """
    Iteratively samples and labels data using a labeling function until label distribution stabilizes.

    Args:
        data (pd.DataFrame): Input dataset with unique IDs.
        labeling_function (callable): Function to assign labels based on the ID.
        id_column (str): Column name with unique IDs (default: 'ID').
        initial_fraction (float): Initial fraction of data to label.
        step_fraction (float): Additional fraction added each iteration.
        max_fraction (float): Maximum fraction of data to label.
        stability_threshold (float): Maximum allowed change in class distribution for convergence.
        max_iterations (int): Maximum number of iterations.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (DataFrame of labeled samples, label distribution as dict, final sample size)
    """
    population_size = len(data)
    all_labeled = pd.DataFrame(columns=[id_column, id_feature])  # Initialize labeled dataset
    previous_dist = None  # Store label distribution from previous iteration
    iteration = 0  # Iteration counter
    # Shuffle the dataset for randomized sampling
    np.random.seed(random_state)
    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # === Iterative sampling loop ===
    while iteration < max_iterations:
        # Calculate current target sample size
        current_fraction = min(initial_fraction + step_fraction * iteration, max_fraction)
        target_size = min(int(population_size * current_fraction), population_size)
        # Filter out already labeled IDs and select next batch
        already_labeled_ids = set(all_labeled[id_column])
        next_sample = shuffled_data[~shuffled_data[id_column].isin(already_labeled_ids)].head(target_size - len(all_labeled))
        if next_sample.empty:
            break  # Stop if no more samples to process
        # Apply labeling function to new samples
        next_sample[id_feature] = next_sample[id_column].apply(labeling_function)
        print("Next sample: ", next_sample)
        all_labeled = pd.concat([all_labeled, next_sample], ignore_index=True)

        # Calculate class distribution
        current_counts = all_labeled[id_feature].value_counts(normalize=True).sort_index()
        current_dist = current_counts.to_dict()

        # Check for stabilization in label distribution
        if previous_dist is not None:
            all_keys = set(previous_dist) | set(current_dist)
            print("Previous dist: ")
            print(previous_dist)
            max_change = max(abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys)
            print(f"Iteration {iteration+1}: Sample size = {len(all_labeled)}, Max Δ = {max_change:.4f}")

            if max_change < stability_threshold:
                print("✅ Class proportions stabilized.")
                return all_labeled, current_dist, len(all_labeled)

        previous_dist = current_dist
        iteration += 1

    print("⚠️ Reached max iterations or sample limit without convergence.")
    return all_labeled, current_dist, len(all_labeled)

# ========== Labeling Function ==========
def labeling_function(image_id, id_feature, data_building):
    """
    Applies a prediction model to a building image given its ID.

    Args:
        image_id (str): Unique identifier for the image.

    Returns:
        str: Predicted LLRS Material class for the building.
    """
    # Building coordinates
    row = data_building.loc[data_building["id"] == image_id, ["latitude", "longitude"]]
    if row.empty:
        print(f"[WARN] Missing lat/lon for id={image_id}")
        return None
    lat = float(row.iloc[0]["latitude"])
    lon = float(row.iloc[0]["longitude"])
    
    location = (lat,lon)
    # API key is required; without it, access to GSV is not possible
    with open("methods/gsv_api_key.txt", "r") as f:
        api_key = f.read().strip() 
                            
    img = safe_get_gsv_image(location, api_key, 0)
   
    if img is None:
        # No GSV available here; return None so pandas ignores it in value_counts
        print(f"[INFO] No GSV imagery for id={image_id} at {location}. Skipping.")
        return None
    
    if id_feature == "llrs":
        return predict_llrs_img_stratified (img)
    elif id_feature == "material":
        return predict_material_img_stratified (img)
    elif id_feature == "n_stories":
        return predict_n_stories_img_stratified (img)
    elif id_feature == "occupancy":
        return predict_occupancy_img_stratified (img)
    elif id_feature == "code_level":
        return predict_code_img_stratified (img)
    elif id_feature == "block_position":
        return predict_block_position_img_stratified (img)
    elif id_feature == "roof_shape":
        return predict_roof_shape_img_stratified (img)
    elif id_feature == "roof_material":
        return predict_roof_material_img_stratified (img)

def _is_valid_img(arr):
    """Return True if arr is a non-empty HxWx3 uint8 NumPy image."""
    return (
        isinstance(arr, np.ndarray) and
        arr.ndim == 3 and arr.shape[2] in (3, 4) and
        arr.size > 0
    )

def safe_get_gsv_image(location, api_key, heading=0):
    """
    Call get_street_view_image and return ONLY the ndarray,
    or None if GSV is unavailable / request fails.
    """
    try:
        url , result, year = get_street_view_image(location, api_key, heading, 5, 120)
        # result can be (url, ndarray) or just ndarray depending on your impl
        if isinstance(result, tuple):
            img = result[1]
        else:
            img = result
        return img if _is_valid_img(img) else None
    except Exception as e:
        print(f"[WARN] GSV fetch failed at {location}: {e}")
        return None

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
dl_dir = (root_dir / '..' / 'dl_weights').resolve()

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

############ Material prediction ################
def predict_material_img_stratified (image_path):
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
def predict_llrs_img_stratified (image_path):
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
def predict_code_img_stratified (image_path):
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
def predict_n_stories_img_stratified (image_path):
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
def predict_occupancy_img_stratified (image_path): 
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
def predict_block_position_img_stratified (image_path):
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
def predict_roof_shape_img_stratified (image_path):
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
def predict_roof_material_img_stratified (image_path):
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
