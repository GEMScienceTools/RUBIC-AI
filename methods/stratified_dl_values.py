import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

############ Material prediction ################
def predict_material_img (image_path):
    """
    Predict the construction material of a building using a pre-trained DenseNet201 model.

    This function loads a trained DenseNet201 model to classify the material of a 
    building from an input image. It applies necessary preprocessing and normalization 
    before performing inference.

    Args:
        image_path (str or np.ndarray): 
            - If `insp_method != 2`, this is expected to be a NumPy array representing 
              an image (assumed to be from an in-memory image).
            - If `insp_method == 2`, this is a file path to the image.
        insp_method (int): Inspection method identifier that determines how the image 
                           is processed.
                           
    Returns:
        int: The predicted class index representing the construction material.

    Effects:
        - Loads a DenseNet201 model and applies necessary transformations.
        - Performs inference on the input image.
        - Returns the class index with the highest probability.

    Notes:
        - The model architecture is initialized with 8 output classes.
        - The function assumes the model weights are stored in `"dl_weights/densenet201_material.pt"`.
        - The image is resized to `(256, 320)` and normalized before inference.
        - Uses `cuda` if available; otherwise, defaults to `cpu`.
        - If `insp_method != 2`, the image is assumed to be a NumPy array and converted 
          to a PIL image before processing.
    """
    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 8),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_material.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    # LLRS building image sets prediction
    material_classes = ['ADO', 'CR', 'MCF', 'MR', 'MUR', 'MX', 'S', 'W']
    material_id = material_classes[prediction]
        
    return material_id


############ LLRS prediction ################
def predict_llrs_img (image_path):

    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 6),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_llrs.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image
    image = Image.open(image_path).convert("RGB") 
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    # LLRS building image sets prediction
    llrs_classes = ['LDUAL', 'LFINF', 'LFM', 'LWAL', 'TW', 'W']
    llrs_id = llrs_classes[prediction]
    
    return llrs_id


############ Code level prediction ################
def predict_code_img (image_path):
    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 4),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_code.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    # code_level building image sets prediction
    code_level_classes = ['CDH', 'CDM', 'CDL', 'CDN']
    code_level_id = code_level_classes[prediction]
    return code_level_id


############ Number of Stories prediction ################
def predict_n_stories_img (image_path):

    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 9),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_n_stories.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    class_names = ['10-12', '13+', '1', '2', '3', '4', '5', '6-7', '8-9']
    n_stories_id = class_names[prediction]
    return n_stories_id


############ Occupancy prediction ################
def predict_occupancy_img (image_path):
    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 7),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_occupancy.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    occupancy_class = ['COM', 'EDU', 'GOV', 'IND', 'MIX', 'OCO', 'RES']
    occupancy_id = occupancy_class[prediction]
    return occupancy_id


############ Block Position prediction ################
def predict_block_position_img (image_path):

    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 3),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_block.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    block_position_classes = ['BP1', 'BP2', 'BPD']
    block_position_id = block_position_classes[prediction]
    return block_position_id

############ Roof Shape prediction ################
def predict_roof_shape_img (image_path):

    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 3),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_roof_shape.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image
    # Load and preprocess the image
    
    # Function to predict the class of an image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    roof_shape_classes = ['RSH1', 'RSH2', 'RSH3']
    roof_shape_id = roof_shape_classes[prediction]
    return roof_shape_id

############ Roof Material prediction ################
def predict_roof_material_img (image_path):

    # Define the device (CPU-only if no GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture
    model = models.densenet201(weights=None)  # Initialize model without pre-trained weights
    num_features = model.classifier.in_features
    
    # Use the correct number of output classes (9 as indicated in the error)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_features, 3),  # Match the number of classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("dl_weights/densenet201_roof_material.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Define the image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Function to predict the class of an image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    roof_material_classes = ['RMN', 'RMT1', 'RMT6']
    roof_material_id = roof_material_classes[prediction]
    return roof_material_id
    
       
####################################################
####################################################
####################################################

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
        all_labeled = pd.concat([all_labeled, next_sample], ignore_index=True)

        # Calculate class distribution
        current_counts = all_labeled[id_feature].value_counts(normalize=True).sort_index()
        current_dist = current_counts.to_dict()

        # Check for stabilization in label distribution
        if previous_dist is not None:
            all_keys = set(previous_dist) | set(current_dist)
            max_change = max(abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys)
            print(f"Iteration {iteration+1}: Sample size = {len(all_labeled)}, Max Δ = {max_change:.4f}")

            if max_change < stability_threshold:
                print("✅ Class proportions stabilized.")
                return all_labeled, current_dist, len(all_labeled)

        previous_dist = current_dist
        iteration += 1

    print("⚠️ Reached max iterations or sample limit without convergence.")
    return all_labeled, current_dist, len(all_labeled)

####################################################
####################################################
####################################################

# =========== Data to change ============


# ========== Labeling Function ==========
def labeling_function(image_id, id_feature):
    """
    Applies a prediction model to a building image given its ID.

    Args:
        image_id (str): Unique identifier for the image.

    Returns:
        str: Predicted LLRS Material class for the building.
    """
    image_path = f"H:/My Drive/Sura_2025_AI/Microsoft_buildings/el_socorro_images/{image_id}.jpg"
    if id_feature == "LLRS":
        return predict_llrs_img(image_path)
    elif id_feature == "LLRS Material":
        return predict_material_img(image_path)
    elif id_feature == "Number of Stories":
        return predict_n_stories_img(image_path)
    elif id_feature == "Occupancy":
        return predict_occupancy_img(image_path)
    elif id_feature == "Code Level":
        return predict_code_img(image_path)
    elif id_feature == "Block Position":
        return predict_block_position_img(image_path)
    elif id_feature == "Roof Shape":
        return predict_roof_shape_img(image_path)
    elif id_feature == "Roof Material":
        return predict_roof_material_img(image_path)
    else:
        return None

