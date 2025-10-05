import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

from methods.get_building_orientation import get_street_view_image

############ LLRS prediction ################
def predict_llrs_img (image_path, cont):
    print("Image: ", cont)
    cont += 1
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
    # Accept ndarray or (url, ndarray)
    img_arr = None
    if isinstance(image_path, tuple):
        img_arr = image_path[1]
    else:
        img_arr = image_path
    
    if not _is_valid_img(img_arr):
        print("[WARN] Skipping: invalid/empty image passed to predict_llrs_img")
        return None  # will be ignored in value_counts
    
    # Ensure 3-channel RGB
    if img_arr.shape[2] == 4:  # RGBA
        img_arr = img_arr[:, :, :3]
    
    image = Image.fromarray(np.uint8(img_arr)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        
    # LLRS building image sets prediction
    llrs_classes = ['LDUAL', 'LFINF', 'LFM', 'LWAL', 'TW', 'W']
    llrs_id = llrs_classes[prediction]
    
    return llrs_id


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
    
    # image_path = f"C:/Users/User/Documents/GitHub/RUBIC-AI/demos/local_images/images_ex1/{image_id}"
    
    # Building coordinates
    row = building_data.loc[building_data["id"] == image_id, ["latitude", "longitude"]]
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
    
    if id_feature == "LLRS":
        return predict_llrs_img(img, 1)

def _is_valid_img(arr):
    """Return True if arr is a non-empty HxWx3 uint8 NumPy image."""
    import numpy as np
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
        result = get_street_view_image(location, api_key, heading)
        # result can be (url, ndarray) or just ndarray depending on your impl
        if isinstance(result, tuple):
            img = result[1]
        else:
            img = result
        return img if _is_valid_img(img) else None
    except Exception as e:
        print(f"[WARN] GSV fetch failed at {location}: {e}")
        return None


# ========== Load Dataset ==========
local_building_info = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\local_images\data_ex1.csv"
building_data = pd.read_csv(local_building_info)  # Dataset must include an 'ID' column

# ========== Run the Optimized Sampling ==========
analysis_features = ["LLRS"]
for aux in analysis_features:
    print(" ========== " + aux + " ===========")
    final_sample, class_dist, final_size = iterative_label_discovery_cached_fractional(
        data=building_data,
        labeling_function=lambda x: labeling_function(x, aux),
        id_column='id',
        id_feature=aux,
        initial_fraction=3/6,
        step_fraction=1/6,
        max_fraction=1.00,
        max_iterations=2,
        stability_threshold=0.05
    )
    final_sample.to_csv(f"C:/Users/User/Documents/GitHub/RUBIC-AI/demos/extrapolation/stratified/stratified_{aux}.csv", index=False)
    print("")
