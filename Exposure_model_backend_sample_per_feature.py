import pandas as pd
import numpy as np
from methods.get_building_orientation import get_street_view_image
import requests
     
from dl_stratified import predict_llrs_img, predict_material_img, predict_code_img, predict_roof_shape_img
from dl_stratified import predict_occupancy_img, predict_block_position_img, predict_n_stories_img, predict_roof_material_img
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
      
    # if check_street_view(lat,lon) == True:
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

####################################################
####################################################
####################################################

# =========== Data to change ============
############ Checks if there is GSV availability ################  
def check_street_view(lat,lon):
    # Input parameters
    with open("methods/gsv_api_key.txt", "r") as f:
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


# ========== Labeling Function ==========
def labeling_function(image_id, id_feature, extra_mode):
    
    """
    Applies a prediction model to a building image given its ID.

    Args:
        image_id (str): Unique identifier for the image.

    Returns:
        str: Predicted LLRS Material class for the building.
    """
    if extra_mode == 0:
        lat_row = building_data.loc[building_data["id"] == image_id, "latitude"]
        lon_row = building_data.loc[building_data["id"] == image_id, "longitude"]
        lat = lat_row.iloc[0]
        lon = lon_row.iloc[0]
        # Building coordinates
        location = (lat,lon)
        # angles for taking the images
        angle = 0
        # Input parameters
        with open("methods/gsv_api_key.txt", "r") as f:
            api_key = f.read().strip()
            
        image_path = get_street_view_image(location, api_key, angle) 
    else:
        
        image_path = f"C:/Users/User/Documents/GitHub/RUBIC-AI/demos/local_images/images_ex1/{image_id}"
    if id_feature == "LLRS":
        return predict_llrs_img(image_path)
    elif id_feature == "LLRS Material":
        return predict_material_img(image_path, extra_mode)
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

# ========== Load Dataset ==========
local_building_info = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\local_images\data_ex1.csv"
building_data = pd.read_csv(local_building_info)  # Dataset must include an 'ID' column

# ========== Run the Optimized Sampling ==========
analysis_features = ["LLRS Material"]
extra_mode = 1
sample_size_def = []
for aux in analysis_features:
    print(" ========== " + aux + " ===========")
    final_sample, class_dist, final_size = iterative_label_discovery_cached_fractional(
        data=building_data,
        labeling_function=lambda x: labeling_function(x, aux, extra_mode),
        id_column='id',
        id_feature=aux,
        initial_fraction=5/15,
        step_fraction=3/15,
        max_fraction=1.00,
        max_iterations=3,
        stability_threshold=0.05
    )
    sample_size_def.append(len(final_sample))
    final_sample.to_csv(f"C:/Users/User/Documents/GitHub/RUBIC-AI/demos/extrapolation/stratified_dl_{aux}.csv", index=False)
    print("")
    
print("Sample size definitive: ", np.max(sample_size_def))