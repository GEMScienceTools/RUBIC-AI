import os
import pandas as pd
import numpy as np
import requests

from methods.get_building_orientation import get_street_view_image

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# =========================
# USER SETTINGS (edit here)
# =========================
MATERIAL_CLASSES = ['ADO', 'CR', 'MCF', 'MR', 'MUR', 'MX', 'S', 'W']  # <- exactly 8
DL_WEIGHTS_PATH = "dl_weights/densenet201_material.pt"

# Data & output
LOCAL_BUILDING_INFO = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\extrapolation\stratified\data_ex1.csv"
OUTPUT_DIR = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\extrapolation\stratified"

# Extra mode:
#   0 -> fetch GSV (needs lat/lon & API key in methods/gsv_api_key.txt)
#   1 -> load local images (builds path with LOCAL_IMAGES_DIR)
EXTRA_MODE = 0
LOCAL_IMAGES_DIR = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\local_images\images_ex1"

# Sampling parameters
INITIAL_FRAC = 5/15
STEP_FRAC = 3/15
MAX_FRAC = 1.00
MAX_ITERS = 3
STAB_THRESHOLD = 0.05
RANDOM_STATE = 42

# =========================
# MODEL: material predictor
# =========================
def predict_material_img(image_input, extra_mode):
    """
    Predict the construction material from a building image using a DenseNet201 model.

    Args:
        image_input:
            - if extra_mode == 0: NumPy array (BGR/RGB ok; will be converted)
            - if extra_mode == 1: string path to an image file
        extra_mode (int): 0 = array, 1 = file path

    Returns:
        str | None: One of MATERIAL_CLASSES, or None if prediction failed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model head with EXACT 8 outputs (no Flatten, no LogSoftmax)
    model = models.densenet201(weights=None)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, len(MATERIAL_CLASSES))

    # Load weights
    try:
        state = torch.load(DL_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[ERROR] Could not load weights '{DL_WEIGHTS_PATH}': {e}")
        return None

    model.to(device)
    model.eval()

    # Input transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build PIL image
    try:
        if extra_mode == 0:
            # image_input is a NumPy array
            pil_img = Image.fromarray(image_input[..., ::-1]) if image_input.ndim == 3 else Image.fromarray(image_input)
            pil_img = pil_img.convert("RGB")
        else:
            pil_img = Image.open(image_input).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to load/convert image: {e}")
        return None

    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)              # shape: [1, 8]
        pred_idx = int(torch.argmax(logits, dim=1).item())

    # Safety: ensure index in range
    if 0 <= pred_idx < len(MATERIAL_CLASSES):
        return MATERIAL_CLASSES[pred_idx]
    else:
        print(f"[WARN] Predicted index out of range: {pred_idx}")
        return None

# =========================
# GSV coverage check
# =========================
def check_street_view(lat, lon):
    try:
        with open("methods/gsv_api_key.txt", "r") as f:
            api_key = f.read().strip()
    except Exception as e:
        print(f"[ERROR] Could not read GSV API key: {e}")
        return False

    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "key": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data.get("status") == "OK"
    except Exception as e:
        print(f"[WARN] GSV metadata request failed: {e}")
        return False

# =========================
# Labeling function
# =========================
def labeling_function(image_id, id_feature, extra_mode, building_data):
    """
    Returns a material class or None (if not labeled).
    """
    # try:
    if extra_mode == 0:
        # ---- GSV path ----
        row = building_data.loc[building_data["id"] == image_id, ["latitude", "longitude"]]
        if row.empty:
            print(f"[WARN] Missing lat/lon for id={image_id}")
            return None

        lat = float(row.iloc[0]["latitude"])
        lon = float(row.iloc[0]["longitude"])

        if not check_street_view(lat, lon):
            # No coverage — skip labeling (do not contaminate class distribution)
            return None

        # angles could be a loop if you want multiple views; keep 0 for simplicity
        angle = 0
        try:
            # Expecting: (maps_url, img)
            maps_url, img = get_street_view_image((lat, lon), _read_api_key(), angle)
        except TypeError:
            # If your function returns only the image, handle that:
            res = get_street_view_image((lat, lon), _read_api_key(), angle)
            if isinstance(res, tuple):
                _, img = res
            else:
                img = res

        if img is None:
            return None

        return predict_material_img(img, extra_mode=0)

    else:
        # ---- Local images path ----
        img_path = os.path.join(LOCAL_IMAGES_DIR, str(image_id))
        if not os.path.isfile(img_path):
            print(f"[WARN] Image not found: {img_path}")
            return None
        return predict_material_img(img_path, extra_mode=1)

    # except Exception as e:
    #     print(f"[WARN] Labeling failed for id={image_id}: {e}")
    #     return None

def _read_api_key():
    with open("methods/gsv_api_key.txt", "r") as f:
        return f.read().strip()

# =========================
# Iterative sampling
# =========================
def iterative_label_discovery_cached_fractional(
    id_feature,
    data: pd.DataFrame,
    labeling_fn,
    id_column: str = 'id',
    initial_fraction: float = 0.10,
    step_fraction: float = 0.05,
    max_fraction: float = 1.00,
    stability_threshold: float = 0.05,
    max_iterations: int = 20,
    random_state: int = 42
):
    """
    Iteratively samples and labels data until class distribution stabilizes.
    Rows that cannot be labeled (label None) are excluded from the labeled pool & distribution.
    """
    population_size = len(data)
    all_labeled = pd.DataFrame(columns=[id_column, id_feature])
    previous_dist = None

    np.random.seed(random_state)
    shuffled = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    iteration = 0
    while iteration < max_iterations:
        current_fraction = min(initial_fraction + step_fraction * iteration, max_fraction)
        target_size = min(int(population_size * current_fraction), population_size)

        already_ids = set(all_labeled[id_column])
        need = target_size - len(all_labeled)
        if need <= 0:
            # We already have enough labeled rows to test stability
            need = 0

        # pull a batch bigger than 'need' to offset possible None labels
        # (simple heuristic: 2x)
        batch = shuffled[~shuffled[id_column].isin(already_ids)].head(max(need * 2, 1)).copy()
        if batch.empty:
            break

        # label the batch
        batch[id_feature] = batch[id_column].apply(lambda x: labeling_fn(x))
        # keep only valid labels
        valid_batch = batch.dropna(subset=[id_feature])

        if not valid_batch.empty:
            all_labeled = pd.concat([all_labeled, valid_batch[[id_column, id_feature]]], ignore_index=True)
            all_labeled.drop_duplicates(subset=[id_column], inplace=True)

        # compute current distribution over ONLY material classes
        if not all_labeled.empty:
            current_counts = all_labeled[id_feature].value_counts(normalize=True).sort_index()
            current_dist = current_counts.to_dict()
        else:
            current_dist = {}

        if previous_dist is not None and current_dist:
            all_keys = set(previous_dist) | set(current_dist)
            max_change = max(abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys)
            print(f"Iteration {iteration+1}: labeled={len(all_labeled)}  maxΔ={max_change:.4f}")
            if max_change < stability_threshold:
                print("✅ Class proportions stabilized.")
                return all_labeled, current_dist, len(all_labeled)

        previous_dist = current_dist
        iteration += 1

    print("⚠️ Reached max iterations or sample limit without convergence.")
    return all_labeled, previous_dist if previous_dist else {}, len(all_labeled)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Load data (must include 'id', 'latitude', 'longitude' if EXTRA_MODE==0)
    building_data = pd.read_csv(LOCAL_BUILDING_INFO)

    analysis_features = ["LLRS Material"]
    sample_sizes = []

    for feat in analysis_features:
        print(f"\n========== {feat} ==========")

        # bind a labeling_fn with current feature & mode
        def _lf(image_id):
            return labeling_function(image_id, feat, EXTRA_MODE, building_data)

        final_sample, class_dist, final_size = iterative_label_discovery_cached_fractional(
            id_feature=feat,
            data=building_data,
            labeling_fn=_lf,
            id_column="id",
            initial_fraction=INITIAL_FRAC,
            step_fraction=STEP_FRAC,
            max_fraction=MAX_FRAC,
            max_iterations=MAX_ITERS,
            stability_threshold=STAB_THRESHOLD,
            random_state=RANDOM_STATE
        )

        sample_sizes.append(final_size)

        # Save only labeled rows
        out_csv = os.path.join(OUTPUT_DIR, f"stratified_dl_{feat}.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_sample.to_csv(out_csv, index=False)

        print("Class distribution:", class_dist)
        print("Saved:", out_csv)

    print("Max sample size over features:", np.max(sample_sizes) if sample_sizes else 0)
