import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from PyQt5.QtWidgets import QMessageBox
import torch.nn as nn

############ Material prediction ################
def predict_material_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained DenseNet201 model.
    """

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- MODEL: DenseNet201 with correct classifier for this checkpoint ----
    model = models.densenet201(weights=None)  # base architecture
    num_features = model.classifier.in_features  # 1920 for densenet201

    # During training you had: classifier[1] = Linear(1920, 8) with a Dropout before
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 8)   # 8 material classes
    )

    # Load the trained DenseNet201 weights
    state_dict = torch.load("dl_weights/densenet201_material.pt",
                            map_location=device)
    model.load_state_dict(state_dict)  # strict=True (default)
    model.to(device)
    model.eval()
    
    # ---- Transforms (same as you used) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # ---- Inference logic: same structure as your original ----
    try:
        if insp_method != 2:
            # image_path is a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            if box_id is not None:
                # image_path is a NumPy array (cropped bbox)
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                # image_path is a file path
                image = Image.open(image_path).convert("RGB")
   
        image = transform(image).unsqueeze(0).to(device)
        
        # ---- Inference ----
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
            
        return prediction

    except Exception as e:
        print(f"[predict_material_img] Error: {e}")

        QMessageBox.warning(
            self,
            "Image Error",
            f"No Street View image found or no building detected for ID: {box_id}"
        )
        return None



############ LLRS prediction ################
def predict_llrs_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained DenseNet201 model.
    """

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- MODEL: DenseNet201 with correct classifier for this checkpoint ----
    model = models.densenet201(weights=None)  # base architecture
    num_features = model.classifier.in_features  # 1920 for densenet201

    # During training you had: classifier[1] = Linear(1920, 8) with a Dropout before
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 6)   # 8 material classes
    )

    # Load the trained DenseNet201 weights
    state_dict = torch.load("dl_weights/densenet201_llrs.pt",
                            map_location=device)
    model.load_state_dict(state_dict)  # strict=True (default)
    model.to(device)
    model.eval()
    
    # ---- Transforms (same as you used) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # ---- Inference logic: same structure as your original ----
    try:
        if insp_method != 2:
            # image_path is a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            if box_id is not None:
                # image_path is a NumPy array (cropped bbox)
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                # image_path is a file path
                image = Image.open(image_path).convert("RGB")
   
        image = transform(image).unsqueeze(0).to(device)
        
        # ---- Inference ----
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
            
        return prediction

    except Exception as e:
        print(f"[predict_material_img] Error: {e}")
        return None



############ Code level prediction ################
def predict_code_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained ConvNeXt-Tiny model.

    This function loads a trained ConvNeXt-Tiny model to classify the material of a 
    building from an input image. It applies necessary preprocessing and normalization 
    before performing inference.

    Args:
        image_path (str or np.ndarray): 
            - If `insp_method != 2`, this is expected to be a NumPy array representing 
              an image (assumed to be from an in-memory image).
            - If `insp_method == 2`, this is a file path to the image OR a NumPy array
              if a cropped bounding box is passed.
        insp_method (int): Inspection method identifier that determines how the image 
                           is processed.
                           
    Returns:
        int or None: The predicted class index (0–7) representing the construction material,
                     or None if there is an error.

    Notes:
        - Uses ConvNeXt-Tiny with 8 output classes.
        - Assumes weights are stored in "dl_weights/convnext_tiny_material.pt".
        - The image is resized to (256, 256) and normalized before inference.
        - Uses `cuda` if available; otherwise defaults to `cpu`.
    """
    # ---- Device (same structure as before) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- MODEL: ConvNeXt-Tiny instead of DenseNet (this is the key fix) ----
    # Base ConvNeXt-Tiny architecture
    model = models.convnext_tiny(weights=None)

    # Default ConvNeXt classifier is:
    # classifier[0] = LayerNorm(768)
    # classifier[1] = Flatten
    # classifier[2] = Linear(768, num_classes)
    #
    # During training (based on the checkpoint keys: classifier.2.0, classifier.2.1),
    # you used something like:
    # classifier[2] = nn.Sequential(Dropout(p=0.2), Linear(768, 8))
    in_features = model.classifier[2].in_features  # should be 768

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 4)   # 8 material classes
    )

    # Load the trained ConvNeXt weights
    state_dict = torch.load("dl_weights/convnext_tiny_code_level.pt", map_location=device)
    model.load_state_dict(state_dict)   # strict=True by default
    model.to(device)
    model.eval()

    # ---- Transforms (kept as in your code) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---- Inference logic (kept as close as possible to your structure) ----
    try:
        if insp_method != 2:
            # image_path is expected to be a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            # If box_id is not None, image_path is a NumPy array (cropped bbox)
            # Otherwise, image_path is a file path
            if box_id is not None:
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return prediction

    except Exception as e:
        # (Optional) print error to console/log for debugging
        print(f"[predict_material_img] Error: {e}")
        return None


############ Number of Stories prediction ################
def predict_n_stories_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained ConvNeXt-Tiny model.

    This function loads a trained ConvNeXt-Tiny model to classify the material of a 
    building from an input image. It applies necessary preprocessing and normalization 
    before performing inference.

    Args:
        image_path (str or np.ndarray): 
            - If `insp_method != 2`, this is expected to be a NumPy array representing 
              an image (assumed to be from an in-memory image).
            - If `insp_method == 2`, this is a file path to the image OR a NumPy array
              if a cropped bounding box is passed.
        insp_method (int): Inspection method identifier that determines how the image 
                           is processed.
                           
    Returns:
        int or None: The predicted class index (0–7) representing the construction material,
                     or None if there is an error.

    Notes:
        - Uses ConvNeXt-Tiny with 8 output classes.
        - Assumes weights are stored in "dl_weights/convnext_tiny_material.pt".
        - The image is resized to (256, 256) and normalized before inference.
        - Uses `cuda` if available; otherwise defaults to `cpu`.
    """
    # ---- Device (same structure as before) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- MODEL: ConvNeXt-Tiny instead of DenseNet (this is the key fix) ----
    # Base ConvNeXt-Tiny architecture
    model = models.convnext_tiny(weights=None)

    # Default ConvNeXt classifier is:
    # classifier[0] = LayerNorm(768)
    # classifier[1] = Flatten
    # classifier[2] = Linear(768, num_classes)
    #
    # During training (based on the checkpoint keys: classifier.2.0, classifier.2.1),
    # you used something like:
    # classifier[2] = nn.Sequential(Dropout(p=0.2), Linear(768, 8))
    in_features = model.classifier[2].in_features  # should be 768

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 9)   # 8 material classes
    )

    # Load the trained ConvNeXt weights
    state_dict = torch.load("dl_weights/convnext_tiny_n_stories.pt", map_location=device)
    model.load_state_dict(state_dict)   # strict=True by default
    model.to(device)
    model.eval()

    # ---- Transforms (kept as in your code) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---- Inference logic (kept as close as possible to your structure) ----
    try:
        if insp_method != 2:
            # image_path is expected to be a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            # If box_id is not None, image_path is a NumPy array (cropped bbox)
            # Otherwise, image_path is a file path
            if box_id is not None:
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return prediction

    except Exception as e:
        # (Optional) print error to console/log for debugging
        print(f"[predict_material_img] Error: {e}")
        return None



############ Occupancy prediction ################
def predict_occupancy_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained ConvNeXt-Tiny model.

    This function loads a trained ConvNeXt-Tiny model to classify the material of a 
    building from an input image. It applies necessary preprocessing and normalization 
    before performing inference.

    Args:
        image_path (str or np.ndarray): 
            - If `insp_method != 2`, this is expected to be a NumPy array representing 
              an image (assumed to be from an in-memory image).
            - If `insp_method == 2`, this is a file path to the image OR a NumPy array
              if a cropped bounding box is passed.
        insp_method (int): Inspection method identifier that determines how the image 
                           is processed.
                           
    Returns:
        int or None: The predicted class index (0–7) representing the construction material,
                     or None if there is an error.

    Notes:
        - Uses ConvNeXt-Tiny with 8 output classes.
        - Assumes weights are stored in "dl_weights/convnext_tiny_material.pt".
        - The image is resized to (256, 256) and normalized before inference.
        - Uses `cuda` if available; otherwise defaults to `cpu`.
    """
    # ---- Device (same structure as before) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- MODEL: ConvNeXt-Tiny instead of DenseNet (this is the key fix) ----
    # Base ConvNeXt-Tiny architecture
    model = models.convnext_tiny(weights=None)

    # Default ConvNeXt classifier is:
    # classifier[0] = LayerNorm(768)
    # classifier[1] = Flatten
    # classifier[2] = Linear(768, num_classes)
    #
    # During training (based on the checkpoint keys: classifier.2.0, classifier.2.1),
    # you used something like:
    # classifier[2] = nn.Sequential(Dropout(p=0.2), Linear(768, 8))
    in_features = model.classifier[2].in_features  # should be 768

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 4)   # 8 material classes
    )

    # Load the trained ConvNeXt weights
    state_dict = torch.load("dl_weights/convnext_tiny_occupancy.pt", map_location=device)
    model.load_state_dict(state_dict)   # strict=True by default
    model.to(device)
    model.eval()

    # ---- Transforms (kept as in your code) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---- Inference logic (kept as close as possible to your structure) ----
    try:
        if insp_method != 2:
            # image_path is expected to be a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            # If box_id is not None, image_path is a NumPy array (cropped bbox)
            # Otherwise, image_path is a file path
            if box_id is not None:
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return prediction

    except Exception as e:
        # (Optional) print error to console/log for debugging
        print(f"[predict_material_img] Error: {e}")
        return None



############ Block Position prediction ################
def predict_block_position_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained DenseNet201 model.
    """

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- MODEL: DenseNet201 with correct classifier for this checkpoint ----
    model = models.densenet201(weights=None)  # base architecture
    num_features = model.classifier.in_features  # 1920 for densenet201

    # During training you had: classifier[1] = Linear(1920, 8) with a Dropout before
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 4)   # 8 material classes
    )

    # Load the trained DenseNet201 weights
    state_dict = torch.load("dl_weights/densenet201_block_position.pt",
                            map_location=device)
    model.load_state_dict(state_dict)  # strict=True (default)
    model.to(device)
    model.eval()
    
    # ---- Transforms (same as you used) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # ---- Inference logic: same structure as your original ----
    try:
        if insp_method != 2:
            # image_path is a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            if box_id is not None:
                # image_path is a NumPy array (cropped bbox)
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                # image_path is a file path
                image = Image.open(image_path).convert("RGB")
   
        image = transform(image).unsqueeze(0).to(device)
        
        # ---- Inference ----
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
            
        return prediction

    except Exception as e:
        print(f"[predict_material_img] Error: {e}")
        return None


############ Roof Shape prediction ################
def predict_roof_shape_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained DenseNet201 model.
    """

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- MODEL: DenseNet201 with correct classifier for this checkpoint ----
    model = models.densenet201(weights=None)  # base architecture
    num_features = model.classifier.in_features  # 1920 for densenet201

    # During training you had: classifier[1] = Linear(1920, 8) with a Dropout before
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 5)   # 8 material classes
    )

    # Load the trained DenseNet201 weights
    state_dict = torch.load("dl_weights/densenet201_roof_shape.pt",
                            map_location=device)
    model.load_state_dict(state_dict)  # strict=True (default)
    model.to(device)
    model.eval()
    
    # ---- Transforms (same as you used) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # ---- Inference logic: same structure as your original ----
    try:
        if insp_method != 2:
            # image_path is a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            if box_id is not None:
                # image_path is a NumPy array (cropped bbox)
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                # image_path is a file path
                image = Image.open(image_path).convert("RGB")
   
        image = transform(image).unsqueeze(0).to(device)
        
        # ---- Inference ----
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
            
        return prediction

    except Exception as e:
        print(f"[predict_material_img] Error: {e}")
        return None


############ Roof Material prediction ################
def predict_roof_material_img(image_path, insp_method, box_id, self):
    """
    Predict the construction material of a building using a pre-trained DenseNet201 model.
    """

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- MODEL: DenseNet201 with correct classifier for this checkpoint ----
    model = models.densenet201(weights=None)  # base architecture
    num_features = model.classifier.in_features  # 1920 for densenet201

    # During training you had: classifier[1] = Linear(1920, 8) with a Dropout before
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 3)   # 8 material classes
    )

    # Load the trained DenseNet201 weights
    state_dict = torch.load("dl_weights/densenet201_roof_material.pt",
                            map_location=device)
    model.load_state_dict(state_dict)  # strict=True (default)
    model.to(device)
    model.eval()
    
    # ---- Transforms (same as you used) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # ---- Inference logic: same structure as your original ----
    try:
        if insp_method != 2:
            # image_path is a NumPy array
            image = Image.fromarray(image_path)
            image = image.convert("RGB")
        else:
            # insp_method == 2
            if box_id is not None:
                # image_path is a NumPy array (cropped bbox)
                image = Image.fromarray(image_path)
                image = image.convert("RGB")
            else:
                # image_path is a file path
                image = Image.open(image_path).convert("RGB")
   
        image = transform(image).unsqueeze(0).to(device)
        
        # ---- Inference ----
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
            
        return prediction

    except Exception as e:
        print(f"[predict_material_img] Error: {e}")
        return None
