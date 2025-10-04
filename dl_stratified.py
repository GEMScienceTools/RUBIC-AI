import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

############ Material prediction ################
def predict_material_img (image_path, extra_mode):
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
    if extra_mode == 0:
        image = Image.fromarray(image_path)
        image = image.convert("RGB")    
        image = transform(image).unsqueeze(0).to(device)
    else:
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
