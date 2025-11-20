import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ==========================================================
# 1. LOAD MODEL (CPU ONLY)
# ==========================================================
device = torch.device("cpu")

model = models.densenet201(weights=None)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 9)  # 9 classes for number of stories

model.load_state_dict(torch.load("dl_weights/densenet201_n_stories.pt",
                                 map_location=device))
model.to(device)
model.eval()


# ==========================================================
# 2. IMAGE TRANSFORMS (must match training)
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# 3. GRAD-CAM FUNCTION
# ==========================================================
def generate_gradcam(image_path, output_path="cam_heatmap.jpg"):
    """
    Generates a Grad-CAM heatmap for a DenseNet201 model
    trained for number-of-stories classification.
    """

    # ------------------------------
    # Load original image
    # ------------------------------
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.float32(img_pil.resize((256, 256))) / 255.0  # normalized RGB array

    # ------------------------------
    # Preprocess for model input
    # ------------------------------
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ------------------------------
    # Choose the last convolution layer
    # ------------------------------
    target_layer = model.features[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    # Run model to get predicted class
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # ------------------------------
    # Generate Grad-CAM mask
    # ------------------------------
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(predicted_class)]
    )[0]  # first (and only) batch element

    # ------------------------------
    # Overlay heatmap on the original image
    # ------------------------------
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    print(f"🔥 CAM saved to: {output_path}")
    print(f"🔢 Predicted stories class: {predicted_class}")

    return predicted_class


# ==========================================================
# 4. USAGE EXAMPLE
# ==========================================================
if __name__ == "__main__":
    test_image = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\local_images\images_ex1\Cropped_images\31731527_28_1_cropped.jpg"   # <-- put your image path here
    generate_gradcam(test_image, "cam_heatmap.jpg")
