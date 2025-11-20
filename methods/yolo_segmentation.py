from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------------------------------------
# Load YOLOv11 SEGMENTATION model (pretrained)
# Options: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, etc.
# -----------------------------------------------------------
model = YOLO("yolo11s-seg.pt")

# -----------------------------------------------------------
# Load the image
# -----------------------------------------------------------
img_path = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\local_images\images_ex1\Cropped_images\31731527_28_1_cropped.jpg"
img = cv2.imread(img_path)

# -----------------------------------------------------------
# Run inference
# -----------------------------------------------------------
results = model(img)[0]

# -----------------------------------------------------------
# Draw results manually (masks + boxes + labels)
# -----------------------------------------------------------
output = img.copy()

if results.masks is not None:
    masks = results.masks.data.cpu().numpy()     # N x H x W
    boxes = results.boxes.xyxy.cpu().numpy()     # N x 4
    confs = results.boxes.conf.cpu().numpy()     # N
    classes = results.boxes.cls.cpu().numpy()    # N

    for i, mask in enumerate(masks):
        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (output.shape[1], output.shape[0]))  # (width, height)
    
        # Convert to uint8
        mask_resized = (mask_resized > 0.5).astype(np.uint8)
    
        # Create the color overlay
        colored_mask = np.zeros_like(output)
        colored_mask[:, :, 2] = mask_resized * 255  # Red channel
    
        # Overlay with transparency
        output = cv2.addWeighted(output, 1, colored_mask, 0.5, 0)
    
        # Draw bounding box
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255), 3)
    
        # Label
        label = f"{model.names[int(classes[i])]} {confs[i]:.2f}"
        cv2.putText(output, label, (int(x1), int(y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# -----------------------------------------------------------
# Save and display
# -----------------------------------------------------------
cv2.imwrite("segmented_output.jpg", output)
print("Output saved as segmented_output.jpg")
