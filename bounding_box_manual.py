import cv2
import numpy as np
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class BoundingBoxWindow(QDialog):
    def __init__(self, image_path, frame, insp_method, cropped_path, parent=None, main_window=None, gui_methods=None):
        super().__init__(parent)
        self.main_window = main_window  # Reference to the main window (GUIInterface)
        self.gui_methods = gui_methods  # Reference to GUIMethods
       
        """Initialize the pop-up window for manual bounding box selection."""
        super().__init__()
        
        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080

        self.setWindowTitle("Manual Bounding Box Selection")
        self.setGeometry(int(100*sf_x), int(100*sf_y), int(700*sf_x), int(600*sf_y))  # Fixed window size
        
        self.image_path = image_path
        self.insp_method = insp_method
        self.cropped_img_path = cropped_path
        self.image = None
        self.points = []  # Store clicked points
        self.fixed_width = int(640*sf_x)   # Fixed image width
        self.fixed_height = int(480*sf_y)  # Fixed image height

        # Layout
        self.layout = QVBoxLayout()
        self.frame = frame 
        
        # Image display area
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        # Button to confirm selection
        self.confirm_button = QPushButton("Confirm Selection", self)
        self.confirm_button.clicked.connect(self.confirm_selection)
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)
        # Load, resize, and display the image in the corresponding image frame
        self.load_image()


    ############ Load image ################
    def load_image(self):
        """
        Load, resize, and display an image in a QLabel.
    
        This method loads an image either from memory (if `insp_method != 2`) or from a file 
        (`insp_method == 2`). It ensures that the image is correctly converted to RGB format 
        for compatibility with PyQt, resizes it to a fixed width and height, and updates 
        the display.
    
        Args:
            None. The method relies on instance attributes such as `image_path`, 
            `insp_method`, `fixed_width`, `fixed_height`, and `image_label`.
    
        Effects:
            - Loads an image from memory or disk.
            - Converts the image from BGR to RGB format for proper display.
            - Resizes the image to a fixed size `(fixed_width, fixed_height)`.
            - Stores a backup copy of the image in `self.image_backup`.
            - Updates the QLabel display with the processed image.
            - Enables mouse event handling for interaction.
    
        Notes:
            - If the image fails to load, an error message is printed.
            - The method uses OpenCV (`cv2`) for image processing.
            - The `update_display()` method is called to refresh the UI.
            - The `mousePressEvent` handler is set for interaction.
        """
        if self.insp_method != 2:
            self.image = cv2.cvtColor(self.image_path, cv2.COLOR_BGR2RGB)
        else:
            self.image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert to RGB for PyQt

        if self.image is None:
            print("Error: Unable to load image.")
            return

        self.image_original = self.image.copy()
        self.original_shape = self.image_original.shape
        
        # Resize image to 480x640
        self.image = cv2.resize(self.image, (self.fixed_width, self.fixed_height))
        
        self.image_backup = self.image.copy()
        
        # Convert OpenCV image to QPixmap and display it in QLabel.
        self.update_display()

        # Enable mouse event handler
        self.image_label.mousePressEvent = self.mouse_click_event


    ############ Update display image ################
    def update_display(self):
        """
        Convert an OpenCV image to QPixmap and display it in a QLabel.
    
        This method takes an image stored in OpenCV format (`self.image`), converts it 
        into a `QPixmap`, and updates the QLabel display. It ensures that the QLabel 
        maintains a fixed size and properly scales the displayed image.
    
        Args:
            None. The method relies on instance attributes such as `image`, `image_label`, 
            `fixed_width`, and `fixed_height`.
    
        Effects:
            - Updates `self.image_label` with the new pixmap.
            - Ensures `image_label` has a fixed size (`fixed_width`, `fixed_height`).
            - Enables `setScaledContents(True)` to allow proper image scaling.
    
        Notes:
            - The input image (`self.image`) must be in RGB format (`Format_RGB888`).
        """
        # Extract image dimensions and number of channels
        height, width, channel = self.image.shape  
        # Calculate the number of bytes per line in the image
        bytes_per_line = 3 * width  # Assuming an RGB image (3 channels per pixel)
        # Convert the NumPy image array to a QImage format
        q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Convert the QImage to a QPixmap, which can be displayed in a QLabel
        pixmap = QPixmap.fromImage(q_img)
        # Set the generated pixmap as the content of the QLabel
        self.image_label.setPixmap(pixmap)
        # Fix the QLabel size to a predetermined width and height
        self.image_label.setFixedSize(self.fixed_width, self.fixed_height)  
        # Enable automatic scaling of the image to fit within the QLabel dimensions
        self.image_label.setScaledContents(True)


    ############ Click for manual bounding box ################
    def mouse_click_event(self, event):
        """
        Handle mouse click events to capture four points and draw a bounding box.
    
        This method records user clicks on an image displayed in a QLabel, scales 
        the coordinates to match the fixed image dimensions (480x640), and stores 
        the clicked points. Once four points are selected, it orders them correctly 
        and draws a dashed bounding box.
    
        Args:
            event (QMouseEvent): The mouse click event containing the position of the click.
    
        Effects:
            - Captures and scales click coordinates to fit the resized image.
            - Stores up to four points in `self.points`.
            - Draws green circles at clicked points.
            - Once four points are selected:
                - Sorts the points into a proper order using `sort_points()`.
                - Draws dashed lines between the points using `draw_dashed_line()`.
                - Updates the displayed image.
    
        Notes:
            - The function scales the clicked coordinates based on the QLabel dimensions.
            - A maximum of four points are collected to define a bounding box.
            - `update_display()` is called to refresh the QLabel after drawing.
            - The bounding box follows the order:
                - Top-left to top-right
                - Top-right to bottom-right
                - Bottom-right to bottom-left
                - Bottom-left to top-left
        """
        if len(self.points) < 4:
            # Scale click coordinates to fit the 480x640 image
            # Scale "x" position in the original size into its correspoding for 640
            x = int(event.pos().x() * (self.fixed_width / self.image_label.width()))
            # Scale "y" position in the original size into its correspoding for 480
            y = int(event.pos().y() * (self.fixed_height / self.image_label.height()))
            
            # Collect the scaled points
            self.points.append((x, y))

            # Draw point on image
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            # Display the image with a green point at the clicked location
            self.update_display()

        if len(self.points) == 4:
            # Ensure the points are ordered correctly
            self.points = self.sort_points(self.points)

            # Draw dashed lines
            self.draw_dashed_line(self.points[0], self.points[1])  # Top-left to top-right
            self.draw_dashed_line(self.points[1], self.points[3])  # Top-right to bottom-right
            self.draw_dashed_line(self.points[3], self.points[2])  # Bottom-right to bottom-left
            self.draw_dashed_line(self.points[2], self.points[0])  # Bottom-left to top-left

            self.update_display()


    ############ Sort Points ################
    def sort_points(self, points):
        """
        Sort four points into a consistent order: top-left, top-right, bottom-left, bottom-right.
    
        This method takes a list of four (x, y) coordinate points and sorts them to ensure 
        they follow a specific order for defining a bounding box.
    
        Args:
            points (list of tuples): A list of four (x, y) coordinates.
    
        Returns:
            list: A sorted list of four points in the order:
                  [top-left, top-right, bottom-left, bottom-right].
    
        Effects:
            - Sorts points first by y-coordinate to separate top and bottom pairs.
            - Within each pair, sorts by x-coordinate to determine left and right positions.
    
        Notes:
            - This method assumes exactly four points are provided.
            - The sorting approach ensures consistency in defining a bounding box.
        """
        points = sorted(points, key=lambda p: (p[1], p[0]))  # Sort by y first, then x

        # Now we have two top points and two bottom points
        top_points = sorted(points[:2], key=lambda p: p[0])   # Leftmost is top-left, rightmost is top-right
        bottom_points = sorted(points[2:], key=lambda p: p[0]) # Leftmost is bottom-left, rightmost is bottom-right

        return [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]


    ############ Draw bounding box ################
    def draw_dashed_line(self, pt1, pt2, color=(255, 0, 0), thickness=3, dash_length=10, gap_length=10):
        """
        Draw a dashed line between two points on an image.
    
        This method calculates the Euclidean distance between two points (`pt1` and `pt2`), 
        determines the number of dashes that can fit along the line, and iteratively 
        draws each dash while maintaining the specified gap length.
    
        Args:
            pt1 (tuple): The starting point of the dashed line (x, y).
            pt2 (tuple): The ending point of the dashed line (x, y).
            color (tuple, optional): The color of the dashed line in BGR format. Default is red `(255, 0, 0)`.
            thickness (int, optional): The thickness of the dashed line. Default is `3`.
            dash_length (int, optional): The length of each dash in pixels. Default is `10`.
            gap_length (int, optional): The gap between each dash in pixels. Default is `10`.
    
        Effects:
            - Draws a dashed line on `self.image` using OpenCV's `cv2.line()` function.
    
        Notes:
            - The number of dashes is computed based on the total Euclidean distance between `pt1` and `pt2`.
            - The function interpolates start and end points for each dash based on the line's total length.
        """
        # Compute the Euclidean distance between two points (pt1 and pt2)
        dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5  
        # Calculate the number of dashes that can fit along the line
        num_dashes = int(dist / (dash_length + gap_length))  
        # Iterate through the number of dashes to draw them
        for i in range(num_dashes):
            # Compute the start point of the current dash
            start_x = int(pt1[0] + (pt2[0] - pt1[0]) * ((i * (dash_length + gap_length)) / dist))
            start_y = int(pt1[1] + (pt2[1] - pt1[1]) * ((i * (dash_length + gap_length)) / dist))
            # Compute the end point of the current dash
            end_x = int(pt1[0] + (pt2[0] - pt1[0]) * (((i * (dash_length + gap_length)) + dash_length) / dist))
            end_y = int(pt1[1] + (pt2[1] - pt1[1]) * (((i * (dash_length + gap_length)) + dash_length) / dist))
            # Draw the dash on the image using OpenCV
            cv2.line(self.image, (start_x, start_y), (end_x, end_y), color, thickness)



    ############ Crop image ################
    def crop_image(self):
        """
        Crop the selected bounding box from the image and save it if necessary.
    
        This method extracts a region from the image based on four user-selected points, 
        applies a perspective transformation, and returns the cropped image. If the 
        inspection method requires local saving, the cropped image is saved to a specified path.
    
        Returns:
            np.ndarray or None: The cropped image as a NumPy array, or `None` if fewer than 
                                four points are selected.
    
        Effects:
            - Checks if exactly four points are selected; otherwise, prints an error.
            - Sorts the points into a consistent order using `sort_points()`.
            - Computes the average crop width and height based on the selected points.
            - Applies a perspective transformation to extract the cropped region.
            - If `insp_method != 2`, stores the cropped image in `self.prediction_img`.
            - If `insp_method == 2`, saves the cropped image to `self.cropped_img_path`.
    
        Notes:
            - The method assumes that `self.points` contains four (x, y) coordinates.
            - The transformation ensures that the cropped image is properly aligned.
            - The cropped image is saved only when using a local inspection method (`insp_method == 2`).
        """
        # Ensure exactly 4 points are provided for cropping
        if len(self.points) != 4:
            print("Error: Exactly 4 points are required to crop the image.")
            return None
        
        # Sort the provided points to maintain the correct order for perspective transformation
        sorted_pts = self.sort_points(self.points)        
        # Compute the average width of the cropped region using the top and bottom edge distances
        crop_width = int(((sorted_pts[1][0] - sorted_pts[0][0]) + (sorted_pts[3][0] - sorted_pts[2][0])) / 2)
        # Compute the average height of the cropped region using the left and right edge distances
        crop_height = int(((sorted_pts[2][1] - sorted_pts[0][1]) + (sorted_pts[3][1] - sorted_pts[1][1])) / 2)
        
        # Define destination points for the perspective transformation
        dst_pts = np.array([
            [0, 0],                          # Top-left corner
            [crop_width - 1, 0],             # Top-right corner
            [0, crop_height - 1],            # Bottom-left corner
            [crop_width - 1, crop_height - 1] # Bottom-right corner
        ], dtype=np.float32)
        
        # Compute the perspective transformation matrix from input points to destination points
        matrix = cv2.getPerspectiveTransform(np.array(sorted_pts, dtype=np.float32), dst_pts)
        # Apply the perspective transformation to obtain the cropped image
        cropped_image = cv2.warpPerspective(self.image_backup, matrix, (crop_width, crop_height))
        
    
        # Expanded to the original size
        # Scale points back to the original image resolution
        scale_x = self.original_shape[1] / self.fixed_width  # Scale factor in x (width) direction
        scale_y = self.original_shape[0] / self.fixed_height  # Scale factor in y (height) direction

        # Apply the scaling factors to the sorted points to map them back to original resolution
        sorted_pts_original_size = np.array([
            [pt[0] * scale_x, pt[1] * scale_y] for pt in sorted_pts  # Scale each point individually
        ], dtype=np.float32)

        # Estimate the width of the cropped region in the original image
        # Using average of top and bottom horizontal edges (between point 0-1 and point 2-3)
        crop_width_original = int(((sorted_pts_original_size[1][0] - sorted_pts_original_size[0][0]) + 
                                   (sorted_pts_original_size[3][0] - sorted_pts_original_size[2][0])) / 2)

        # Estimate the height of the cropped region in the original image
        # Using average of left and right vertical edges (between point 0-2 and point 1-3)
        crop_height_original = int(((sorted_pts_original_size[2][1] - sorted_pts_original_size[0][1]) + 
                                    (sorted_pts_original_size[3][1] - sorted_pts_original_size[1][1])) / 2)

        # Define the destination points for the perspective transformation
        # This is a rectangle of the estimated crop size, starting from top-left [0,0]
        dst_pts_original = np.array([
            [0, 0],  # top-left
            [crop_width_original - 1, 0],  # top-right
            [0, crop_height_original - 1],  # bottom-left
            [crop_width_original - 1, crop_height_original - 1]  # bottom-right
        ], dtype=np.float32)

        # Compute the perspective transformation matrix from the original points to the destination rectangle
        matrix_org = cv2.getPerspectiveTransform(sorted_pts_original_size, dst_pts_original)

        # Apply the perspective warp to the original image to get the rectified and cropped region
        cropped_image_original = cv2.warpPerspective(self.image_original, matrix_org, (crop_width_original, crop_height_original))

        # Save the cropped image as an RGB JPEG file (convert to BGR format first for OpenCV compatibility)
        cv2.imwrite(self.cropped_img_path, cv2.cvtColor(cropped_image_original, cv2.COLOR_RGB2BGR))

       
        # Check the inspection method condition to determine whether to save or assign the image
        if self.insp_method != 2:
            # Store the cropped image for further processing
            self.prediction_img = cropped_image
        else:
            if self.gui_methods.box_id == None:
                # Save the cropped image to the specified path
                save_path = self.cropped_img_path
                cv2.imwrite(save_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR before saving
            else:
                self.prediction_img = cropped_image
                
        
        # Return the cropped image
        return cropped_image



    ############ Confirm button method ################
    def confirm_selection(self):
        """
        Save the modified image and close the dialog after confirming the selection.
    
        This method verifies that exactly four points have been selected before proceeding 
        with cropping the image. It then updates the displayed image in the UI and 
        closes the selection dialog.
    
        Effects:
            - Checks if exactly four points have been selected; otherwise, prints an error message.
            - Calls `crop_image()` to extract the selected region.
            - Converts the modified image from RGB format to a `QImage`.
            - Converts the `QImage` to a `QPixmap` and updates `self.frame` in the UI.
            - Uses `Qt.SmoothTransformation` to ensure high-quality image scaling.
            - Closes the dialog using `self.accept()`.
    
        Notes:
            - The method assumes `self.image` is in RGB format.
            - The user must select exactly four points before confirming.
            - The image update process ensures the modified selection is displayed in the UI.
        """
        if len(self.points) == 4:
            
            image_bgr = self.image 
            self.crop_image()
            # Convert the RGB image to QImage
            height, width, channel = image_bgr.shape
            bytes_per_line = 3 * width
            qimage = QImage(image_bgr.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert QImage to QPixmap
            building_pixmap = QPixmap.fromImage(qimage)
            
            # Set the pixmap to QLabel
            self.frame.setPixmap(
                building_pixmap.scaled(
                    self.frame.width(),
                    self.frame.height(),
                    Qt.IgnoreAspectRatio,  # Adjust scaling mode as needed
                    Qt.SmoothTransformation))  # Ensure high-quality scaling
        else:
            print("You must select exactly 4 points before confirming.")
            
        self.accept()
