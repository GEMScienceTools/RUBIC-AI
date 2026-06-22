"""
bounding_box_manual.py
======================
This module provides a PyQt5-based interactive dialog for manually selecting a bounding box
region on an image through a four-point perspective crop workflow.
"""
import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class BoundingBoxWindow(QDialog):
    def __init__(self, image_path, frame, insp_method, path_cropped, parent=None, main_window=None, gui_methods=None):
        super().__init__(parent)
        self.main_window = main_window
        self.gui_methods = gui_methods
        
        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        #
        DESIGN_WIDTH = 1920
        DESIGN_HEIGHT = 1080
        
        # Scale the GUI based on resolution
        sf_x = screen_width / DESIGN_WIDTH
        sf_y = screen_height / DESIGN_HEIGHT
        sf_factor = np.sqrt(sf_x * sf_y)

        # DPI-based scale
        # Get a reliable DPI value
        if sys.platform.startswith("win"):
            # Windows: use ctypes to get real DPI
            import ctypes
            LOGPIXELSX = 88
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
            ctypes.windll.user32.ReleaseDC(0, hdc)
        else:
            # macOS / Linux: start with logical DPI
            dpi = screen.logicalDotsPerInch()
            # If logical DPI looks weird, fallback to physical
            if dpi < 60 or dpi > 200:
                dpi = screen.physicalDotsPerInch()
        
        # For geometry: mainly resolution-based
        sf_x = sf_factor
        sf_y = sf_factor

        self.setWindowTitle("Manual Bounding Box Selection")
        self.setGeometry(int(100*sf_x), int(100*sf_y), int(700*sf_x), int(600*sf_y))
        
        self.image_path = image_path
        self.insp_method = insp_method
        self.cropped_img_path = path_cropped
        self.image = None
        self.points = []
        self.fixed_width = int(640*sf_x)
        self.fixed_height = int(480*sf_y)

        # Layout
        self.layout = QVBoxLayout()
        self.frame = frame 
        
        # Image display area
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        # Buttons
        self.reset_button = QPushButton("Reset Points", self)
        self.reset_button.clicked.connect(self.reset_points)
        self.layout.addWidget(self.reset_button)
        
        self.confirm_button = QPushButton("Confirm Selection", self)
        self.confirm_button.clicked.connect(self.confirm_selection)
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)
        self.load_image()

    # ================= LOAD IMAGE =================
    def load_image(self):
        """
        Loads the input image according to the inspection method, converts it to RGB format, 
        resizes it for display, stores backup copies, and updates the image viewer in the interface.
        """
        
        if self.insp_method != 2:
            if self.gui_methods.color_control:
                self.gui_methods.color_control = False
                self.image = self.image_path.copy()
            else:
                self.image = cv2.cvtColor(self.image_path, cv2.COLOR_RGB2BGR)
        else:
            self.image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        if self.image is None:
            print("Error: Unable to load image.")
            return

        self.image_original = self.image.copy()
        self.original_shape = self.image_original.shape
        self.image = cv2.resize(self.image, (self.fixed_width, self.fixed_height))
        self.image_backup = self.image.copy()

        self.update_display()
        self.image_label.mousePressEvent = self.mouse_click_event

    # ================= UPDATE DISPLAY =================
    def update_display(self):
        """
        Converts the current image into a Qt-compatible format and updates the image label 
        to display it with the predefined size settings.
        """
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(self.fixed_width, self.fixed_height)
        self.image_label.setScaledContents(True)

    # ================= RESET POINTS =================
    def reset_points(self):
        """Clear all selected points and restore the original image."""
        self.points = []
        self.image = self.image_backup.copy()
        self.update_display()

    # ================= MOUSE CLICK EVENT =================
    def mouse_click_event(self, event):
        """
        Records up to four user-selected points on the image, draws markers for each click, 
        and connects the points with dashed lines once the selection is complete.
        """
        if len(self.points) < 4:
            x = int(event.pos().x() * (self.fixed_width / self.image_label.width()))
            y = int(event.pos().y() * (self.fixed_height / self.image_label.height()))
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            self.update_display()

        if len(self.points) == 4:
            self.points = self.sort_points(self.points)
            self.draw_dashed_line(self.points[0], self.points[1])
            self.draw_dashed_line(self.points[1], self.points[3])
            self.draw_dashed_line(self.points[3], self.points[2])
            self.draw_dashed_line(self.points[2], self.points[0])
            
            displayed_path = self.main_window.output_folder_value + "/Mapillary/displayed_images/"+str(self.gui_methods.click_count+1)+".jpg"
            cv2.imwrite(displayed_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            
            self.update_display()

    # ================= SORT POINTS =================
    def sort_points(self, points):
        """
        Sorts four selected points into a consistent order: top-left, top-right, 
        bottom-left, and bottom-right.
        """
        points = sorted(points, key=lambda p: (p[1], p[0]))
        top_points = sorted(points[:2], key=lambda p: p[0])
        bottom_points = sorted(points[2:], key=lambda p: p[0])
        return [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]

    # ================= DRAW DASHED LINE =================
    def draw_dashed_line(self, pt1, pt2, color=(255, 0, 0), thickness=3, dash_length=10, gap_length=10):
        """
        Draws a dashed line between two points on the image using the specified style settings.
        """
        dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
        num_dashes = int(dist / (dash_length + gap_length))
        for i in range(num_dashes):
            start_x = int(pt1[0] + (pt2[0] - pt1[0]) * ((i * (dash_length + gap_length)) / dist))
            start_y = int(pt1[1] + (pt2[1] - pt1[1]) * ((i * (dash_length + gap_length)) / dist))
            end_x = int(pt1[0] + (pt2[0] - pt1[0]) * (((i * (dash_length + gap_length)) + dash_length) / dist))
            end_y = int(pt1[1] + (pt2[1] - pt1[1]) * (((i * (dash_length + gap_length)) + dash_length) / dist))
            cv2.line(self.image, (start_x, start_y), (end_x, end_y), color, thickness)

    # ================= CROP IMAGE =================
    def crop_image(self):
        """
        Crops the selected image region using a perspective transformation based on four user-defined 
        points, generates both display-size and original-resolution cropped outputs, and saves or stores 
        the result depending on the inspection method.
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
        
        # Check the inspection method condition to determine whether to save or assign the image
        if self.insp_method != 2:
            # Store the cropped image for further processing
            self.prediction_img = cropped_image
            
            # Mapillary images
            if self.main_window.img_source == 2:
                save_path = self.main_window.output_folder_value + "/Mapillary/Cropped_images/"+str(self.gui_methods.click_count+1)+".jpg"
                cv2.imwrite(save_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))  
        else:
            if self.gui_methods.box_id is None:
                # Save the cropped image to the specified path
                save_path = self.cropped_img_path
                cv2.imwrite(save_path, cv2.cvtColor(cropped_image_original, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR before saving
            else:
                self.prediction_img = cropped_image
        
        # Return the cropped image
        return cropped_image

    # ================= CONFIRM SELECTION =================
    def confirm_selection(self):
        """
        Confirms the selected four-point region, crops the image, updates the preview frame 
        with the selected result, and closes the dialog.
        """
        if len(self.points) == 4:
            image_bgr = self.image
            self.crop_image()
            height, width, channel = image_bgr.shape
            bytes_per_line = 3 * width
            qimage = QImage(image_bgr.data, width, height, bytes_per_line, QImage.Format_RGB888)
            building_pixmap = QPixmap.fromImage(qimage)
            self.frame.setPixmap(
                building_pixmap.scaled(
                    self.frame.width(),
                    self.frame.height(),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation))
            
            self.accept()
        else:
            QMessageBox.warning(self,"Selection Required","Please select exactly four points before confirming.")
        
