import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class BoundingBoxWindow(QDialog):
    """
    Dialog window for manual bounding box selection on an image.

    This class provides a PyQt5 dialog interface that allows users to manually
    select bounding box coordinates on an image by clicking points. It handles
    image loading, display scaling based on screen resolution and DPI, and
    coordinates the bounding box selection workflow.

    Parameters
    ----------
    image_path : str or ndarray
        Path to the image file or numpy array containing image data.
    frame : object
        Frame object containing additional context or data for the inspection.
    insp_method : int
        Inspection method identifier that determines image processing behavior.
        Different values trigger different image format conversion routines.
    path_cropped : str
        File path for storing the cropped/processed image output.
    parent : QWidget, optional
        Parent widget for this dialog (default is None).
    main_window : QMainWindow, optional
        Reference to the main application window (default is None).
    gui_methods : object, optional
        Object containing GUI utility methods and configuration flags
        (default is None).

    Attributes
    ----------
    main_window : QMainWindow
        Reference to the parent main window.
    gui_methods : object
        GUI methods utility object.
    image : ndarray
        Current image being displayed (scaled to display size).
    image_original : ndarray
        Original unscaled image data.
    image_backup : ndarray
        Backup copy of the scaled image.
    original_shape : tuple
        Shape (height, width, channels) of the original image.
    points : list
        List of (x, y) coordinates selected by the user via mouse clicks.
    fixed_width : int
        Display width of the image in pixels, scaled based on screen resolution.
    fixed_height : int
        Display height of the image in pixels, scaled based on screen resolution.
    image_path : str
        Path to the source image file.
    insp_method : int
        Inspection method identifier.
    cropped_img_path : str
        Path for output of cropped image.
    image_label : QLabel
        Label widget displaying the image.
    reset_button : QPushButton
        Button to clear all selected bounding box points.
    confirm_button : QPushButton
        Button to confirm and save the bounding box selection.
    layout : QVBoxLayout
        Main layout container for dialog widgets.

    Notes
    -----
    - Window geometry and scaling are automatically calculated based on the
      primary screen resolution and DPI. Scaling factors are computed from
      the ratio of current resolution to a design baseline of 1920x1080.
    - DPI detection is platform-specific: Windows uses ctypes to retrieve
      actual DPI, while macOS/Linux use logical DPI with fallback to
      physical DPI if out of expected range (60-200).
    - Image processing differs based on `insp_method` value. When insp_method
      is not 2, RGB conversion may be skipped if color_control flag is set.

    See Also
    --------
    load_image : Loads and prepares the image for display.
    update_display: Adjust the image size for display.
    reset_points : Clears all user-selected bounding box points.
    mouse_click_event: Records up to four user-selected points on the image.
    sort_points:Sorts four selected points into a consistent order.
    draw_dashed_line: Draws a dashed line for manual bounding box.
    crop_image: Crops the target building image.
    confirm_selection : Validates and confirms the bounding box selection.

    Examples
    --------
    >>> window = BoundingBoxWindow(
    ...     image_path="path/to/image.jpg",
    ...     frame=frame_data,
    ...     insp_method=1,
    ...     path_cropped="path/to/output.jpg",
    ...     parent=main_window,
    ... )
    >>> window.show()
    """

    def __init__(
        self,
        image_path,
        frame,
        insp_method,
        path_cropped,
        parent=None,
        main_window=None,
        gui_methods=None,
    ):
        super().__init__(parent)
        self.main_window = main_window
        self.gui_methods = gui_methods

        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        #
        design_width = 1920
        design_height = 1080

        # Scale the GUI based on resolution
        sf_x = screen_width / design_width
        sf_y = screen_height / design_height
        sf_factor = np.sqrt(sf_x * sf_y)

        # DPI-based scale
        # Get a reliable DPI value
        if sys.platform.startswith("win"):
            # Windows: use ctypes to get real DPI
            import ctypes

            logpixelsx = 88
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, logpixelsx)
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
        self.setGeometry(
            int(100 * sf_x), int(100 * sf_y), int(700 * sf_x), int(600 * sf_y)
        )

        self.image_path = image_path
        self.insp_method = insp_method
        self.cropped_img_path = path_cropped
        self.image = None
        self.points = []
        self.fixed_width = int(640 * sf_x)
        self.fixed_height = int(480 * sf_y)

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
        """Load and prepare the input image.

        Convert the image according to the inspection method, resize it for
        display, store backup copies, and update the image viewer.
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
        """Update the image label with a Qt-compatible image.

        Convert the current image to a Qt-compatible format and display it
        using the predefined dimensions.
        """
        height, width, _channel = self.image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            self.image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
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
        """Record a point selected by the user.

        Draw a marker for each click and connect the four selected points with
        dashed lines once the selection is complete.
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

            displayed_path = (
                self.main_window.output_folder_value
                + "/Mapillary/displayed_images/"
                + str(self.gui_methods.click_count + 1)
                + ".jpg"
            )
            cv2.imwrite(displayed_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

            self.update_display()

    # ================= SORT POINTS =================
    def sort_points(self, points):
        """Sort four points into a consistent corner order.

        Return the points as top-left, top-right, bottom-left, and
        bottom-right.
        """
        points = sorted(points, key=lambda p: (p[1], p[0]))
        top_points = sorted(points[:2], key=lambda p: p[0])
        bottom_points = sorted(points[2:], key=lambda p: p[0])
        return [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]

    # ================= DRAW DASHED LINE =================
    def draw_dashed_line(
        self, pt1, pt2, color=(255, 0, 0), thickness=3, dash_length=10, gap_length=10
    ):
        """Draw a dashed line between two points on the image."""
        dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
        num_dashes = int(dist / (dash_length + gap_length))
        for i in range(num_dashes):
            start_x = int(
                pt1[0] + (pt2[0] - pt1[0]) * ((i * (dash_length + gap_length)) / dist)
            )
            start_y = int(
                pt1[1] + (pt2[1] - pt1[1]) * ((i * (dash_length + gap_length)) / dist)
            )
            end_x = int(
                pt1[0]
                + (pt2[0] - pt1[0])
                * (((i * (dash_length + gap_length)) + dash_length) / dist)
            )
            end_y = int(
                pt1[1]
                + (pt2[1] - pt1[1])
                * (((i * (dash_length + gap_length)) + dash_length) / dist)
            )
            cv2.line(self.image, (start_x, start_y), (end_x, end_y), color, thickness)

    # ================= CROP IMAGE =================
    def crop_image(self):
        """Crop the region defined by four selected points.

        Apply a perspective transformation at display and original resolution,
        then save or store the result according to the inspection method.
        """
        # Ensure exactly 4 points are provided for cropping
        if len(self.points) != 4:
            print("Error: Exactly 4 points are required to crop the image.")
            return None

        # Sort the points into the order required for perspective transformation.
        sorted_pts = self.sort_points(self.points)
        # Average the top and bottom edge lengths to estimate the crop width.
        crop_width = int(
            (
                (sorted_pts[1][0] - sorted_pts[0][0])
                + (sorted_pts[3][0] - sorted_pts[2][0])
            )
            / 2
        )
        # Average the left and right edge lengths to estimate the crop height.
        crop_height = int(
            (
                (sorted_pts[2][1] - sorted_pts[0][1])
                + (sorted_pts[3][1] - sorted_pts[1][1])
            )
            / 2
        )

        # Define destination points for the perspective transformation
        dst_pts = np.array(
            [
                [0, 0],  # Top-left corner
                [crop_width - 1, 0],  # Top-right corner
                [0, crop_height - 1],  # Bottom-left corner
                [crop_width - 1, crop_height - 1],  # Bottom-right corner
            ],
            dtype=np.float32,
        )

        # Compute the perspective transformation matrix.
        matrix = cv2.getPerspectiveTransform(
            np.array(sorted_pts, dtype=np.float32), dst_pts
        )
        # Apply the perspective transformation to obtain the cropped image
        cropped_image = cv2.warpPerspective(
            self.image_backup, matrix, (crop_width, crop_height)
        )

        # Expanded to the original size
        # Scale points back to the original image resolution
        scale_x = (
            self.original_shape[1] / self.fixed_width
        )  # Scale factor in x (width) direction
        scale_y = (
            self.original_shape[0] / self.fixed_height
        )  # Scale factor in y (height) direction

        # Map the sorted points back to the original image resolution.
        sorted_pts_original_size = np.array(
            [
                [pt[0] * scale_x, pt[1] * scale_y]
                for pt in sorted_pts  # Scale each point individually
            ],
            dtype=np.float32,
        )

        # Estimate the width of the cropped region in the original image
        # Average the top and bottom horizontal edges.
        crop_width_original = int(
            (
                (sorted_pts_original_size[1][0] - sorted_pts_original_size[0][0])
                + (sorted_pts_original_size[3][0] - sorted_pts_original_size[2][0])
            )
            / 2
        )

        # Estimate the height of the cropped region in the original image
        # Average the left and right vertical edges.
        crop_height_original = int(
            (
                (sorted_pts_original_size[2][1] - sorted_pts_original_size[0][1])
                + (sorted_pts_original_size[3][1] - sorted_pts_original_size[1][1])
            )
            / 2
        )

        # Define the destination points for the perspective transformation
        # This is a rectangle of the estimated crop size, starting from top-left [0,0]
        dst_pts_original = np.array(
            [
                [0, 0],  # top-left
                [crop_width_original - 1, 0],  # top-right
                [0, crop_height_original - 1],  # bottom-left
                [crop_width_original - 1, crop_height_original - 1],  # bottom-right
            ],
            dtype=np.float32,
        )

        # Compute the transformation matrix at the original resolution.
        matrix_org = cv2.getPerspectiveTransform(
            sorted_pts_original_size, dst_pts_original
        )

        # Warp the original image to obtain the rectified crop.
        cropped_image_original = cv2.warpPerspective(
            self.image_original, matrix_org, (crop_width_original, crop_height_original)
        )

        # Save or retain the crop according to the inspection method.
        if self.insp_method != 2:
            # Store the cropped image for further processing
            self.prediction_img = cropped_image

            # Mapillary images
            if self.main_window.img_source == 2:
                save_path = (
                    self.main_window.output_folder_value
                    + "/Mapillary/Cropped_images/"
                    + str(self.gui_methods.click_count + 1)
                    + ".jpg"
                )
                cv2.imwrite(save_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        else:
            if self.gui_methods.box_id is None:
                # Save the cropped image to the specified path
                save_path = self.cropped_img_path
                cv2.imwrite(
                    save_path, cv2.cvtColor(cropped_image_original, cv2.COLOR_RGB2BGR)
                )  # Convert RGB to BGR before saving
            else:
                self.prediction_img = cropped_image

        # Return the cropped image
        return cropped_image

    # ================= CONFIRM SELECTION =================
    def confirm_selection(self):
        """Confirm the selected region and close the dialog.

        Crop the four-point region and update the preview frame with the
        selected result.
        """
        if len(self.points) == 4:
            image_bgr = self.image
            self.crop_image()
            height, width, _channel = image_bgr.shape
            bytes_per_line = 3 * width
            qimage = QImage(
                image_bgr.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            building_pixmap = QPixmap.fromImage(qimage)
            self.frame.setPixmap(
                building_pixmap.scaled(
                    self.frame.width(),
                    self.frame.height(),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

            self.accept()
        else:
            QMessageBox.warning(
                self,
                "Selection Required",
                "Please select exactly four points before confirming.",
            )
