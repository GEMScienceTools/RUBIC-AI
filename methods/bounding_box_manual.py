import cv2
import numpy as np
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class BoundingBoxWindow(QDialog):
    def __init__(self, image_path, frame, insp_method, cropped_path, parent=None, main_window=None, gui_methods=None):
        super().__init__(parent)
        self.main_window = main_window
        self.gui_methods = gui_methods
        
        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080

        self.setWindowTitle("Manual Bounding Box Selection")
        self.setGeometry(int(100*sf_x), int(100*sf_y), int(700*sf_x), int(600*sf_y))
        
        self.image_path = image_path
        self.insp_method = insp_method
        self.cropped_img_path = cropped_path
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
        if self.insp_method != 2:
            self.image = cv2.cvtColor(self.image_path, cv2.COLOR_BGR2RGB)
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
            self.update_display()

    # ================= SORT POINTS =================
    def sort_points(self, points):
        points = sorted(points, key=lambda p: (p[1], p[0]))
        top_points = sorted(points[:2], key=lambda p: p[0])
        bottom_points = sorted(points[2:], key=lambda p: p[0])
        return [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]

    # ================= DRAW DASHED LINE =================
    def draw_dashed_line(self, pt1, pt2, color=(255, 0, 0), thickness=3, dash_length=10, gap_length=10):
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
        if len(self.points) != 4:
            print("Error: Exactly 4 points are required to crop the image.")
            return None

        sorted_pts = self.sort_points(self.points)
        crop_width = int(((sorted_pts[1][0] - sorted_pts[0][0]) + (sorted_pts[3][0] - sorted_pts[2][0])) / 2)
        crop_height = int(((sorted_pts[2][1] - sorted_pts[0][1]) + (sorted_pts[3][1] - sorted_pts[1][1])) / 2)

        dst_pts = np.array([
            [0, 0],
            [crop_width - 1, 0],
            [0, crop_height - 1],
            [crop_width - 1, crop_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(np.array(sorted_pts, dtype=np.float32), dst_pts)
        cropped_image = cv2.warpPerspective(self.image_backup, matrix, (crop_width, crop_height))
        cv2.imwrite(self.cropped_img_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        return cropped_image

    # ================= CONFIRM SELECTION =================
    def confirm_selection(self):
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
        
