"""PyQt5 manual image-blurring dialog, adapted from bounding_box_manual.py
and blur_manual.py. Run directly with: python manual_blur_window.py in.jpg out.jpg
"""

import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout,
)


class ManualBlurWindow(QDialog):
    """Drag rectangles to blur, preview, undo/reset, and save full-resolution output.

    Inputs match BoundingBoxWindow's integration pattern. ``image_path`` may
    be a path or a NumPy array. NumPy arrays are assumed BGR unless
    ``gui_methods.color_control`` is True (then already RGB), matching the
    original BoundingBoxWindow convention.
    """

    def __init__(
        self, 
        image_path, 
        frame=None, 
        insp_method=2, 
        path_blurred=None,
        parent=None, 
        main_window=None, 
        gui_methods=None, 
        blur_kernel=51,
    ):
        super().__init__(parent)
        self.image_path = image_path
        self.frame = frame
        self.insp_method = insp_method
        self.blurred_img_path = os.fspath(path_blurred) if path_blurred else None
        self.main_window = main_window
        self.gui_methods = gui_methods
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer.")
        self.blur_kernel = int(blur_kernel)

        self.rects_display = []
        self.current_rect = None
        self.start_pt = None
        self.drawing = False
        self.blurred_image = None
        self.rects_original = []

        # Use the 1920 x 1080 baseline from BoundingBoxWindow.
        screen = QApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("Create a QApplication before ManualBlurWindow.")
        available = screen.availableGeometry()
        sf_x = available.width() / 1920
        sf_y = available.height() / 1080
        sf_factor = np.sqrt(sf_x * sf_y)
        self.max_width = max(1, min(int(1100 * sf_factor),
                                    int(available.width() * 0.90)))
        self.max_height = max(1, min(int(750 * sf_factor),
                                     int(available.height() * 0.78)))

        self.setWindowTitle("Manual Image Blurring")
        self.setModal(True)
        self.layout = QVBoxLayout()

        instruction = QLabel(
            "Drag to blur  |  U: Undo  |  R: Reset  |  "
            "S/Enter: Save  |  Q/Esc: Cancel", self,
        )
        instruction.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(instruction)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setCursor(Qt.CrossCursor)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        buttons = QHBoxLayout()
        self.undo_button = QPushButton("Undo Last", self)
        self.undo_button.clicked.connect(self.undo_last_rectangle)
        buttons.addWidget(self.undo_button)

        self.reset_button = QPushButton("Reset All", self)
        self.reset_button.clicked.connect(self.reset_points)
        buttons.addWidget(self.reset_button)

        self.confirm_button = QPushButton("Save Blurred Image", self)
        self.confirm_button.clicked.connect(self.confirm_selection)
        self.confirm_button.setDefault(True)
        buttons.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        self.layout.addLayout(buttons)
        self.setLayout(self.layout)

        self.load_image()
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event

        self.adjustSize()
        self.move(
            available.x() + (available.width() - self.width()) // 2,
            available.y() + (available.height() - self.height()) // 2,
        )

    # ================= LOAD IMAGE =================
    def load_image(self):
        """Accept a filename or NumPy array and prepare an RGB display image."""
        if isinstance(self.image_path, (str, os.PathLike)):
            source = cv2.imread(os.fspath(self.image_path), cv2.IMREAD_COLOR)
            if source is None:
                raise FileNotFoundError(f"Unable to read image: {self.image_path}")
            self.image_original = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        elif isinstance(self.image_path, np.ndarray):
            source = self.image_path.copy()
            if source.ndim == 2:
                self.image_original = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
            elif source.ndim == 3 and source.shape[2] == 4:
                self.image_original = cv2.cvtColor(source, cv2.COLOR_BGRA2RGB)
            elif source.ndim == 3 and source.shape[2] == 3:
                if getattr(self.gui_methods, "color_control", False):
                    self.gui_methods.color_control = False
                    self.image_original = source  # Already RGB
                else:
                    self.image_original = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Unsupported NumPy image shape.")
        else:
            raise TypeError("image_path must be a filename or NumPy array.")

        self.original_shape = self.image_original.shape
        h, w = self.original_shape[:2]
        self.scale = min(self.max_width / w, self.max_height / h, 1.0)
        self.fixed_width = max(1, int(round(w * self.scale)))
        self.fixed_height = max(1, int(round(h * self.scale)))
        self.image_backup = cv2.resize(
            self.image_original, (self.fixed_width, self.fixed_height),
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR,
        )
        self.image_label.setFixedSize(self.fixed_width, self.fixed_height)
        self.update_display()

    # ================= COORDINATE CONVERSION =================
    @staticmethod
    def clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def to_original_coords(self, rect):
        """Map display coordinates to original pixels (right/bottom exclusive)."""
        x1, y1, x2, y2 = rect
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        h, w = self.original_shape[:2]
        sx, sy = w / self.fixed_width, h / self.fixed_height
        return (
            self.clamp(int(np.floor(left * sx)), 0, w),
            self.clamp(int(np.floor(top * sy)), 0, h),
            self.clamp(int(np.ceil((right + 1) * sx)), 0, w),
            self.clamp(int(np.ceil((bottom + 1) * sy)), 0, h),
        )

    # ================= APPLY BLUR =================
    def apply_blur(self, image, rectangles):
        """Blur rectangular ROIs using the source script's Gaussian kernel."""
        result = image.copy()
        h, w = result.shape[:2]
        for x1, y1, x2, y2 in rectangles:
            left, right = sorted((self.clamp(x1, 0, w), self.clamp(x2, 0, w)))
            top, bottom = sorted((self.clamp(y1, 0, h), self.clamp(y2, 0, h)))
            roi = result[top:bottom, left:right]
            if roi.size:
                result[top:bottom, left:right] = cv2.GaussianBlur(
                    roi, (self.blur_kernel, self.blur_kernel), 0,
                )
        return result

    # ================= UPDATE DISPLAY =================
    def update_display(self):
        """Recompute the preview after a rectangle is added, undone or reset."""
        original_rects = [self.to_original_coords(r) for r in self.rects_display]
        if original_rects:
            preview = self.apply_blur(self.image_original, original_rects)
            self.preview_image = cv2.resize(
                preview, (self.fixed_width, self.fixed_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            self.preview_image = self.image_backup.copy()
        self.draw_display()

    def draw_display(self):
        """Show already selected rectangles (yellow) and active drag (green)."""
        self.image = self.preview_image.copy()
        for x1, y1, x2, y2 in self.rects_display:
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 200, 0), 2)
        if self.current_rect is not None:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb = np.ascontiguousarray(self.image)
        h, w = rgb.shape[:2]
        q_img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img.copy()))

    # ================= MOUSE EVENTS =================
    def mouse_position(self, event):
        return (
            self.clamp(event.pos().x(), 0, self.fixed_width - 1),
            self.clamp(event.pos().y(), 0, self.fixed_height - 1),
        )

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pt = self.mouse_position(event)
            x, y = self.start_pt
            self.current_rect = (x, y, x, y)
            self.draw_display()
            event.accept()

    def mouse_move_event(self, event):
        if self.drawing:
            self.current_rect = (*self.start_pt, *self.mouse_position(event))
            self.draw_display()
            event.accept()

    def mouse_release_event(self, event):
        if not self.drawing or event.button() != Qt.LeftButton:
            return
        x, y = self.mouse_position(event)
        x1, y1 = self.start_pt
        if abs(x - x1) > 3 and abs(y - y1) > 3:
            self.rects_display.append((x1, y1, x, y))
        self.current_rect = None
        self.start_pt = None
        self.drawing = False
        self.update_display()
        event.accept()

    # ================= UNDO / RESET =================
    def undo_last_rectangle(self):
        if self.rects_display:
            self.rects_display.pop()
        self.update_display()

    def reset_points(self):
        self.rects_display.clear()
        self.current_rect = None
        self.start_pt = None
        self.drawing = False
        self.update_display()

    # ================= SAVE / CONFIRM =================
    def confirm_selection(self):
        if not self.rects_display:
            QMessageBox.warning(self, "Selection Required", "Select a region to blur.")
            return
        if not self.blurred_img_path:
            QMessageBox.warning(self, "Missing Output Path", "Specify path_blurred.")
            return

        rects = [self.to_original_coords(r) for r in self.rects_display]
        result_rgb = self.apply_blur(self.image_original, rects)
        destination = os.path.abspath(self.blurred_img_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(destination, result_bgr)

        self.blurred_image = result_rgb  # RGB array at original resolution
        self.rects_original = rects
        self.blurred_img_path = destination
        if self.frame is not None and hasattr(self.frame, "setPixmap"):
            rgb = np.ascontiguousarray(result_rgb)
            h, w = rgb.shape[:2]
            q_img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img.copy())
            self.frame.setPixmap(
                pixmap.scaled(self.frame.width(), self.frame.height(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self.accept()

    # ================= KEYBOARD SHORTCUTS =================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_U:
            self.undo_last_rectangle()
        elif event.key() == Qt.Key_R:
            self.reset_points()
        elif event.key() in (Qt.Key_S, Qt.Key_Return, Qt.Key_Enter):
            self.confirm_selection()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.reject()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python manual_blur_window.py input.jpg output.jpg")
        sys.exit(2)
    app = QApplication(sys.argv)
    dialog = ManualBlurWindow(sys.argv[1], insp_method=2,
                              path_blurred=sys.argv[2])
    sys.exit(0 if dialog.exec_() == QDialog.Accepted else 1)
