"""Provide a dialog for configuring Google Street View image parameters."""

import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 120.0
MIN_REASONABLE_DPI = 60
MAX_REASONABLE_DPI = 200


class GSVAngleSetting(QtWidgets.QDialog):
    """Configure pitch, heading, and field of view for Street View images.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget of the dialog.
    main_window : object, optional
        Reference to the main application window.
    gui_methods : object, optional
        Reference to the GUI helper object.

    Attributes
    ----------
    pitch_value : QSpinBox
        Spin box containing the vertical camera angle.
    heading_value : QSpinBox
        Spin box containing the horizontal camera angle.
    fov_value : QSpinBox
        Spin box containing the field-of-view value.
    """

    def __init__(self, parent=None, main_window=None, gui_methods=None):
        super().__init__(parent)
        self.main_window = main_window
        self.gui_methods = gui_methods

        scale_x, scale_y, font_scale = self._calculate_scale_factors()

        self.setObjectName("GSV_angle_setting")
        self.resize(int(680 * scale_x), int(900 * scale_y))
        self.setWindowTitle("Virtual image angle settings")

        self._create_interface(scale_x, scale_y, font_scale)

    @staticmethod
    def _get_screen_dpi(screen):
        """Return a reliable screen DPI value."""
        if sys.platform.startswith("win"):
            import ctypes

            log_pixels_x = 88
            device_context = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(
                device_context,
                log_pixels_x,
            )
            ctypes.windll.user32.ReleaseDC(0, device_context)
            return dpi

        dpi = screen.logicalDotsPerInch()
        if not MIN_REASONABLE_DPI <= dpi <= MAX_REASONABLE_DPI:
            dpi = screen.physicalDotsPerInch()
        return dpi

    @classmethod
    def _calculate_scale_factors(cls):
        """Calculate geometry and font scale factors for the current screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()

        width_scale = screen_geometry.width() / DESIGN_WIDTH
        height_scale = screen_geometry.height() / DESIGN_HEIGHT
        geometry_scale = np.sqrt(width_scale * height_scale)

        dpi = cls._get_screen_dpi(screen)
        dpi_scale = DESIGN_DPI / dpi
        font_scale = geometry_scale * dpi_scale

        return geometry_scale, geometry_scale, font_scale

    @staticmethod
    def _make_font(
        point_size,
        font_scale,
        *,
        bold=False,
        italic=False,
        underline=False,
    ):
        """Create a scaled Qt font with the requested style."""
        font = QtGui.QFont()
        font.setPointSize(int(point_size * font_scale))
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)
        return font

    @staticmethod
    def _set_geometry(widget, x, y, width, height, scale_x, scale_y):
        """Assign scaled geometry to a widget."""
        widget.setGeometry(
            QtCore.QRect(
                int(x * scale_x),
                int(y * scale_y),
                int(width * scale_x),
                int(height * scale_y),
            )
        )

    def _create_interface(self, scale_x, scale_y, font_scale):
        """Create and arrange all widgets in the dialog."""
        self.gsv_angle_frame = QtWidgets.QWidget(self)
        self.gsv_angle_frame.setObjectName("gsv_angle_frame")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.gsv_angle_frame)

        self._create_background(scale_x, scale_y)
        self._create_title(scale_x, scale_y, font_scale)
        self._create_pitch_controls(scale_x, scale_y, font_scale)
        self._create_heading_controls(scale_x, scale_y, font_scale)
        self._create_fov_controls(scale_x, scale_y, font_scale)
        self._create_save_button(scale_x, scale_y, font_scale)

    def _create_background(self, scale_x, scale_y):
        """Create the dialog background panel."""
        self.backg_1 = QtWidgets.QLabel(self.gsv_angle_frame)
        self._set_geometry(
            self.backg_1,
            10,
            10,
            631,
            811,
            scale_x,
            scale_y,
        )
        self.backg_1.setStyleSheet("background-color: rgb(255, 215, 255);")
        self.backg_1.setObjectName("backg_1")

    def _create_title(self, scale_x, scale_y, font_scale):
        """Create the dialog title label."""
        self.gsv_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self._set_geometry(
            self.gsv_label,
            200,
            10,
            270,
            30,
            scale_x,
            scale_y,
        )
        self.gsv_label.setFont(
            self._make_font(
                12,
                font_scale,
                bold=True,
                italic=True,
                underline=True,
            )
        )
        self.gsv_label.setObjectName("gsv_label")
        self.gsv_label.setText("Virtual image parameters")

    def _create_pitch_controls(self, scale_x, scale_y, font_scale):
        """Create controls and guidance for the pitch angle."""
        self.pitch_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self._set_geometry(
            self.pitch_label,
            20,
            39,
            51,
            31,
            scale_x,
            scale_y,
        )
        self.pitch_label.setFont(
            self._make_font(10, font_scale, bold=True, underline=True)
        )
        self.pitch_label.setObjectName("pitch_label")
        self.pitch_label.setText("Pitch:")

        self.pitch_value = QtWidgets.QSpinBox(self.gsv_angle_frame)
        self._set_geometry(
            self.pitch_value,
            80,
            40,
            51,
            31,
            scale_x,
            scale_y,
        )
        self.pitch_value.setFont(self._make_font(10, font_scale))
        self.pitch_value.setRange(0, 60)
        self.pitch_value.setValue(5)
        self.pitch_value.setObjectName("pitch_value")

        self.pitch_descrip = QtWidgets.QTextBrowser(self.gsv_angle_frame)
        self._set_geometry(
            self.pitch_descrip,
            20,
            80,
            611,
            81,
            scale_x,
            scale_y,
        )
        self.pitch_descrip.setObjectName("pitch_descrip")
        self.pitch_descrip.setHtml(
            self._description_html(
                int(10 * font_scale),
                "pitch angle",
                "vertical inclination of the camera",
                (
                    "allowing a full view of tall buildings. This angle is "
                    "limited between <b>0° and 60°</b>, with <b>5°</b> as "
                    "the default value."
                ),
            )
        )

        self.pitch_img = QtWidgets.QLabel(self.gsv_angle_frame)
        self._set_geometry(
            self.pitch_img,
            20,
            170,
            611,
            361,
            scale_x,
            scale_y,
        )
        self.pitch_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pitch_img.setObjectName("pitch_img")
        self.pitch_img.setPixmap(QtGui.QPixmap("help_img/pitch_angle.png"))
        self.pitch_img.setScaledContents(True)

    def _create_heading_controls(self, scale_x, scale_y, font_scale):
        """Create controls and guidance for the heading angle."""
        self.heading_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self._set_geometry(
            self.heading_label,
            20,
            539,
            81,
            31,
            scale_x,
            scale_y,
        )
        self.heading_label.setFont(
            self._make_font(10, font_scale, bold=True, underline=True)
        )
        self.heading_label.setObjectName("heading_label")
        self.heading_label.setText("Heading:")

        self.heading_value = QtWidgets.QSpinBox(self.gsv_angle_frame)
        self._set_geometry(
            self.heading_value,
            110,
            540,
            61,
            31,
            scale_x,
            scale_y,
        )
        self.heading_value.setFont(self._make_font(10, font_scale))
        self.heading_value.setRange(-180, 180)
        self.heading_value.setValue(0)
        self.heading_value.setObjectName("heading_value")

        self.heading_descrip = QtWidgets.QTextBrowser(self.gsv_angle_frame)
        self._set_geometry(
            self.heading_descrip,
            20,
            580,
            611,
            81,
            scale_x,
            scale_y,
        )
        self.heading_descrip.setObjectName("heading_descrip")
        self.heading_descrip.setHtml(
            self._description_html(
                int(10 * font_scale),
                "horizontal camera angle",
                "direction of the camera",
                (
                    "including views of buildings on the opposite side of "
                    "the street. This angle is limited between <b>-180° and "
                    "180°</b>, with <b>0°</b> as the default value."
                ),
            )
        )

    def _create_fov_controls(self, scale_x, scale_y, font_scale):
        """Create controls and guidance for the field of view."""
        self.fov_label = QtWidgets.QLabel(self.gsv_angle_frame)
        self._set_geometry(
            self.fov_label,
            20,
            670,
            81,
            31,
            scale_x,
            scale_y,
        )
        self.fov_label.setFont(
            self._make_font(10, font_scale, bold=True, underline=True)
        )
        self.fov_label.setObjectName("fov_label")
        self.fov_label.setText("FOV:")

        self.fov_value = QtWidgets.QSpinBox(self.gsv_angle_frame)
        self._set_geometry(
            self.fov_value,
            70,
            670,
            61,
            31,
            scale_x,
            scale_y,
        )
        self.fov_value.setFont(self._make_font(10, font_scale))
        self.fov_value.setRange(1, 120)
        self.fov_value.setValue(120)
        self.fov_value.setObjectName("fov_value")

        self.fov_descrip = QtWidgets.QTextBrowser(self.gsv_angle_frame)
        self._set_geometry(
            self.fov_descrip,
            20,
            710,
            611,
            101,
            scale_x,
            scale_y,
        )
        self.fov_descrip.setObjectName("fov_descrip")
        self.fov_descrip.setHtml(
            self._description_html(
                int(10 * font_scale),
                "field of view",
                "camera zoom",
                (
                    "where smaller values zoom in and larger values zoom "
                    "out. The default value is <b>120</b>, which is the "
                    "maximum allowed value."
                ),
            )
        )

    def _create_save_button(self, scale_x, scale_y, font_scale):
        """Create the button that saves the selected parameters."""
        self.save_button_gsv = QtWidgets.QPushButton(self.gsv_angle_frame)
        self._set_geometry(
            self.save_button_gsv,
            240,
            830,
            191,
            31,
            scale_x,
            scale_y,
        )
        self.save_button_gsv.setFont(self._make_font(10, font_scale, bold=True))
        self.save_button_gsv.setObjectName("save_button_gsv")
        self.save_button_gsv.setText("Save and continue")
        self.save_button_gsv.clicked.connect(self.save_and_close)

    @staticmethod
    def _description_html(font_size, subject, definition, details):
        """Return reusable HTML for a parameter-description panel."""
        return f"""
        <html>
          <head>
            <style>p {{ white-space: pre-wrap; }}</style>
          </head>
          <body style="font-family:'MS Shell Dlg 2'; font-size:{font_size}pt;">
            <p align="justify">
              The <b>{subject}</b> controls the <b>{definition}</b>, {details}
            </p>
          </body>
        </html>
        """

    def save_and_close(self):
        """Accept the selected parameters and close the dialog."""
        self.accept()


# Backward-compatible alias for existing imports in RUBIC-AI.
gsv_angle_setting = GSVAngleSetting
