"""Provide the dialog for configuring the specific-coordinates workflow."""

import ctypes
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from methods.utilities import save_coordinates, select_output_folder, upload_csv

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 120.0
DEFAULT_DPI = 96.0
MIN_REASONABLE_DPI = 60.0
MAX_REASONABLE_DPI = 200.0
LOGPIXELSX = 88


class SpecificLocationSetting(QtWidgets.QDialog):
    """Configure inputs for the specific-coordinates inspection workflow."""

    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method

        sf_x, sf_y, self.sf_font = self._get_scale_factors()
        self._configure_window(sf_x, sf_y)
        self._create_widgets(sf_x, sf_y)
        self._raise_widgets()

    @staticmethod
    def _get_windows_dpi():
        """Return the effective Windows DPI value."""
        hdc = ctypes.windll.user32.GetDC(0)
        try:
            return ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
        finally:
            ctypes.windll.user32.ReleaseDC(0, hdc)

    @classmethod
    def _get_scale_factors(cls):
        """Calculate geometry and font scale factors for the active screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return 1.0, 1.0, 1.0

        geometry = screen.geometry()
        resolution_x = geometry.width() / DESIGN_WIDTH
        resolution_y = geometry.height() / DESIGN_HEIGHT
        scale_factor = float(np.sqrt(resolution_x * resolution_y))

        if sys.platform.startswith("win"):
            dpi = cls._get_windows_dpi()
        else:
            dpi = screen.logicalDotsPerInch()
            if not MIN_REASONABLE_DPI <= dpi <= MAX_REASONABLE_DPI:
                dpi = screen.physicalDotsPerInch()

        if dpi <= 0:
            dpi = DEFAULT_DPI

        font_scale = scale_factor * (DESIGN_DPI / dpi)
        return scale_factor, scale_factor, font_scale

    def _configure_window(self, sf_x, sf_y):
        """Configure the dialog title, icon, and dimensions."""
        self.setWindowTitle("Specific Coordinates Method Input")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(643 * sf_x), int(533 * sf_y))
        self.coord_frame = QtWidgets.QWidget(self)

    def _create_widgets(self, sf_x, sf_y):
        """Create and configure all dialog widgets."""
        normal_font = self._font(10)
        bold_font = self._font(10, bold=True)
        title_font = self._font(
            10,
            bold=True,
            italic=True,
            underline=True,
        )

        self.backg_4 = QtWidgets.QLabel(self.coord_frame)
        self._set_geometry(self.backg_4, (10, 9, 621, 471), sf_x, sf_y)
        self.backg_4.setStyleSheet("background-color: rgb(212, 206, 255);")

        self.specific_label = QtWidgets.QLabel(self.coord_frame)
        self._set_geometry(self.specific_label, (190, 9, 301, 21), sf_x, sf_y)
        self.specific_label.setFont(title_font)
        self.specific_label.setText("Specific Coordinates Method Input")

        self.output_label_specific = QtWidgets.QLabel(self.coord_frame)
        self._set_geometry(
            self.output_label_specific,
            (20, 39, 121, 31),
            sf_x,
            sf_y,
        )
        self.output_label_specific.setFont(bold_font)
        self.output_label_specific.setText("Output name:")

        self.output_specific = QtWidgets.QLineEdit(self.coord_frame)
        self._set_geometry(self.output_specific, (160, 39, 271, 31), sf_x, sf_y)
        self.output_specific.setFont(normal_font)
        self.output_specific.setText("specific_coord")

        self.path_out_folder_bt = QtWidgets.QPushButton(self.coord_frame)
        self._set_geometry(
            self.path_out_folder_bt,
            (20, 80, 231, 31),
            sf_x,
            sf_y,
        )
        self.path_out_folder_bt.setFont(normal_font)
        self.path_out_folder_bt.setText("Select output folder")
        self.path_out_folder_bt.clicked.connect(self._on_select_output_folder)

        self.output_folder_value = QtWidgets.QLabel(self.coord_frame)
        self._set_geometry(
            self.output_folder_value,
            (270, 85, 341, 21),
            sf_x,
            sf_y,
        )
        self.output_folder_value.setFont(normal_font)
        self.output_folder_value.setText("path/where/you/want/to/save/your/results")

        self.csv_button_specific = QtWidgets.QPushButton(self.coord_frame)
        self._set_geometry(
            self.csv_button_specific,
            (20, 125, 231, 31),
            sf_x,
            sf_y,
        )
        self.csv_button_specific.setFont(normal_font)
        self.csv_button_specific.setText("Upload building coordinates")
        self.csv_button_specific.clicked.connect(self._on_upload_csv)

        self.specific_path = QtWidgets.QLabel(self.coord_frame)
        self._set_geometry(self.specific_path, (270, 130, 321, 21), sf_x, sf_y)
        self.specific_path.setFont(normal_font)
        self.specific_path.setText("filename.csv")

        self.feature_collection_label = QtWidgets.QLabel(self.coord_frame)
        self._set_geometry(
            self.feature_collection_label,
            (20, 170, 221, 31),
            sf_x,
            sf_y,
        )
        self.feature_collection_label.setFont(bold_font)
        self.feature_collection_label.setObjectName("feature_collection_label")
        self.feature_collection_label.setText("Feature collection mode:")

        self.collection_mode = QtWidgets.QComboBox(self.coord_frame)
        self._set_geometry(self.collection_mode, (250, 170, 191, 31), sf_x, sf_y)
        self.collection_mode.setFont(normal_font)
        self.collection_mode.setObjectName("collection_mode")
        self.collection_mode.addItems(["Manual", "AI Powered"])

        self.img_source_label = QtWidgets.QLabel(self.coord_frame)
        self._set_geometry(self.img_source_label, (20, 220, 131, 31), sf_x, sf_y)
        self.img_source_label.setFont(bold_font)
        self.img_source_label.setObjectName("img_source_label")
        self.img_source_label.setText("Image source:")

        self.img_source_mode = QtWidgets.QComboBox(self.coord_frame)
        self._set_geometry(self.img_source_mode, (150, 220, 191, 31), sf_x, sf_y)
        self.img_source_mode.setFont(normal_font)
        self.img_source_mode.setObjectName("img_source_mode")
        self.img_source_mode.addItem("Google Street View", 1)
        self.img_source_mode.addItem("Mapillary", 2)

        self.tableWidget = QtWidgets.QTableWidget(self.coord_frame)
        self._set_geometry(self.tableWidget, (20, 270, 601, 192), sf_x, sf_y)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self._set_geometry(self.save_button, (220, 490, 191, 31), sf_x, sf_y)
        self.save_button.setFont(bold_font)
        self.save_button.setText("Save and Continue")
        self.save_button.clicked.connect(self._on_save_coordinates)

    def _font(self, point_size, *, bold=False, italic=False, underline=False):
        """Create a font scaled for the active display."""
        font = QtGui.QFont()
        font.setPointSize(max(1, int(point_size * self.sf_font)))
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)
        return font

    @staticmethod
    def _set_geometry(widget, geometry, sf_x, sf_y):
        """Apply scaled geometry to a widget."""
        x, y, width, height = geometry
        widget.setGeometry(
            QtCore.QRect(
                int(x * sf_x),
                int(y * sf_y),
                int(width * sf_x),
                int(height * sf_y),
            )
        )

    def _raise_widgets(self):
        """Apply the intended widget stacking order."""
        widgets = (
            self.backg_4,
            self.specific_label,
            self.output_label_specific,
            self.output_specific,
            self.path_out_folder_bt,
            self.output_folder_value,
            self.csv_button_specific,
            self.specific_path,
            self.save_button,
            self.tableWidget,
            self.collection_mode,
            self.feature_collection_label,
            self.img_source_label,
            self.img_source_mode,
        )
        for widget in widgets:
            widget.raise_()

    def _on_select_output_folder(self):
        """Select and display the output folder."""
        output_folder, self.display_folder = select_output_folder(self)
        self.method.output_folder_value = output_folder
        self.output_folder_value.setText(self.display_folder)

    def _on_upload_csv(self):
        """Load and preview the selected coordinates CSV file."""
        upload_csv(self)

    def _on_save_coordinates(self):
        """Validate and save the uploaded coordinates."""
        save_coordinates(self)
