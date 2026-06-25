"""Provide the dialog used to configure the local-images workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from methods.utilities import save_coordinates, select_output_folder, upload_csv

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 120.0
WINDOW_WIDTH = 611
WINDOW_HEIGHT = 512
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
UNSUPPORTED_IMAGE_EXTENSIONS = (".heic",)


class LocalImageSetting(QtWidgets.QDialog):
    """Configure input data for the local-images inspection method.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget of the dialog.
    method : object, optional
        Object that stores the selected paths and settings, typically an
        instance of ``GUIMethods``.
    """

    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method

        sf_x, sf_y, self.sf_font = self._calculate_scale_factors()
        self.setWindowTitle("Local Images Method Input")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(WINDOW_WIDTH * sf_x), int(WINDOW_HEIGHT * sf_y))

        self.coord_frame = QtWidgets.QWidget(self)
        self._create_widgets(sf_x, sf_y)
        self._set_stacking_order()

    @staticmethod
    def _calculate_scale_factors() -> tuple[float, float, float]:
        """Calculate geometry and font scale factors for the active screen.

        Returns
        -------
        tuple[float, float, float]
            Horizontal, vertical, and font scale factors.
        """
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return 1.0, 1.0, 1.0

        screen_geometry = screen.geometry()
        sf_x = screen_geometry.width() / DESIGN_WIDTH
        sf_y = screen_geometry.height() / DESIGN_HEIGHT
        sf_factor = float(np.sqrt(sf_x * sf_y))

        dpi = LocalImageSetting._get_screen_dpi(screen)
        scale_dpi = DESIGN_DPI / dpi if dpi > 0 else 1.0
        return sf_factor, sf_factor, sf_factor * scale_dpi

    @staticmethod
    def _get_screen_dpi(screen) -> float:
        """Return a reliable screen DPI value for the current platform."""
        if sys.platform.startswith("win"):
            import ctypes

            logpixels_x = 88
            hdc = ctypes.windll.user32.GetDC(0)
            try:
                return float(ctypes.windll.gdi32.GetDeviceCaps(hdc, logpixels_x))
            finally:
                ctypes.windll.user32.ReleaseDC(0, hdc)

        dpi = float(screen.logicalDotsPerInch())
        if not 60 <= dpi <= 200:
            dpi = float(screen.physicalDotsPerInch())
        return dpi

    @staticmethod
    def _scaled_rect(
        x: int,
        y: int,
        width: int,
        height: int,
        sf_x: float,
        sf_y: float,
    ) -> QtCore.QRect:
        """Create a rectangle scaled to the current screen dimensions."""
        return QtCore.QRect(
            int(x * sf_x),
            int(y * sf_y),
            int(width * sf_x),
            int(height * sf_y),
        )

    def _make_font(
        self,
        point_size: int = 10,
        *,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
    ) -> QtGui.QFont:
        """Create a font scaled for the current display."""
        font = QtGui.QFont()
        font.setPointSize(max(1, int(point_size * self.sf_font)))
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)
        return font

    def _create_widgets(self, sf_x: float, sf_y: float) -> None:
        """Create and configure all widgets in the dialog."""
        self.backg_4 = QtWidgets.QLabel(self.coord_frame)
        self.backg_4.setGeometry(self._scaled_rect(10, 9, 591, 451, sf_x, sf_y))
        self.backg_4.setStyleSheet("background-color: rgb(255, 253, 187);")

        regular_font = self._make_font()
        bold_font = self._make_font(bold=True)

        self.local_label = QtWidgets.QLabel(self.coord_frame)
        self.local_label.setGeometry(self._scaled_rect(190, 9, 251, 21, sf_x, sf_y))
        self.local_label.setFont(
            self._make_font(bold=True, italic=True, underline=True)
        )
        self.local_label.setText("Local Images Method Input")

        self.output_label_local = QtWidgets.QLabel(self.coord_frame)
        self.output_label_local.setGeometry(
            self._scaled_rect(20, 39, 121, 31, sf_x, sf_y)
        )
        self.output_label_local.setFont(bold_font)
        self.output_label_local.setText("Output name:")

        self.output_local = QtWidgets.QLineEdit(self.coord_frame)
        self.output_local.setGeometry(self._scaled_rect(160, 39, 270, 31, sf_x, sf_y))
        self.output_local.setFont(regular_font)
        self.output_local.setText("Local_images")

        self.path_out_folder_bt = QtWidgets.QPushButton(self.coord_frame)
        self.path_out_folder_bt.setGeometry(
            self._scaled_rect(20, 80, 231, 31, sf_x, sf_y)
        )
        self.path_out_folder_bt.setFont(regular_font)
        self.path_out_folder_bt.setText("Select output folder")
        self.path_out_folder_bt.clicked.connect(self._on_select_output_folder)

        self.output_folder_value = QtWidgets.QLabel(self.coord_frame)
        self.output_folder_value.setGeometry(
            self._scaled_rect(270, 82, 291, 21, sf_x, sf_y)
        )
        self.output_folder_value.setFont(regular_font)
        self.output_folder_value.setText("path/where/will/save/your/results")

        self.folder_local_button = QtWidgets.QPushButton(self.coord_frame)
        self.folder_local_button.setGeometry(
            self._scaled_rect(20, 125, 231, 31, sf_x, sf_y)
        )
        self.folder_local_button.setFont(regular_font)
        self.folder_local_button.setText("Select image folder")
        self.folder_local_button.clicked.connect(self.select_folder)

        self.local_folder_path = QtWidgets.QLabel(self.coord_frame)
        self.local_folder_path.setGeometry(
            self._scaled_rect(270, 130, 291, 21, sf_x, sf_y)
        )
        self.local_folder_path.setFont(regular_font)
        self.local_folder_path.setText("---")

        self.csv_button_local = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_local.setGeometry(
            self._scaled_rect(20, 165, 231, 31, sf_x, sf_y)
        )
        self.csv_button_local.setFont(regular_font)
        self.csv_button_local.setText("Upload building information")
        self.csv_button_local.clicked.connect(self._on_upload_csv)

        self.local_path = QtWidgets.QLabel(self.coord_frame)
        self.local_path.setGeometry(self._scaled_rect(270, 170, 291, 21, sf_x, sf_y))
        self.local_path.setFont(regular_font)
        self.local_path.setText("filename.csv")

        self.feature_collection_label = QtWidgets.QLabel(self.coord_frame)
        self.feature_collection_label.setGeometry(
            self._scaled_rect(20, 210, 221, 31, sf_x, sf_y)
        )
        self.feature_collection_label.setFont(bold_font)
        self.feature_collection_label.setObjectName("feature_collection_label")
        self.feature_collection_label.setText("Feature collection mode:")

        self.collection_mode = QtWidgets.QComboBox(self.coord_frame)
        self.collection_mode.setGeometry(
            self._scaled_rect(250, 210, 191, 31, sf_x, sf_y)
        )
        self.collection_mode.setFont(regular_font)
        self.collection_mode.setObjectName("collection_mode")
        self.collection_mode.addItems(["Manual", "AI Powered"])

        # Keep this Qt Designer-style name for compatibility with utilities.py.
        self.tableWidget = QtWidgets.QTableWidget(self.coord_frame)
        self.tableWidget.setGeometry(self._scaled_rect(20, 250, 571, 192, sf_x, sf_y))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(self._scaled_rect(220, 470, 191, 31, sf_x, sf_y))
        self.save_button.setFont(bold_font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self._on_save_coordinates)

    def _set_stacking_order(self) -> None:
        """Raise the interactive widgets above the background label."""
        widgets = (
            self.backg_4,
            self.local_label,
            self.csv_button_local,
            self.local_path,
            self.output_label_local,
            self.output_local,
            self.folder_local_button,
            self.local_folder_path,
            self.save_button,
            self.path_out_folder_bt,
            self.output_folder_value,
            self.tableWidget,
            self.collection_mode,
            self.feature_collection_label,
        )
        for widget in widgets:
            widget.raise_()

    def _on_select_output_folder(self) -> None:
        """Select and display the folder used to save workflow outputs."""
        output_folder, display_folder = select_output_folder(self)
        self.display_folder = display_folder

        if self.method is not None:
            self.method.output_folder_value = output_folder

        self.output_folder_value.setText(display_folder)

    def select_folder(self) -> None:
        """Select an image folder and report unsupported image formats."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
        )
        if not folder_path:
            return

        if self.method is not None:
            self.method.folder_path = folder_path

        folder = Path(folder_path)
        self.local_folder_path.setText(folder.name)

        files = [path for path in folder.iterdir() if path.is_file()]
        supported_images = [
            path for path in files if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        unsupported_images = [
            path.name
            for path in files
            if path.suffix.lower() in UNSUPPORTED_IMAGE_EXTENSIONS
        ]

        if not supported_images:
            QtWidgets.QMessageBox.warning(
                self,
                "No Valid Images",
                (
                    "The selected folder does not contain any valid image "
                    "files (.jpg, .jpeg, or .png)."
                ),
            )
        elif unsupported_images:
            QtWidgets.QMessageBox.information(
                self,
                "Unsupported Formats Found",
                (
                    "The folder contains unsupported HEIC images:\n\n"
                    + "\n".join(unsupported_images)
                ),
            )

    def _on_upload_csv(self) -> None:
        """Load, validate, and display the selected building-information CSV."""
        upload_csv(self)

    def _on_save_coordinates(self) -> None:
        """Validate the coordinate data and save it as a GeoPackage file."""
        save_coordinates(self)
