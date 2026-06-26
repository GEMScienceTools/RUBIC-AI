"""Provide the dialog used to select a RUBIC-AI inspection workflow."""

from __future__ import annotations

import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QMessageBox

from methods.extrapolation_options import ExtrapolationOptions
from methods.local_images_method import LocalImageSetting
from methods.polygon_method import PolygonSetting
from methods.specific_coordinates_method import SpecificLocationSetting

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 120.0
WINDOW_WIDTH = 890
WINDOW_HEIGHT = 800
LOGPIXELS_X = 88


def _scaled_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
) -> QtCore.QRect:
    """Return a rectangle scaled for the current screen."""
    return QtCore.QRect(
        int(x * scale_x),
        int(y * scale_y),
        int(width * scale_x),
        int(height * scale_y),
    )


def _description_html(text: str, font_size: float) -> str:
    """Build HTML used by the inspection-method description widgets."""
    return f"""
    <html>
      <head>
        <meta name="qrichtext" content="1" />
        <style>p, li {{ white-space: pre-wrap; }}</style>
      </head>
      <body style="font-family:'MS Shell Dlg 2';
                   font-size:{font_size:.1f}pt;
                   font-weight:400;">
        <p align="justify" style="margin:0; text-indent:0;">
          <span>{text}</span>
        </p>
      </body>
    </html>
    """


def _screen_scales() -> tuple[float, float, float]:
    """Calculate geometry and font scales for the primary display."""
    screen = QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return 1.0, 1.0, 1.0

    geometry = screen.geometry()
    scale_x = geometry.width() / DESIGN_WIDTH
    scale_y = geometry.height() / DESIGN_HEIGHT
    scale_factor = float(np.sqrt(scale_x * scale_y))

    if sys.platform.startswith("win"):
        import ctypes

        hdc = ctypes.windll.user32.GetDC(0)
        try:
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELS_X)
        finally:
            ctypes.windll.user32.ReleaseDC(0, hdc)
    else:
        dpi = screen.logicalDotsPerInch()
        if not 60 <= dpi <= 200:
            dpi = screen.physicalDotsPerInch()

    if dpi <= 0:
        dpi = DESIGN_DPI

    font_scale = scale_factor * (DESIGN_DPI / dpi)
    return scale_factor, scale_factor, font_scale


class InspectionSetting(QDialog):
    """Display the available inspection workflows and collect the selection.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget of the dialog.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        scale_x, scale_y, font_scale = _screen_scales()

        self.setWindowTitle("Select Inspection Method")
        self.resize(
            int(WINDOW_WIDTH * scale_x),
            int(WINDOW_HEIGHT * scale_y),
        )
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))

        self.method_frame = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.method_frame)

        self._create_header(scale_x, scale_y, font_scale)
        self._create_polygon_section(scale_x, scale_y, font_scale)
        self._create_specific_section(scale_x, scale_y, font_scale)
        self._create_local_section(scale_x, scale_y, font_scale)
        self._create_extrapolation_section(scale_x, scale_y, font_scale)
        self._set_stacking_order()

    @staticmethod
    def _font(point_size: float, bold: bool = False) -> QtGui.QFont:
        """Create a font with the requested size and weight."""
        font = QtGui.QFont()
        font.setPointSize(max(1, int(point_size)))
        font.setBold(bold)
        return font

    def _create_header(
        self,
        scale_x: float,
        scale_y: float,
        font_scale: float,
    ) -> None:
        """Create the dialog title and confirmation button."""
        self.w_tittle = QtWidgets.QLabel(self.method_frame)
        self.w_tittle.setGeometry(_scaled_rect(300, 0, 301, 41, scale_x, scale_y))
        self.w_tittle.setFont(self._font(12 * font_scale, bold=True))
        self.w_tittle.setObjectName("w_tittle")
        self.w_tittle.setText("Setting Inspection Method")

        self.save_button = QtWidgets.QPushButton(self.method_frame)
        self.save_button.setGeometry(_scaled_rect(340, 760, 191, 31, scale_x, scale_y))
        self.save_button.setFont(self._font(10 * font_scale, bold=True))
        self.save_button.setObjectName("save_button")
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.select_method)

    def _create_section_background(
        self,
        object_name: str,
        geometry: tuple[int, int, int, int],
        color: str,
        scale_x: float,
        scale_y: float,
    ) -> QtWidgets.QLabel:
        """Create and return a colored section background."""
        label = QtWidgets.QLabel(self.method_frame)
        label.setGeometry(_scaled_rect(*geometry, scale_x, scale_y))
        label.setStyleSheet(f"background-color: {color};")
        label.setText("")
        label.setObjectName(object_name)
        return label

    def _create_method_checkbox(
        self,
        text: str,
        object_name: str,
        geometry: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
        font_scale: float,
    ) -> QtWidgets.QCheckBox:
        """Create and return a method-selection checkbox."""
        checkbox = QtWidgets.QCheckBox(self.method_frame)
        checkbox.setGeometry(_scaled_rect(*geometry, scale_x, scale_y))
        checkbox.setFont(self._font(10 * font_scale, bold=True))
        checkbox.setObjectName(object_name)
        checkbox.setText(text)
        return checkbox

    def _create_image(
        self,
        path: str,
        object_name: str,
        geometry: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> QtWidgets.QLabel:
        """Create and return a scaled reference image label."""
        label = QtWidgets.QLabel(self.method_frame)
        label.setGeometry(_scaled_rect(*geometry, scale_x, scale_y))
        label.setObjectName(object_name)
        label.setPixmap(QtGui.QPixmap(path))
        label.setScaledContents(True)
        return label

    def _create_description(
        self,
        text: str,
        object_name: str,
        geometry: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
        font_scale: float,
    ) -> QtWidgets.QTextEdit:
        """Create a scaled, justified description text box."""
        description = QtWidgets.QTextEdit(self)
        description.setObjectName(object_name)
    
        x, y, width, height = geometry
        description.setGeometry(
            int(x * scale_x),
            int(y * scale_y),
            int(width * scale_x),
            int(height * scale_y),
        )
    
        description.setReadOnly(True)
        description.setFrameShape(QtWidgets.QFrame.StyledPanel)
        description.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        description.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
    
        # Apply the resolution-dependent font scale exactly once.
        font = QtGui.QFont()
        font.setPointSizeF(10 * font_scale)
    
        description.setFont(font)
        description.document().setDefaultFont(font)
        description.document().setDocumentMargin(4)
    
        # Add the content after configuring the font.
        description.setPlainText(text)
    
        cursor = description.textCursor()
        cursor.select(QtGui.QTextCursor.Document)
    
        block_format = QtGui.QTextBlockFormat()
        block_format.setAlignment(QtCore.Qt.AlignJustify)
        block_format.setTextIndent(0)
        block_format.setLeftMargin(0)
        block_format.setRightMargin(0)
        block_format.setTopMargin(0)
        block_format.setBottomMargin(0)
    
        cursor.mergeBlockFormat(block_format)
    
        character_format = QtGui.QTextCharFormat()
        character_format.setFont(font)
        cursor.mergeCharFormat(character_format)
    
        cursor.clearSelection()
        cursor.movePosition(QtGui.QTextCursor.Start)
        description.setTextCursor(cursor)
    
        description.setContentsMargins(0, 0, 0, 0)
        description.setViewportMargins(0, 0, 0, 0)
    
        return description
    
    def _create_polygon_section(
        self,
        scale_x: float,
        scale_y: float,
        font_scale: float,
    ) -> None:
        """Create widgets for the polygon workflow."""
        self.backg_1 = self._create_section_background(
            "backg_1",
            (10, 40, 870, 171),
            "rgb(255, 224, 185)",
            scale_x,
            scale_y,
        )
    
        self.dafault_img = self._create_image(
            "help_img/default_buildings.png",
            "dafault_img",
            (680, 50, 181, 141),
            scale_x,
            scale_y,
        )
    
        text = (
            "Creates a polygon from uploaded vertex coordinates, provided in "
            "either clockwise or counterclockwise order, or from an existing "
            "polygon file in SHP or GPKG format. Virtual inspections are then "
            "performed using either the specified sample size or the entire "
            "building population within the polygon."
        )
    
        self.default_descrip = self._create_description(
            text,
            "default_descrip",
            (230, 50, 430, 140),
            scale_x,
            scale_y,
            font_scale,
        )
    
        self.default_check = self._create_method_checkbox(
            "Polygon method",
            "default_check",
            (20, 120, 171, 21),
            scale_x,
            scale_y,
            font_scale,
        )

    def _create_specific_section(
        self,
        scale_x: float,
        scale_y: float,
        font_scale: float,
    ) -> None:
        """Create widgets for the specific-coordinate workflow."""
        self.backg_2 = self._create_section_background(
            "backg_2",
            (10, 220, 870, 171),
            "rgb(215, 213, 255)",
            scale_x,
            scale_y,
        )
        text = (
            "Allows the user to upload specific building locations from one "
            "or more geographic areas. The input CSV file must contain the "
            "coordinate pairs of the buildings to inspect."
        )
        self.specific_descrip = self._create_description(
            text,
            "specific_descrip",
            (230, 230, 430, 140),
            scale_x,
            scale_y,
            font_scale,
        )
        self.specific_img = self._create_image(
            "help_img/specific_buildings.png",
            "specific_img",
            (680, 230, 181, 141),
            scale_x,
            scale_y,
        )
        self.specific_check = self._create_method_checkbox(
            "Specific coordinates",
            "specific_check",
            (20, 300, 201, 21),
            scale_x,
            scale_y,
            font_scale,
        )

    def _create_local_section(
        self,
        scale_x: float,
        scale_y: float,
        font_scale: float,
    ) -> None:
        """Create widgets for the local-image workflow."""
        self.backg_3 = self._create_section_background(
            "backg_3",
            (10, 400, 870, 201),
            "rgb(255, 253, 187)",
            scale_x,
            scale_y,
        )
        self.local_img = self._create_image(
            "help_img/local_buildings.png",
            "local_img",
            (680, 430, 181, 141),
            scale_x,
            scale_y,
        )
        text = (
            "In this mode, users can analyse local images captured using mobile "
            "phones, drones, or other devices. Users must provide a folder "
            "containing the building images and a CSV file with the corresponding "
            "image information. The CSV file must contain columns named 'id', "
            "'latitude', and 'longitude'. The 'id' value must match the complete "
            "image filename, including its extension (e.g., .png or .jpg). HEIC "
            "images are not supported."
        )
        self.local_descrip = self._create_description(
            text,
            "local_descrip",
            (230, 410, 430, 180),
            scale_x,
            scale_y,
            font_scale,
        )
        self.local_check = self._create_method_checkbox(
            "Local images",
            "local_check",
            (30, 480, 171, 21),
            scale_x,
            scale_y,
            font_scale,
        )

    def _create_extrapolation_section(
        self,
        scale_x: float,
        scale_y: float,
        font_scale: float,
    ) -> None:
        """Create widgets for the extrapolation workflow."""
        self.backg_4 = self._create_section_background(
            "backg_4",
            (10, 610, 870, 141),
            "rgb(157, 218, 255)",
            scale_x,
            scale_y,
        )
        self.extrapolation_check = self._create_method_checkbox(
            "Neighbor",
            "extrapolation_check",
            (30, 650, 221, 21),
            scale_x,
            scale_y,
            font_scale,
        )
        self.extra_label = QtWidgets.QLabel(self.method_frame)
        self.extra_label.setGeometry(_scaled_rect(50, 670, 121, 21, scale_x, scale_y))
        self.extra_label.setFont(self._font(10 * font_scale, bold=True))
        self.extra_label.setObjectName("extra_label")
        self.extra_label.setText("extrapolation")

        text = (
            "Requires locations for buildings without imagery and a reference "
            "dataset containing building attributes. The tool extrapolates "
            "the most probable attributes for the target buildings."
        )
        self.extrapolation_descrip = self._create_description(
            text,
            "extrapolation_descrip",
            (230, 620, 430, 121),
            scale_x,
            scale_y,
            font_scale,
        )
        self.extra_img = self._create_image(
            "help_img/extrapolation.png",
            "extra_img",
            (680, 630, 171, 101),
            scale_x,
            scale_y,
        )

    def _set_stacking_order(self) -> None:
        """Raise interactive widgets above their colored backgrounds."""
        widgets = (
            self.backg_3,
            self.backg_2,
            self.backg_1,
            self.w_tittle,
            self.save_button,
            self.dafault_img,
            self.default_descrip,
            self.specific_descrip,
            self.specific_img,
            self.local_img,
            self.default_check,
            self.specific_check,
            self.local_check,
            self.local_descrip,
            self.backg_4,
            self.extrapolation_check,
            self.extra_label,
            self.extrapolation_descrip,
            self.extra_img,
        )
        for widget in widgets:
            widget.raise_()

    def select_method(self) -> None:
        """Open the configuration dialog for the selected workflow."""
        checked_count = sum(
            checkbox.isChecked()
            for checkbox in (
                self.default_check,
                self.specific_check,
                self.local_check,
                self.extrapolation_check,
            )
        )

        if checked_count != 1:
            message = (
                "Please select one inspection method."
                if checked_count == 0
                else "You can only select one method at a time."
            )
            QMessageBox.warning(self, "Selection Warning", message)
            return

        if self.default_check.isChecked():
            self._open_polygon_dialog()
        elif self.specific_check.isChecked():
            self._open_specific_dialog()
        elif self.local_check.isChecked():
            self._open_local_dialog()
        else:
            self._open_extrapolation_dialog()

    def _open_polygon_dialog(self) -> None:
        """Open the polygon configuration dialog and copy its outputs."""
        self.insp_method = 0
        self.polygon_dialog = PolygonSetting(method=self)
        if self.polygon_dialog.exec_() != QDialog.Accepted:
            return

        self.output_polygon = self.polygon_dialog.output_polygon
        self.ai_value = self.polygon_dialog.ai_value
        self.accept()

    def _open_specific_dialog(self) -> None:
        """Open the specific-coordinate dialog and copy its outputs."""
        self.insp_method = 1
        self.specific_dialog = SpecificLocationSetting(method=self)
        if self.specific_dialog.exec_() != QDialog.Accepted:
            return

        self.data_specific = self.specific_dialog.df
        self.ai_value = self.specific_dialog.ai_value
        self.img_source = self.specific_dialog.img_source_mode.currentData()
        self.accept()

    def _open_local_dialog(self) -> None:
        """Open the local-image dialog and copy its outputs."""
        self.insp_method = 2
        self.local_dialog = LocalImageSetting(method=self)
        if self.local_dialog.exec_() != QDialog.Accepted:
            return

        self.data_local = self.local_dialog.df
        self.ai_value = self.local_dialog.ai_value
        self.accept()

    def _open_extrapolation_dialog(self) -> None:
        """Open the extrapolation dialog and copy its outputs."""
        self.insp_method = 3
        self.extra_dialog = ExtrapolationOptions(self)
        if self.extra_dialog.exec_() != QDialog.Accepted:
            return

        self.extrapolation_mode = self.extra_dialog.extrapolation_mode
        if self.extrapolation_mode == 2:
            self._copy_knn_settings()
        else:
            self._copy_stratified_settings()
        self.accept()

    def _copy_knn_settings(self) -> None:
        """Copy KNN extrapolation settings from the child dialog."""
        self.info_existing = self.extra_dialog.info_existing
        self.info_pending = self.extra_dialog.info_pending
        self.extrapolation_name = self.extra_dialog.extrapolation_name
        self.coord_reference = self.extra_dialog.coord_reference
        self.use_coord_reference = self.extra_dialog.use_coord_reference
        self.output_path = self.extra_dialog.output_path
        self.k_value = self.extra_dialog.k_value

        if self.use_coord_reference:
            self.knn_dl_saved_path = self.extra_dialog.knn_dl_saved_path
            self.coord_reference_building_feature_path = (
                self.extra_dialog.coord_reference_building_feature_path
            )

    def _copy_stratified_settings(self) -> None:
        """Copy stratified extrapolation settings from the child dialog."""
        self.data_population = self.extra_dialog.data_population
        self.folder_path_new = self.extra_dialog.folder_path_new
        self.initial_fraction = self.extra_dialog.initial_fraction
        self.step_fraction = self.extra_dialog.step_fraction
        self.max_fraction = self.extra_dialog.max_fraction
        self.max_iterations = self.extra_dialog.max_iterations
        self.stability_threshold = self.extra_dialog.stability_threshold
        self.feature_strata = self.extra_dialog.feature_strata
