"""Provide a scrollable PyQt5 help dialog with reference images."""

import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 120
DEFAULT_WINDOW_WIDTH = 400
DEFAULT_WINDOW_HEIGHT = 300
ITEMS_PER_ROW = 3
LOGPIXELSX = 88
MIN_REASONABLE_DPI = 60
MAX_REASONABLE_DPI = 200
GEM_TAXONOMY_URL = "https://taxonomy.openquake.org/terms/"


class HelpDialog(QDialog):
    """Display reference images and descriptions in a scrollable dialog.

    Parameters
    ----------
    image_paths : dict
        Mapping from class labels to image paths or dictionaries containing
        ``path`` and ``description`` entries.
    w_size_width : int
        Final width of the dialog in pixels.
    w_size_height : int
        Final height of the dialog in pixels.
    w_title : str
        Window title.
    img_width : int
        Maximum displayed image width in pixels.
    img_height : int
        Maximum displayed image height in pixels.
    description : str, optional
        General explanatory text displayed above the image grid.
    parent : QWidget, optional
        Parent widget of the dialog.
    main_window : QWidget, optional
        Reference to the main application window.
    """

    def __init__(
        self,
        image_paths,
        w_size_width,
        w_size_height,
        w_title,
        img_width,
        img_height,
        description=None,
        parent=None,
        main_window=None,
    ):
        super().__init__(parent)
        self.main_window = main_window

        scale_factor, font_scale = self._calculate_scale_factors()
        self.setWindowTitle(w_title)
        self.resize(
            int(DEFAULT_WINDOW_WIDTH * scale_factor),
            int(DEFAULT_WINDOW_HEIGHT * scale_factor),
        )

        layout = QVBoxLayout()
        if description:
            layout.addWidget(self._create_general_description(description, font_scale))

        layout.addWidget(
            self._create_image_scroll_area(
                image_paths,
                img_width,
                img_height,
                font_scale,
            )
        )
        layout.addWidget(self._create_reference_link())

        self.setLayout(layout)
        self.resize(w_size_width, w_size_height)

    @staticmethod
    def _calculate_scale_factors():
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1.0, 1.0

        screen_geometry = screen.geometry()
        scale_x = screen_geometry.width() / DESIGN_WIDTH
        scale_y = screen_geometry.height() / DESIGN_HEIGHT
        scale_factor = float(np.sqrt(scale_x * scale_y))
        dpi = HelpDialog._get_screen_dpi(screen)
        font_scale = scale_factor * (DESIGN_DPI / dpi)
        return scale_factor, font_scale

    @staticmethod
    def _get_screen_dpi(screen):
        if sys.platform.startswith("win"):
            import ctypes

            device_context = ctypes.windll.user32.GetDC(0)
            try:
                return ctypes.windll.gdi32.GetDeviceCaps(
                    device_context,
                    LOGPIXELSX,
                )
            finally:
                ctypes.windll.user32.ReleaseDC(0, device_context)

        dpi = screen.logicalDotsPerInch()
        if not MIN_REASONABLE_DPI <= dpi <= MAX_REASONABLE_DPI:
            dpi = screen.physicalDotsPerInch()
        return dpi or DESIGN_DPI

    @staticmethod
    def _create_general_description(description, font_scale):
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        font = QFont()
        font.setPointSize(max(1, int(10 * font_scale)))
        description_label.setFont(font)
        description_label.setStyleSheet("padding: 6px; margin-bottom: 8px;")
        return description_label

    def _create_image_scroll_area(
        self,
        image_paths,
        img_width,
        img_height,
        font_scale,
    ):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        grid = QGridLayout()

        for index, (label, item) in enumerate(image_paths.items()):
            path, item_description = self._parse_image_item(item)
            cell = self._create_image_cell(
                label,
                path,
                item_description,
                img_width,
                img_height,
                font_scale,
            )
            row, column = divmod(index, ITEMS_PER_ROW)
            grid.addWidget(cell, row, column)

        content.setLayout(grid)
        scroll_area.setWidget(content)
        return scroll_area

    @staticmethod
    def _parse_image_item(item):
        if isinstance(item, dict):
            return item.get("path", ""), item.get("description", "")
        return item, ""

    @staticmethod
    def _create_image_cell(
        label,
        path,
        description,
        img_width,
        img_height,
        font_scale,
    ):
        image_label = QLabel()
        pixmap = QPixmap(path).scaled(
            img_width,
            img_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        text_label = QLabel(label)
        text_label.setAlignment(Qt.AlignCenter)
        label_font = QFont()
        label_font.setBold(True)
        text_label.setFont(label_font)
        text_label.setStyleSheet("color: darkred;")

        cell_layout = QVBoxLayout()
        cell_layout.addWidget(image_label)
        cell_layout.addWidget(text_label)

        if description:
            description_label = QLabel(description)
            description_label.setWordWrap(True)
            description_label.setAlignment(Qt.AlignCenter)
            description_font = QFont()
            description_font.setPointSize(max(1, int(9 * font_scale)))
            description_label.setFont(description_font)
            description_label.setStyleSheet(
                "color: black; padding-left: 4px; padding-right: 4px;"
            )
            cell_layout.addWidget(description_label)

        cell = QWidget()
        cell.setLayout(cell_layout)
        return cell

    @staticmethod
    def _create_reference_link():
        link_label = QLabel()
        link_label.setText(
            '<span style="font-style: italic; color: darkred; '
            'font-size: 16px;">For more information:</span> '
            f'<a href="{GEM_TAXONOMY_URL}" style="font-size: 16px;">'
            f"{GEM_TAXONOMY_URL}</a>"
        )
        link_label.setOpenExternalLinks(True)
        link_label.setAlignment(Qt.AlignCenter)
        link_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        return link_label
