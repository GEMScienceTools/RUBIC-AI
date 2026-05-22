"""
help_window.py
==============
This module provides a PyQt5-based, scrollable help dialog that displays labeled 
reference images and includes access to the GEM documentation website.
"""

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QDialog, QGridLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import sys
import numpy as np

class HelpDialog(QDialog):
    
    def __init__(self, image_paths, w_size_width, w_size_height, w_title, img_width, img_height, description=None, parent=None, main_window=None):
    # def __init__(self, image_paths, w_size_width, w_size_height, w_title, img_width, img_height, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window  # Reference to the main window (GUIInterface)
        
        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

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


        # Window Title
        self.setWindowTitle(w_title)
        self.resize(int(400*sf_x), int(300*sf_y))
        
        layout = QVBoxLayout()
        
        if description:
            description_label = QLabel(description)
            description_label.setWordWrap(True)
            description_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
            font = QFont()
            font.setPointSize(10)
            description_label.setFont(font)
        
            description_label.setStyleSheet(
                "padding: 6px; "
                "margin-bottom: 8px;"
            )
        
            layout.addWidget(description_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout()

        row, col = 0, 0
        for i, (label, item) in enumerate(image_paths.items()):
            # Accept both formats:
            # 1) "Concrete": "help_img/concrete.jpg"
            # 2) "Concrete": {"path": "help_img/concrete.jpg", "description": "..."}
            if isinstance(item, dict):
                path = item.get("path", "")
                description = item.get("description", "")
            else:
                path = item
                description = ""
        
            # Image
            pixmap = QPixmap(path).scaled(
                img_width,
                img_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
        
            # Class label
            text_label = QLabel(label)
            text_label.setAlignment(Qt.AlignCenter)
        
            font = QFont()
            font.setBold(True)
            text_label.setFont(font)
            text_label.setStyleSheet("color: darkred;")
        
            # Class description
            description_label = QLabel(description)
            description_label.setWordWrap(True)
            description_label.setAlignment(Qt.AlignCenter)
        
            description_font = QFont()
            description_font.setPointSize(9)
            description_label.setFont(description_font)
        
            description_label.setStyleSheet(
                "color: black; "
                "padding-left: 4px; "
                "padding-right: 4px;"
            )
        
            # Layout for image + label + description
            cell_layout = QVBoxLayout()
            cell_layout.addWidget(img_label)
            cell_layout.addWidget(text_label)
        
            if description:
                cell_layout.addWidget(description_label)

            cell = QWidget()
            cell.setLayout(cell_layout)

            grid.addWidget(cell, row, col)

            col += 1
            if col == 3:  # 3 items per row
                row += 1
                col = 0

        content.setLayout(grid)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Add hyperlink
        link_label = QLabel()
        link_label.setText(
            '<span style="font-style: italic; color: darkred; font-size: 16px;">For more information:</span> '
            '<a href="https://taxonomy.openquake.org/terms/" style="font-size: 16px;">https://taxonomy.openquake.org/terms/</a>'
        )
        link_label.setOpenExternalLinks(True)
        link_label.setAlignment(Qt.AlignCenter)
        link_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        
        layout.addWidget(link_label)
        
        self.setLayout(layout)
        self.resize(w_size_width, w_size_height)

