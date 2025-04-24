from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QDialog, QGridLayout, QScrollArea
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class HelpDialog(QDialog):

    def __init__(self, image_paths, w_size_width, w_size_height, w_title, img_width, img_height, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window  # Reference to the main window (GUIInterface)
        
        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080

        # Window Title
        self.setWindowTitle(w_title)
        self.resize(400*sf_x, 300*sf_y)
        
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout()

        row, col = 0, 0
        for i, (label, path) in enumerate(image_paths.items()):
            # Image
            pixmap = QPixmap(path).scaled(img_width, img_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)

            # Label (bold dark red)
            text_label = QLabel(label)
            text_label.setAlignment(Qt.AlignCenter)
            font = QFont()
            font.setBold(True)
            text_label.setFont(font)
            text_label.setStyleSheet("color: darkred;")

            # Layout for image + label
            cell_layout = QVBoxLayout()
            cell_layout.addWidget(img_label)
            cell_layout.addWidget(text_label)

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

