import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtGui import QGuiApplication

from methods.polygon_method import PolygonSetting
from methods.specific_locations_method import SpecificLocationSetting
from methods.local_images_method import LocalImageSetting 
from methods.extrapolation_options import ExtrapolationOptions
from methods.extrapolation_options import ExtrapolationOptions

class InspectionSetting(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080
        sf_factor = np.sqrt(sf_x**2 * sf_y**2)

        try:
            # Smart scaling for Windows
            import ctypes
            # Reference DPI for 100% scaling
            LOGPIXELSX = 88
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
            ctypes.windll.user32.ReleaseDC(0, hdc)
            scale = 1.25/(dpi / 96)  # 96 DPI is 100%
        except:
            # Smart scaling for MacOS
            scale = QGuiApplication.primaryScreen().devicePixelRatio()
            scale = scale * 0.75
            print(f"MacOS scale factor: {scale}")

        # Scale the GUI based on resolution
        sf_x_font = sf_factor * scale

        # Window Title
        self.setWindowTitle("Select Inspection Method")
        self.resize(int(808*sf_x), int(800*sf_y))
        
        # Main widget
        self.method_frame = QtWidgets.QWidget(self)
        # Title
        self.w_tittle = QtWidgets.QLabel(self.method_frame)
        self.w_tittle.setGeometry(QtCore.QRect(int(270 * sf_x), int(0 * sf_y), int(301 * sf_x), int(41 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(12 * sf_x_font))
        font.setBold(True)
        font.setWeight(75)
        self.w_tittle.setFont(font)
        self.w_tittle.setObjectName("w_tittle")
        self.w_tittle.setText("Setting Inspection Method")  # Set text directly
        
        # Save Button
        self.save_button = QtWidgets.QPushButton(self.method_frame)
        self.save_button.setGeometry(QtCore.QRect(int(310 * sf_x), int(760 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x_font))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.setText("Save and continue")  # Set text directly
        self.save_button.clicked.connect(self.select_method)
        
        # Background polygon
        self.backg_1 = QtWidgets.QLabel(self.method_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(40 * sf_y), int(791 * sf_x), int(171 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(255, 224, 185);")
        self.backg_1.setText("")
        self.backg_1.setObjectName("backg_1")
        
        # Default (Polygon method) image
        self.dafault_img = QtWidgets.QLabel(self.method_frame)
        self.dafault_img.setGeometry(QtCore.QRect(int(600 * sf_x), int(50 * sf_y), int(181 * sf_x), int(141 * sf_y)))
        self.dafault_img.setObjectName("dafault_img")
        self.dafault_img.setPixmap(QtGui.QPixmap("help_img/default_buildings.png"))
        self.dafault_img.setScaledContents(True)
        
        # Default (Polygon method) description
        self.default_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.default_descrip.setGeometry(QtCore.QRect(int(230 * sf_x), int(50 * sf_y), int(351 * sf_x), int(151 * sf_y)))
        self.default_descrip.setObjectName("default_descrip")

        # Reusable CSS style block for HTML descriptions
        css_style = "<style type=\"text/css\">p, li { white-space: pre-wrap; }</style>"

        font_size = 10 * sf_x_font  # or any base value that looks right
        default_html = (
            f"<html><head><meta name=\"qrichtext\" content=\"1\" />"
            f"{css_style}</head>"
            f"<body style=\" font-family:'MS Shell Dlg 2'; font-size:{font_size:.1f}pt; font-weight:400; font-style:normal;\">"
            "<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; "
            "-qt-block-indent:0; text-indent:0px;\">"
            "<span style=\" font-size:inherit;\">"
            "Creates a polygon using its vertices coordinates (which must be uploaded in either clockwise or counterclockwise order) "
            "and performs virtual inspections based on the sample size or the entire building population within the polygon"
            "</span></p></body></html>"
            )
        self.default_descrip.setHtml(default_html)

        # Background specific
        self.backg_2 = QtWidgets.QLabel(self.method_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(220 * sf_y), int(791 * sf_x), int(171 * sf_y)))
        self.backg_2.setStyleSheet("background-color: rgb(215, 213, 255)")
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        
        # Specific method description
        self.specific_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.specific_descrip.setGeometry(QtCore.QRect(int(230 * sf_x), int(230 * sf_y), int(351 * sf_x), int(151 * sf_y)))
        self.specific_descrip.setObjectName("specific_descrip")

        # Compute adaptive font size
        font_size = 10 * sf_x_font  # You can adjust 7.8 as your base size

        # Construct the HTML with dynamic font size
        specific_html = (
            f"<html><head><meta name=\"qrichtext\" content=\"1\" />"
            f"{css_style}</head>"
            f"<body style=\" font-family:'MS Shell Dlg 2'; font-size:{font_size:.1f}pt; font-weight:400; font-style:normal;\">"
            "<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; "
            "-qt-block-indent:0; text-indent:0px;\">"
            "<span style=\" font-size:inherit;\">"
            "Allows the user to upload a set of specific building locations (buildings of interest), which may be located in multiple geographic areas, such as different cities or countries. "
            "Users can upload a CSV file containing all pairs of coordinates for the buildings they wish to inspect."
            "This enables users to upload a CSV file containing all the pairs of coordinates for the buildings of interest."
            "</span></p></body></html>"
            )
        self.specific_descrip.setHtml(specific_html)

        # Specific image
        self.specific_img = QtWidgets.QLabel(self.method_frame)
        self.specific_img.setGeometry(QtCore.QRect(int(600 * sf_x), int(230 * sf_y), int(181 * sf_x), int(141 * sf_y)))
        self.specific_img.setObjectName("specific_img")
        self.specific_img.setPixmap(QtGui.QPixmap("help_img/specific_buildings.png"))
        self.specific_img.setScaledContents(True)
        
        # Local image
        self.local_img = QtWidgets.QLabel(self.method_frame)
        self.local_img.setGeometry(QtCore.QRect(int(600 * sf_x), int(430 * sf_y), int(181 * sf_x), int(141 * sf_y)))
        self.local_img.setObjectName("local_img")
        self.local_img.setPixmap(QtGui.QPixmap("help_img/local_buildings.png"))
        self.local_img.setScaledContents(True)
        
        # Local method description
        self.local_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.local_descrip.setGeometry(QtCore.QRect(int(230 * sf_x), int(410 * sf_y), int(351 * sf_x), int(181 * sf_y)))
        self.local_descrip.setObjectName("local_descrip")

        # Compute adaptive font size
        font_size = 10 * sf_x_font  # Base font size scaled

        # Construct the HTML with dynamic font size
        local_html = (
            f"<html><head><meta name=\"qrichtext\" content=\"1\" />"
            f"{css_style}</head>"
            f"<body style=\" font-family:'MS Shell Dlg 2'; font-size:{font_size:.1f}pt; font-weight:400; font-style:normal;\">"
            "<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; "
            "-qt-block-indent:0; text-indent:0px;\">"
            "<span style=\" font-size:inherit;\">"
            "Users should provide a folder containing all the images, as well as a CSV file with the metadata. The CSV file must include columns named 'image_file', 'latitude', and 'longitude', where 'image_file' matches the image filename and 'latitude'/'longitude' specify the location. "
            "This is useful in cases where users might not want to use Google Street View, or where GSV might simply not exist for the region of interest."
            "</span></p></body></html>"
        )
        # Background local
        self.backg_3 = QtWidgets.QLabel(self.method_frame)
        self.backg_3.setGeometry(QtCore.QRect(int(10 * sf_x), int(400 * sf_y), int(791 * sf_x), int(201 * sf_y)))
        self.backg_3.setStyleSheet("background-color: rgb(255, 253, 187);")
        self.backg_3.setText("")
        self.backg_3.setObjectName("backg_3")
        
        self.local_descrip.setHtml(local_html)
        
        # Polygon method checkbox
        self.default_check = QtWidgets.QCheckBox(self.method_frame)
        self.default_check.setGeometry(QtCore.QRect(int(20 * sf_x), int(120 * sf_y), int(171 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x_font))
        font.setBold(True)
        font.setWeight(75)
        self.default_check.setFont(font)
        self.default_check.setObjectName("default_check")
        self.default_check.setText("Polygon method")  # Set text directly
        
        # Specific method checkbox
        self.specific_check = QtWidgets.QCheckBox(self.method_frame)
        self.specific_check.setGeometry(QtCore.QRect(int(20 * sf_x), int(300 * sf_y), int(201 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x_font))
        font.setBold(True)
        font.setWeight(75)
        self.specific_check.setFont(font)
        self.specific_check.setObjectName("specific_check")
        self.specific_check.setText("Specific coordinates")  # Set text directly
        
        # Local method checkbox
        self.local_check = QtWidgets.QCheckBox(self.method_frame)
        self.local_check.setGeometry(QtCore.QRect(int(30 * sf_x), int(480 * sf_y), int(171 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x_font))
        font.setBold(True)
        font.setWeight(75)
        self.local_check.setFont(font)
        self.local_check.setObjectName("local_check")
        self.local_check.setText("Local images")  # Set text directly
        
        # Extrapolation method
        self.backg_4 = QtWidgets.QLabel(self.method_frame)
        self.backg_4.setGeometry(QtCore.QRect(int(10 * sf_x), int(610 * sf_y), int(791 * sf_x), int(141 * sf_y)))
        self.backg_4.setStyleSheet("background-color: rgb(157, 218, 255);")
        self.backg_4.setText("")
        self.backg_4.setObjectName("backg_4")
        
        self.extrapolation_check = QtWidgets.QCheckBox(self.method_frame)
        self.extrapolation_check.setGeometry(QtCore.QRect(int(30 * sf_x), int(650 * sf_y), int(221 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x_font))
        font.setBold(True)
        font.setWeight(75)
        self.extrapolation_check.setFont(font)
        self.extrapolation_check.setObjectName("extrapolation_check")
        self.extrapolation_check.setText("Neighbor")
        
        self.extra_label = QtWidgets.QLabel(self.method_frame)
        self.extra_label.setGeometry(QtCore.QRect(int(50 * sf_x), int(670 * sf_y), int(121 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x_font))
        font.setBold(True)
        font.setWeight(75)
        self.extra_label.setFont(font)
        self.extra_label.setObjectName("extra_label")
        self.extra_label.setText("extrapolation")
        
        self.extrapolation_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.extrapolation_descrip.setGeometry(QtCore.QRect(int(230 * sf_x), int(620 * sf_y), int(351 * sf_x), int(121 * sf_y)))
        self.extrapolation_descrip.setObjectName("extrapolation_descrip")

        # Compute adaptive font size
        font_size = 10 * sf_x_font  # Adjust base size if needed

        # Construct the HTML with dynamic font size
        extrapolation_html = (
            f"<html><head><meta name=\"qrichtext\" content=\"1\" />"
            f"{css_style}</head>"
            f"<body style=\" font-family:'MS Shell Dlg 2'; font-size:{font_size:.1f}pt; font-weight:400; font-style:normal;\">"
            "<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; "
            "-qt-block-indent:0; text-indent:0px;\">"
            "<span style=\" font-size:inherit;\">"
            "User should provide a file with the locations of buildings without available images, as well as a file containing building features. "
            "Using this information, the tool will extrapolate the most probable features for these buildings."
            "</span></p></body></html>"
        )
        self.extrapolation_descrip.setHtml(extrapolation_html)
        
        # Extrapolation image
        self.extra_img = QtWidgets.QLabel(self.method_frame)
        self.extra_img.setGeometry(QtCore.QRect(int(610*sf_x), int(630*sf_y), int(171*sf_x), int(101*sf_y)))
        self.extra_img.setObjectName("extra_img")
        self.extra_img.setPixmap(QtGui.QPixmap("help_img/extrapolation.jpg"))
        self.extra_img.setScaledContents(True)
        
        """ GEM icon GUI elements """
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        
        # Stacking order
        self.backg_3.raise_()
        self.backg_2.raise_()
        self.backg_1.raise_()
        self.w_tittle.raise_()
        self.save_button.raise_()
        self.dafault_img.raise_()
        self.default_descrip.raise_()
        self.specific_descrip.raise_()
        self.specific_img.raise_()
        self.local_img.raise_()
        self.default_check.raise_()
        self.specific_check.raise_()
        self.local_check.raise_()
        self.local_descrip.raise_()
        self.backg_4.raise_()
        self.extrapolation_check.raise_()
        self.extra_label.raise_()
        self.extrapolation_descrip.raise_()
        self.extra_img.raise_()


    def select_method(self):
        """
        Select an inspection method based on user checkbox selection.
    
        This method ensures that only one inspection method is selected at a time. 
        If multiple checkboxes are selected, it displays a warning message and prevents 
        selection. If a single checkbox is checked, it assigns the corresponding inspection 
        method to `insp_method` in the main window and confirms the selection.
    
        Effects:
            - Displays a warning if more than one checkbox is selected.
            - Assigns the corresponding inspection method:
                - `0` for polygon method (`default_check`).
                - `1` for specific method (`specific_check`).
                - `2` for local method (`local_check`).
                - `3` for local method (`extrapolation_check`).
            - Calls `accept()` to confirm the selection.
    
        Notes:
            - Only one checkbox can be selected at a time.
        """

        # Check how many checkboxes are checked
        checked_count = sum([self.default_check.isChecked(), 
                             self.specific_check.isChecked(), 
                             self.local_check.isChecked(),
                             self.extrapolation_check.isChecked()])
        
        if checked_count > 1:
            # Show a warning if more than one checkbox is checked
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("You can only select one method at a time.")
            msg.setWindowTitle("Selection Warning")
            msg.exec_()
        else:
            # Print the selected method if only one checkbox is checked
            if self.default_check.isChecked():
                self.insp_method = 0
                # QMessageBox.warning(self, "Usage mode error", "This option is currently unavailable. Please select either 'Local Images' or 'Neighbor Extrapolation'.")
                self.accept()
                self.polygon_dialog = PolygonSetting(method=self)  # Pass main window reference if needed
                self.polygon_dialog.exec_()
                self.output_polygon = self.polygon_dialog.output_polygon
                self.ai_value = self.polygon_dialog.ai_value 
                self.city = self.polygon_dialog.city 
                self.country = self.polygon_dialog.country 
                
            if self.specific_check.isChecked():
                self.insp_method = 1
                self.accept()
                self.specific_dialog = SpecificLocationSetting(method=self)  # Pass main window reference if needed
                self.specific_dialog.exec_()
                self.data_specific = self.specific_dialog.df 
                self.ai_value = self.specific_dialog.ai_value 
                
            if self.local_check.isChecked():
                self.insp_method = 2
                self.accept()
                self.local_dialog = LocalImageSetting(method=self)  # Pass main window reference if needed
                self.local_dialog.exec_()
                self.data_local = self.local_dialog.df
                self.ai_value = self.local_dialog.ai_value 
                
            if self.extrapolation_check.isChecked():
                self.insp_method = 3
                self.accept()
                self.extra_dialog = ExtrapolationOptions()  # Pass main window reference if needed
                self.extra_dialog.exec_()
                self.extrapolation_mode = self.extra_dialog.extrapolation_mode
                # KNN method
                if self.extrapolation_mode == 2:
                    self.info_existing = self.extra_dialog.info_existing
                    self.info_pending = self.extra_dialog.info_pending
                    self.extrapolation_name = self.extra_dialog.extrapolation_name
                    self.coord_reference = self.extra_dialog.coord_reference
                    self.output_path = self.extra_dialog.output_path
                    self.k_value = self.extra_dialog.k_value
                else:
                    # Stratified method
                    self.data_population = self.extra_dialog.data_population
                    self.initial_fraction = self.extra_dialog.initial_fraction
                    self.step_fraction = self.extra_dialog.step_fraction
                    self.max_fraction = self.extra_dialog.max_fraction
                    self.max_iterations = self.extra_dialog.max_iterations
                    self.stability_threshold = self.extra_dialog.stability_threshold
                    
                    self.feature_strata = self.extra_dialog.feature_strata
                
                try:
                    self.knn_dl_saved_path = self.extra_dialog.knn_dl_saved_path
                    self.coord_reference_building_feature_path = self.extra_dialog.coord_reference_building_feature_path
                except:
                    pass
                
        
        
 