from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import os
import sys
import numpy as np
from shapely.geometry import Point
import geopandas as gpd

class SpecificLocationSetting(QtWidgets.QDialog):
    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method

        ## Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        #
        DESIGN_WIDTH = 1920
        DESIGN_HEIGHT = 1080
        DESIGN_DPI = 96 * 1.25  # 125% Windows baseline -> 120 DPI
        
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

        # Normalize to your design environment (Windows @ 125% = 120 DPI)
        # If dpi == 120 => scale_dpi = 1 (your original machine)
        scale_dpi = DESIGN_DPI / dpi
        
        # For geometry: mainly resolution-based
        sf_x = sf_factor
        sf_y = sf_factor
        # Scale the GUI based on resolution
        sf_font = sf_factor * scale_dpi

        self.setWindowTitle("Specific Coordinates Method Input")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(643 * sf_x), int(483 * sf_y))

        self.coord_frame = QtWidgets.QWidget(self)

        # Background
        self.backg_4 = QtWidgets.QLabel(self.coord_frame)
        self.backg_4.setGeometry(QtCore.QRect(int(10 * sf_x), int(9 * sf_y), int(621 * sf_x), int(411 * sf_y)))
        self.backg_4.setStyleSheet("background-color: rgb(212, 206, 255);")

        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))

        # Title
        self.specific_label = QtWidgets.QLabel(self.coord_frame)
        self.specific_label.setGeometry(QtCore.QRect(int(190 * sf_x), int(9 * sf_y), int(301 * sf_x), int(21 * sf_y)))
        title_font = QtGui.QFont()
        title_font.setPointSize(int(10 * sf_font))
        title_font.setBold(True)
        title_font.setItalic(True)
        title_font.setUnderline(True)
        self.specific_label.setFont(title_font)
        self.specific_label.setText("Specific Coordinates Method Input")

        # Output name
        self.output_label_specific = QtWidgets.QLabel(self.coord_frame)
        self.output_label_specific.setGeometry(QtCore.QRect(int(20 * sf_x), int(39 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        output_font = QtGui.QFont()
        output_font.setPointSize(int(10 * sf_font))
        output_font.setBold(True)
        self.output_label_specific.setFont(output_font)
        self.output_label_specific.setText("Output name:")

        self.output_specific = QtWidgets.QLineEdit(self.coord_frame)
        self.output_specific.setGeometry(QtCore.QRect(int(160 * sf_x), int(39 * sf_y), int(271 * sf_x), int(31 * sf_y)))
        self.output_specific.setFont(font)
        self.output_specific.setText("specific_coord")

        # Output folder
        self.path_out_folder_bt = QtWidgets.QPushButton(self.coord_frame)
        self.path_out_folder_bt.setGeometry(QtCore.QRect(int(20 * sf_x), int(80 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.path_out_folder_bt.setFont(font)
        self.path_out_folder_bt.setText("Select output folder")
        self.path_out_folder_bt.clicked.connect(self.select_output_folder)

        self.output_folder_value = QtWidgets.QLabel(self.coord_frame)
        self.output_folder_value.setGeometry(QtCore.QRect(int(270 * sf_x), int(85 * sf_y), int(341 * sf_x), int(21 * sf_y)))
        self.output_folder_value.setFont(font)
        self.output_folder_value.setText("path/where/you/want/to/save/your/results")

        # Upload CSV
        self.csv_button_specific = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_specific.setGeometry(QtCore.QRect(int(20 * sf_x), int(125 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.csv_button_specific.setFont(font)
        self.csv_button_specific.setText("Upload building coordinates")
        self.csv_button_specific.clicked.connect(self.upload_csv)

        self.specific_path = QtWidgets.QLabel(self.coord_frame)
        self.specific_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(130 * sf_y), int(321 * sf_x), int(21 * sf_y)))
        self.specific_path.setFont(font)
        self.specific_path.setText("filename.csv")

        # Buttons
        bold_font = QtGui.QFont()
        bold_font.setPointSize(int(10 * sf_font))
        bold_font.setBold(True)

        self.load_data_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_data_button.setGeometry(QtCore.QRect(int(100 * sf_x), int(430 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.load_data_button.setFont(bold_font)
        self.load_data_button.setText("Load data")
        self.load_data_button.clicked.connect(self.save_coordinates)

        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(340 * sf_x), int(430 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(bold_font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.building_sample)
              
        # Feature Collection Label
        self.feature_collection_label = QtWidgets.QLabel(self.coord_frame)
        self.feature_collection_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(170 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.feature_collection_label.setFont(font)
        self.feature_collection_label.setObjectName("feature_collection_label")
        self.feature_collection_label.setText("Feature collection mode:")
        
        # ComboBox for Collection Mode
        self.collection_mode = QtWidgets.QComboBox(self.coord_frame)
        self.collection_mode.setGeometry(QtCore.QRect(int(250 * sf_x), int(170 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.collection_mode.setFont(font)
        self.collection_mode.setObjectName("collection_mode")
        self.collection_mode.addItem("Manual")
        self.collection_mode.addItem("AI Powered")
        
        # TableWidget
        self.tableWidget = QtWidgets.QTableWidget(self.coord_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(20 * sf_x), int(210 * sf_y), int(601 * sf_x), int(192 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        # Raise stacking
        self.backg_4.raise_()
        self.specific_label.raise_()
        self.output_label_specific.raise_()
        self.output_specific.raise_()
        self.path_out_folder_bt.raise_()
        self.output_folder_value.raise_()
        self.csv_button_specific.raise_()
        self.specific_path.raise_()
        self.load_data_button.raise_()
        self.save_button.raise_()
        self.tableWidget.raise_()
        self.collection_mode.raise_()
        self.feature_collection_label.raise_()
        
    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        self.method.output_folder_value = folder_path
        if folder_path:
            folder_display = os.path.basename(folder_path)
            self.output_folder_value.setText(folder_display)

    def upload_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        self.file_specific_csv = file_path
        self.method.file_specific_csv = file_path
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                display_name = os.path.basename(file_path)
                self.specific_path.setText(display_name)
                self.preview_data()
            except Exception as e:
                self.specific_path.setText(f"Error: {str(e)}")
                QMessageBox.warning(self, "Input Error", "Invalid file selected or parsing error.")
        else:
            self.specific_path.setText("No file selected.")
            QMessageBox.warning(self, "Input Error", "No file selected.")

    def save_coordinates(self):
        try:
            output_gpkg = self.method.output_folder_value+"/"+self.output_specific.text()+".gpkg"    
            # Check if the GeoPackage file already exists
            if os.path.exists(output_gpkg):
                os.remove(output_gpkg)  # Delete the file to ensure only one layer is created
            # Load the CSV file
            try:
                df = self.df
                required_columns = ['id', 'latitude', 'longitude']
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    QMessageBox.warning(self, "Missing Columns", f"Required columns missing: {', '.join(missing)}")
                    return
                
                self.population = True
                
                QMessageBox.information(self, "Success", "Done! Please click save and continue.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load file:\n\n{str(e)}")
            # Create geometries for the points using latitude and longitude
            geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            # Set the coordinate reference system (CRS) to WGS84 (latitude/longitude)
            gdf.set_crs('EPSG:4326', inplace=True)
            # Save the GeoDataFrame to a file, if needed (e.g., to GeoPackage or Shapefile)
            gdf.to_file(output_gpkg, driver='GPKG')  # This saves as GeoPackage
        except:
            QMessageBox.warning(self, "Error input", "Please complete the input information")
        
        
    def building_sample(self):
        try:
            if self.population == True:
                self.method.specific_output_name = self.output_specific
                self.mode_use()
                self.accept()
        except:
            QMessageBox.warning(self, "Input Error", "Please load the data first using **Load Data**")


    def preview_data(self):
        if hasattr(self, 'df') and not self.df.empty:
            preview_df = self.df.head(10)  # Only show first 10 rows
    
            self.tableWidget.clear()
            self.tableWidget.setRowCount(len(preview_df))
            self.tableWidget.setColumnCount(len(preview_df.columns))
            self.tableWidget.setHorizontalHeaderLabels(preview_df.columns)
    
            for row in range(len(preview_df)):
                for column in range(len(preview_df.columns)):
                    value = str(preview_df.iloc[row, column])
                    item = QtWidgets.QTableWidgetItem(value)
                    self.tableWidget.setItem(row, column, item)
    
            self.tableWidget.resizeColumnsToContents()
        else:
            QMessageBox.warning(self, "No Data", "No data available to preview. Please upload a valid CSV first.")

    def mode_use(self):
        if self.collection_mode.currentText() == "Manual":
            self.ai_value = False 
        elif self.collection_mode.currentText() == "AI Powered":
            self.ai_value = True