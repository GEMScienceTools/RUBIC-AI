from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import os
from shapely.geometry import Point
import geopandas as gpd

class SpecificLocationSetting(QtWidgets.QDialog):
    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method

        # Screen scaling
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        sf_x = screen_geometry.width() / 1920
        sf_y = screen_geometry.height() / 1080

        self.setWindowTitle("Specific Location Method Input")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(643 * sf_x), int(218 * sf_y))

        self.coord_frame = QtWidgets.QWidget(self)

        # Background
        self.backg_4 = QtWidgets.QLabel(self.coord_frame)
        self.backg_4.setGeometry(QtCore.QRect(int(10 * sf_x), int(9 * sf_y), int(621 * sf_x), int(161 * sf_y)))
        self.backg_4.setStyleSheet("background-color: rgb(212, 206, 255);")

        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))

        # Title
        self.specific_label = QtWidgets.QLabel(self.coord_frame)
        self.specific_label.setGeometry(QtCore.QRect(int(190 * sf_x), int(9 * sf_y), int(301 * sf_x), int(21 * sf_y)))
        title_font = QtGui.QFont()
        title_font.setPointSize(int(10 * sf_x))
        title_font.setBold(True)
        title_font.setItalic(True)
        title_font.setUnderline(True)
        self.specific_label.setFont(title_font)
        self.specific_label.setText("Specific Locations Method Input")

        # Output name
        self.output_label_specific = QtWidgets.QLabel(self.coord_frame)
        self.output_label_specific.setGeometry(QtCore.QRect(int(20 * sf_x), int(39 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        output_font = QtGui.QFont()
        output_font.setPointSize(int(10 * sf_x))
        output_font.setBold(True)
        self.output_label_specific.setFont(output_font)
        self.output_label_specific.setText("Output name:")

        self.output_specific = QtWidgets.QLineEdit(self.coord_frame)
        self.output_specific.setGeometry(QtCore.QRect(int(160 * sf_x), int(39 * sf_y), int(271 * sf_x), int(31 * sf_y)))
        self.output_specific.setFont(font)
        self.output_specific.setText("specific_location")

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
        self.csv_button_specific.setText("Upload building locations")
        self.csv_button_specific.clicked.connect(self.upload_csv)

        self.specific_path = QtWidgets.QLabel(self.coord_frame)
        self.specific_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(130 * sf_y), int(321 * sf_x), int(21 * sf_y)))
        self.specific_path.setFont(font)
        self.specific_path.setText("filename.csv")

        # Buttons
        bold_font = QtGui.QFont()
        bold_font.setPointSize(int(10 * sf_x))
        bold_font.setBold(True)

        self.load_data_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_data_button.setGeometry(QtCore.QRect(int(100 * sf_x), int(180 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.load_data_button.setFont(bold_font)
        self.load_data_button.setText("Load data")
        self.load_data_button.clicked.connect(self.save_coordinates)

        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(320 * sf_x), int(180 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(bold_font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.building_sample)

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
            except Exception as e:
                self.specific_path.setText(f"Error: {str(e)}")
                QMessageBox.warning(self, "Input Error", "Invalid file selected or parsing error.")
        else:
            self.specific_path.setText("No file selected.")
            QMessageBox.warning(self, "Input Error", "No file selected.")

    def save_coordinates(self):
        output_gpkg = self.method.output_folder_value+"/"+self.output_specific.text()+".gpkg"    
        # Check if the GeoPackage file already exists
        if os.path.exists(output_gpkg):
            os.remove(output_gpkg)  # Delete the file to ensure only one layer is created
        # Load the CSV file
        try:
            df = self.df
            required_columns = ['ID', 'latitude', 'longitude']
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
        
        

    def building_sample(self):
        try:
            if self.population == True:
                self.method.specific_output_name = self.output_specific
                self.accept()
        except:
            QMessageBox.warning(self, "Input Error", "Please load the data first using **Load Data**")


