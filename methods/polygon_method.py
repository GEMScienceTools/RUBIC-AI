from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
from geopy.geocoders import Nominatim

from methods.gui_gis import GUI_geofiles

class PolygonSetting(QtWidgets.QDialog):
    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method

        # Screen scaling
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        sf_x = screen_geometry.width() / 1920
        sf_y = screen_geometry.height() / 1080

        self.setWindowTitle("Polygon Method Input")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(618 * sf_x), int(314 * sf_y))

        self.coord_frame = QtWidgets.QWidget(self)

        # Background
        self.backg_1 = QtWidgets.QLabel(self.coord_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(601 * sf_x), int(251 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(255, 224, 185);")

        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))

        # Title
        self.polygon_label = QtWidgets.QLabel(self.coord_frame)
        self.polygon_label.setGeometry(QtCore.QRect(int(200 * sf_x), int(20 * sf_y), int(211 * sf_x), int(21 * sf_y)))
        font_title = QtGui.QFont()
        font_title.setPointSize(int(10 * sf_x))
        font_title.setBold(True)
        font_title.setItalic(True)
        font_title.setUnderline(True)
        self.polygon_label.setFont(font_title)
        self.polygon_label.setText("Polygon Method Input")

        # Output name
        self.output_label_polygon = QtWidgets.QLabel(self.coord_frame)
        self.output_label_polygon.setGeometry(QtCore.QRect(int(20 * sf_x), int(50 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font.setBold(True)
        self.output_label_polygon.setFont(font)
        self.output_label_polygon.setText("Output name:")

        self.output_polygon = QtWidgets.QLineEdit(self.coord_frame)
        self.output_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(50 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font.setBold(False)
        self.output_polygon.setFont(font)
        self.output_polygon.setText("polygon_building")

        # Output folder
        self.path_out_folder_bt = QtWidgets.QPushButton(self.coord_frame)
        self.path_out_folder_bt.setGeometry(QtCore.QRect(int(20 * sf_x), int(95 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.path_out_folder_bt.setFont(font)
        self.path_out_folder_bt.setText("Select output folder")
        self.path_out_folder_bt.clicked.connect(self.select_output_folder)

        self.output_folder_value = QtWidgets.QLabel(self.coord_frame)
        self.output_folder_value.setGeometry(QtCore.QRect(int(270 * sf_x), int(100 * sf_y), int(321 * sf_x), int(21 * sf_y)))
        self.output_folder_value.setFont(font)
        self.output_folder_value.setText("path/where/you/want/to/save/your/results")

        # N° buildings
        self.n_building = QtWidgets.QLabel(self.coord_frame)
        self.n_building.setGeometry(QtCore.QRect(int(20 * sf_x), int(130 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font.setBold(True)
        self.n_building.setFont(font)
        self.n_building.setText("N° buildings:")

        self.building_value_polygon = QtWidgets.QLabel(self.coord_frame)
        self.building_value_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(130 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font.setBold(False)
        self.building_value_polygon.setFont(font)
        self.building_value_polygon.setText("0000")

        # Sample size
        self.sample_size = QtWidgets.QLabel(self.coord_frame)
        self.sample_size.setGeometry(QtCore.QRect(int(20 * sf_x), int(170 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font.setBold(True)
        self.sample_size.setFont(font)
        self.sample_size.setText("Sample size:")

        self.sample_size_polygon = QtWidgets.QLineEdit(self.coord_frame)
        self.sample_size_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(170 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font.setBold(False)
        self.sample_size_polygon.setFont(font)
        self.sample_size_polygon.setText("10")

        # CSV upload
        self.csv_button_polygon = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_polygon.setGeometry(QtCore.QRect(int(20 * sf_x), int(215 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.csv_button_polygon.setFont(font)
        self.csv_button_polygon.setText("Upload file with coordinates")
        self.csv_button_polygon.clicked.connect(self.upload_csv)

        self.polygon_path = QtWidgets.QLabel(self.coord_frame)
        self.polygon_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(220 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        self.polygon_path.setFont(font)
        self.polygon_path.setText("filename.csv")

        # Buttons
        font_bold = QtGui.QFont()
        font_bold.setPointSize(int(10 * sf_x))
        font_bold.setBold(True)

        self.load_data_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_data_button.setGeometry(QtCore.QRect(int(100 * sf_x), int(270 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.load_data_button.setFont(font_bold)
        self.load_data_button.setText("Load data")
        self.load_data_button.clicked.connect(self.save_coordinates)
        self.load_data_button.clicked.connect(self.building_polulation)

        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(320 * sf_x), int(270 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(font_bold)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.building_sample)

        # Raise
        self.backg_1.raise_()
        self.polygon_label.raise_()
        self.output_label_polygon.raise_()
        self.output_polygon.raise_()
        self.path_out_folder_bt.raise_()
        self.output_folder_value.raise_()
        self.n_building.raise_()
        self.building_value_polygon.raise_()
        self.sample_size.raise_()
        self.sample_size_polygon.raise_()
        self.csv_button_polygon.raise_()
        self.polygon_path.raise_()
        self.load_data_button.raise_()
        self.save_button.raise_()

    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if self.method:
            self.method.output_folder_value = folder_path
        if folder_path:
            folder_display = os.path.basename(folder_path)
            self.output_folder_value.setText(folder_display)

    def upload_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.polygon_path.setText(os.path.basename(file_path))
                if self.method:
                    self.method.file_polygon_csv = file_path
            except Exception as e:
                self.polygon_path.setText(f"Error: {str(e)}")
                QMessageBox.warning(self, "Input Error", f"Error loading CSV:\n{e}")
                
            
            try:
                df = self.df
                required_columns = ['ID', 'latitude', 'longitude']
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    QMessageBox.warning(self, "Missing Columns", f"Required columns missing: {', '.join(missing)}")
                    return      
                QMessageBox.information(self, "Success", "Done! Please click load data button.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load file:\n\n{str(e)}")
                
        else:
            self.polygon_path.setText("No file selected.")
            QMessageBox.warning(self, "Input Error", "No file selected.")
            

    def save_coordinates(self):
        try:
            lat = self.df.iloc[0,1]
            lon = self.df.iloc[0,2]
            
            geolocator = Nominatim(user_agent="city_name_locator")
            location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
            
            if location and 'address' in location.raw:
                address = location.raw['address']
                self.city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                self.country = address.get('country', 'Unknown')
                
                self.method.city =self.city
                self.method.country =self.country
                
                self.city_name_manual = self.city+"_"+self.country
                
            output_gpkg = os.path.join(self.method.output_folder_value, f"{self.city_name_manual}_boundary.gpkg")
            self.boundary_path = output_gpkg
            # Remove existing file
            if os.path.exists(output_gpkg):
                os.remove(output_gpkg)
    
            # Use uploaded CSV file
            df = self.df
            if "latitude" not in df.columns or "longitude" not in df.columns:
                QMessageBox.warning(self, "Input Error",
                                    "The CSV file must have 'latitude' and 'longitude' columns.")
                return

            coordinates = list(zip(df["longitude"], df["latitude"]))
            polygon = Polygon(coordinates)
            gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:4326")
            gdf.to_file(output_gpkg, driver="GPKG", layer="polygon_layer")

            QMessageBox.information(
                        self,
                        "Success",
                        f"GeoPackage successfully saved to:\n{output_gpkg}\n\n⚠️ Please verify the sample size before continuing.\n"
                        "The sample size should be smaller than the total number of buildings."
                    )
    
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not generate GPKG:\n{str(e)}")

    ############ Building population ################  
    def building_polulation(self):
        self.population = GUI_geofiles.download_building_footprints(self)
        self.building_value_polygon.setText(str(self.population)) 

        
    def building_sample(self):

        GUI_geofiles.extract_random_subset(self , self.sample_size_polygon.text())
        
        GUI_geofiles.create_centroid_layer(self)  
        
        self.method.output_polygon = self.output_polygon
        self.accept()
